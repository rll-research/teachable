import numpy
import numpy as np
import torch
import torch.nn.functional as F
from babyai.rl import DictList
from meta_mb.logger import logger
import time


from babyai.rl.algos.base import BaseAlgo


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, acmodel, envs, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, aux_info=None):

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, self.obss_preprocessor, None,
                         aux_info)

        num_frames_per_proc = num_frames_per_proc or 128
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.aux_info = aux_info
        self.single_env = envs[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.acmodel.to(self.device)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        assert self.num_frames_per_proc % self.recurrence == 0

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.lr, (self.beta1, self.beta2),
                                          eps=self.adam_eps)

    def obss_preprocessor(self, obs, device=None):
        obs_arr = np.stack(obs, 0)
        return torch.FloatTensor(obs_arr).to(device)

    def update_parameters(self):
        return self.optimize_policy(None, True)

    def preprocess_samples(self, samples_data):

        del samples_data["avg_reward"]

        b, t, _ = samples_data['actions'].shape
        for k, v in samples_data.items():
            if type(v) is np.ndarray:
                samples_data[k] = torch.FloatTensor(v.reshape((b * t, -1))).to(self.device)
            elif type(v) is dict:
                for k2, v2 in v.items():
                    v[k2] = torch.FloatTensor(v2.reshape(b * t, -1)).to(self.device)
                samples_data[k] = DictList(v)
            else:
                print("can't flatten", k, v.shape)

        return DictList(samples_data)


    def compute_advantage(self, samples_data):  # TODO: move this to wherever we currently compute returns
        values = samples_data['agent_infos']['value']
        batch_size, timesteps = values.shape
        rewards = samples_data['rewards']
        masks = samples_data['mask'][:, :, 0]

        next_value = values[:, -1]  # TODO: Check whether this is an off-by-one error

        advantages = np.zeros((batch_size, timesteps))
        for i in reversed(range(timesteps)):
            next_mask = masks[:, i + 1] if i < timesteps - 1 else np.ones(batch_size)
            next_value = values[:, i + 1] if i < timesteps - 1 else next_value
            next_advantage = advantages[:, i + 1] if i < timesteps - 1 else np.zeros(batch_size)

            delta = rewards[:, i] + self.discount * next_value * next_mask - values[:, i]
            advantages[:, i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
        samples_data['advantages'] = advantages
        return samples_data


    def optimize_policy(self, exps, use_teacher=False):  # TODO: later generalize this to which kinds of teacher should be visible to the agent.
        '''
        exps is a DictList with the following keys ['observations', 'memory', 'mask', 'actions', 'value', 'rewards',
         'advantage', 'returns', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''
        self.acmodel.train()

        model_running_time = 0
        backward_time = 0

        for e in range(self.epochs):
            teacher_max = exps.teacher_action.detach().cpu().numpy()
            orig_actions = exps.action.detach().cpu().numpy()

            # Initialize log values
            log_returnn = []
            log_advantage = []
            log_value_clip = []
            log_policy_clip = []
            log_ratio = []
            log_log_prob = []
            log_sb_value = []

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            num_actions = self.single_env.action_space.n
            log_actions_taken = {}
            log_teacher_actions_taken = {}
            log_teacher_following = {}
            for i in range(num_actions):
                log_actions_taken[i] = []
                log_teacher_actions_taken[i] = []
                log_teacher_following[i] = []
            log_losses = []
            model_calls = 0
            model_samples_calls = 0

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            # Currently we process everything as one batch, but if we ever are running into memory errors
            # we could split this up.
            inds = numpy.arange(0, len(exps.action), self.recurrence)
            for inds in [inds[:-1]]:
                # for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                batch_returnn = 0
                batch_advantage = 0
                batch_value_clip = 0
                batch_policy_clip = 0
                batch_ratio = 0
                batch_log_prob = 0
                batch_sb_value = 0

                # Initialize memory

                memory = exps.memory[inds]

                for i in range(self.recurrence):

                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss
                    model_running = time.time()
                    dist, agent_info = self.acmodel(sb.obs, memory * sb.mask, use_teacher=use_teacher)
                    model_calls += 1
                    model_samples_calls += len(sb.obs)
                    model_running_end = time.time() - model_running
                    model_running_time += model_running_end

                    value = agent_info['value']
                    memory = agent_info['memory']
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surrr1 = ratio * sb.advantage
                    surrr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surrr1, surrr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    batch_entropy -= entropy.item() * self.entropy_coef
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item() * self.value_loss_coef
                    batch_loss += loss

                    batch_returnn += sb.returnn.mean().item()
                    batch_advantage += sb.advantage.mean().item()
                    batch_value_clip += (surr1 - surr2).mean().item()
                    batch_policy_clip += (surrr1 - surrr2).mean().item()
                    batch_ratio += ratio.mean().item()
                    batch_log_prob += sb.log_prob.mean().item()
                    batch_sb_value += sb.value.mean().item()



                    # Update memories for next epoch
                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                batch_returnn /= self.recurrence
                batch_advantage /= self.recurrence
                batch_value_clip /= self.recurrence
                batch_policy_clip /= self.recurrence
                batch_ratio /= self.recurrence
                batch_log_prob /= self.recurrence
                batch_sb_value /= self.recurrence

                # Update actor-critic

                backward_start = time.time()
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                desired_action = sb.teacher_action.int()
                accuracy = np.mean((dist.sample() == desired_action).detach().cpu().numpy())

                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                backward_end = time.time() - backward_start
                backward_time += backward_end

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())

                log_returnn.append(batch_returnn)
                log_advantage.append(batch_advantage)
                log_value_clip.append(batch_value_clip)
                log_policy_clip.append(batch_policy_clip)
                log_ratio.append(batch_ratio)
                log_log_prob.append(batch_log_prob)
                log_sb_value.append(batch_sb_value)

                log_losses.append(batch_loss.item())
                d = dist.sample().detach().cpu().numpy()
                for i in range(num_actions):
                    log_actions_taken[i].append(np.mean(d == i))
                    teacher_i = teacher_max == i
                    if np.sum(teacher_i) > 0:
                        actions_i = orig_actions[teacher_i]
                        log_teacher_following[i].append(np.mean(actions_i == i))
                        log_teacher_actions_taken[i].append(np.mean(teacher_max == i))

            # Log some values
            logs = {}

            # DEBUG LOGS
            logs['TeacherError'] = numpy.mean(exps.env_infos.teacher_error)
            logs['Advantage'] = numpy.mean(log_advantage)
            logs['Ratio'] = numpy.mean(log_ratio)
            logs['PolicyClip'] = numpy.mean(log_policy_clip)

            logs['ValueClip'] = numpy.mean(log_value_clip)
            logs['Returnn'] = numpy.mean(log_returnn)

            logs['LogProb'] = numpy.mean(log_log_prob)
            logs['Returnn'] = numpy.mean(log_sb_value)

            logs['Accuracy'] = accuracy
            logs["Entropy_loss"] = numpy.mean(log_entropies)
            logs["Entropy"] = numpy.mean(log_entropies) / self.entropy_coef
            logs["Value"] = numpy.mean(log_values)
            logs["Policy_loss"] = numpy.mean(log_policy_losses)
            logs["Value_loss"] = numpy.mean(log_value_losses)
            logs["Grad_norm"] = numpy.mean(log_grad_norms)
            logs["Loss"] = numpy.mean(log_losses)
            for i in range(num_actions):
                logs[f'Took{i}'] = np.mean(log_actions_taken[i])
                if len(log_teacher_following[i]) > 0:
                    logs[f'Accuracy{i}'] = np.mean(log_teacher_following[i])
                    logs[f'TeacherTook{i}'] = np.mean(log_teacher_actions_taken[i])

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def advance_curriculum(self):
        self.env.advance_curriculum()
