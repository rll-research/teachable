import numpy as np
import torch
import babyai.utils as utils
from itertools import chain, combinations
import time
import copy
from babyai.rl.utils.dictlist import DictList

class ImitationLearning(object):
    def __init__(self, model, env, args, distill_with_teacher, reward_predictor=False, preprocess_obs=lambda x: x,
                 label_weightings=False, instr_dropout_prob=.5, modify_cc3_steps=None):
        self.args = args
        self.distill_with_teacher = distill_with_teacher
        self.reward_predictor = reward_predictor
        self.preprocess_obs = preprocess_obs
        self.label_weightings = label_weightings
        self.instr_dropout_prob = instr_dropout_prob

        utils.seed(self.args.seed)
        self.env = env

        # Define actor-critic model
        self.policy_dict = model
        for policy in model.values():
            policy.train()
            if torch.cuda.is_available():
                policy.cuda()

        self.optimizer_dict = {
            k: torch.optim.Adam(policy.parameters(), self.args.lr, eps=self.args.optim_eps) for k, policy in
            self.policy_dict.items()
        }
        # self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler_dict = {
            k: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99) for k, optimizer in self.optimizer_dict.items()
        }
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modify_cc3_steps = modify_cc3_steps

    def modify_cc3(self, batch):
        obs = batch.obs
        for i in range(len(obs) - self.modify_cc3_steps):
            future_obs = obs[i + self.modify_cc3_steps]['obs']
            obs[i]['CartesianCorrections'] = future_obs
        batch.obs = obs
        return batch

    def preprocess_batch(self, batch, source):
        if self.modify_cc3_steps is not None:
            batch = self.modify_cc3(batch)
        obss = batch.obs
        if source == 'teacher':
            action_true = batch.teacher_action[:, 0]
        elif source == 'agent':
            action_true = batch.action
        elif source == 'agent_argmax':
            action_true = torch.argmax(batch.action_probs, dim=1)
        elif source == 'agent_probs':
            action_true = batch.action_probs
        if not source == 'agent_probs':
            action_true = torch.tensor(action_true, device=self.device, dtype=torch.long)
        action_teacher = batch.teacher_action[:, 0]
        done = batch.full_done

        # rearrange all demos to be in descending order of length
        inds = np.concatenate([[0], torch.where(done == 1)[0].detach().cpu().numpy() + 1])
        obss_list = []
        action_true_list = []
        action_teacher_list = []
        done_list = []

        for i in range(len(inds) - 1):
            start = inds[i]
            end = inds[i + 1]
            obss_list.append(obss[start: end])
            action_true_list.append(action_true[start: end])
            action_teacher_list.append(action_teacher[start: end])
            done_list.append(done[start: end])
        trajs = list(zip(obss_list, action_true_list, action_teacher_list, done_list))
        trajs.sort(key=lambda x: len(x[0]), reverse=True)
        obss_list, action_true_list, action_teacher_list, done_list = zip(*trajs)

        # obss = [o for traj in obss_list for o in traj]
        obss = obss_list
        action_true = torch.cat(action_true_list, dim=0)
        action_teacher = np.concatenate(action_teacher_list, axis=0)
        done = torch.cat(done_list, dim=0)

        inds = torch.where(done == 1)[0].detach().cpu().numpy() + 1
        done = done.detach().cpu().numpy()
        inds = np.concatenate([[0], inds[:-1]])

        mask = np.ones([len(action_true)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        return obss, action_true, action_teacher, done, inds, mask

    def set_mode(self, is_training, acmodel):
        if is_training:
            acmodel.train()
        else:
            acmodel.eval()

    def run_epoch_recurrence_one_batch(self, batch, is_training=False, source='agent', teacher_dict={}):
        active_teachers = [k for k, v in teacher_dict.items() if v]
        assert len(active_teachers) <= 2
        teacher_name = 'none' if len(active_teachers) == 0 else active_teachers[0]
        acmodel = self.policy_dict[teacher_name]
        optimizer = self.optimizer_dict[teacher_name]
        self.set_mode(is_training, acmodel)

        # All the batch demos are in a single flat vector.
        # Inds holds the start of each demo
        obss_list, action_true, action_teacher, done, inds, mask = batch
        # Dropout instructions with probability instr_dropout_prob,
        # unless we have no teachers present in which case we keep the instr.
        instr_dropout_prob = 0 if np.sum(list(teacher_dict.values())) == 0 else self.instr_dropout_prob
        obss_list = [self.preprocess_obs(o, teacher_dict, show_instrs=np.random.uniform() > instr_dropout_prob)
                     for o in obss_list]
        obss = {}
        keys = obss_list[0].keys()
        for k in keys:
            obss[k] = torch.cat([getattr(o, k) for o in obss_list])
        obss = DictList(obss)
        num_frames = len(obss)

        if self.label_weightings:
            weightings = torch.zeros(7, dtype=torch.float32).to(self.device)
            actions, counts = torch.unique(action_true, return_counts=True)
            counts = counts.float()
            weightings[actions] = 1 / counts
            weightings = weightings / torch.sum(weightings)
        else:
            weightings = torch.ones(7, dtype=torch.float32).to(self.device)

        # Memory to be stored
        memories = torch.zeros([num_frames, acmodel.memory_size], device=self.device)
        memory = torch.zeros([len(inds), acmodel.memory_size], device=self.device)

        # We're going to loop through each trajectory together.
        # inds holds the current index of each trajectory (initialized to the start of each trajectory)
        # memories is currently empty, but we will fill it with memory vectors.
        # memory holds the current memory vector
        while True:
            # taking observations and done located at inds
            num_demos = len(inds)
            obs = obss[inds]
            done_step = done[inds]

            with torch.no_grad():
                # Taking memory up until num_demos, as demos after that have finished
                dist, info = acmodel(obs, memory[:num_demos])
                new_memory = info['memory']

            memories[inds] = memory[:num_demos]
            memory[:num_demos] = new_memory

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            num_trajs_finished = sum(done_step)
            inds = inds[:int(num_demos - num_trajs_finished)]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]
        # Here, we take our trajectories and split them into chunks and compute the loss for each chunk.
        # Indexes currently holds the first index of each chunk.
        indexes = self.starting_indexes(num_frames)
        self.initialize_logs(indexes)
        memory = memories[indexes]
        final_loss = 0
        for i in range(self.args.recurrence):
            # Get the current obs for each chunk, and compute the agent's output
            obs = obss[indexes]
            action_step = action_true[indexes]
            mask_step = mask[indexes]
            dist, info = acmodel(obs, memory * mask_step)
            memory = info["memory"]

            if source == 'agent_probs':
                loss_fn = torch.nn.BCEWithLogitsLoss()
                policy_loss = loss_fn(dist.logits, action_step)
                action_step = torch.argmax(action_step, dim=1)
            else:
                action_weightings = weightings[action_step]
                policy_loss = -dist.log_prob(action_step) * action_weightings
            policy_loss = policy_loss.mean()
            # Compute the cross-entropy loss with an entropy bonus
            entropy = dist.entropy().mean()
            loss = policy_loss - self.args.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=False)[1]  # argmax action
            final_loss += loss
            self.log_t(action_pred, action_step, action_teacher, indexes, entropy, policy_loss)
            # Increment indexes to hold the next step for each chunk
            indexes += 1
        # Update the model
        final_loss /= self.args.recurrence
        if is_training:
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

        # Store log info
        log = self.log_final()
        return log

    def initialize_logs(self, indexes):
        self.per_token_correct = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_teacher_correct = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_count = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_teacher_count = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_agent_count = [0, 0, 0, 0, 0, 0, 0]
        self.precision_correct = [0, 0, 0, 0, 0, 0, 0]
        self.precision_count = [0, 0, 0, 0, 0, 0, 0]
        self.final_entropy = 0
        self.final_policy_loss = 0
        self.final_value_loss = 0
        self.accuracy = 0
        self.label_accuracy = 0
        self.total_frames = len(indexes) * self.args.recurrence
        self.accuracy_list = []
        self.lengths_list = []
        self.agent_running_count_long = 0
        self.teacher_running_count_long = 0

    def log_t(self, action_pred, action_step, action_teacher, indexes, entropy, policy_loss):
        self.accuracy_list.append(float((action_pred == action_step).sum()))
        self.lengths_list.append((action_pred.shape, action_step.shape, indexes.shape))
        self.accuracy += float((action_pred == action_step).sum()) / self.total_frames
        if action_step.shape == action_teacher[indexes].shape:
            self.label_accuracy += np.sum(action_teacher[indexes] == action_step.detach().cpu().numpy()) / self.total_frames

        self.final_entropy += entropy
        self.final_policy_loss += policy_loss

        action_step = action_step.detach().cpu().numpy()  # ground truth action
        action_pred = action_pred.detach().cpu().numpy()  # action we took
        agent_running_count = 0
        teacher_running_count = 0
        for j in range(len(self.per_token_count)):
            # Sort by agent action
            action_taken_indices = np.where(action_pred == j)[0]
            self.precision_correct[j] += np.sum(action_step[action_taken_indices] == action_pred[action_taken_indices])
            self.precision_count[j] += len(action_taken_indices)

            # Sort by label action
            token_indices = np.where(action_step == j)[0]
            count = len(token_indices)
            correct = np.sum(action_step[token_indices] == action_pred[token_indices])
            self.per_token_correct[j] += correct
            self.per_token_count[j] += count

            action_teacher_index = action_teacher[indexes]
            assert action_teacher_index.shape == action_pred.shape == action_step.shape, (
                action_teacher_index.shape, action_pred.shape, action_step.shape)
            teacher_token_indices = np.where(action_teacher_index == j)[0]
            teacher_count = len(teacher_token_indices)
            teacher_correct = np.sum(action_teacher_index[teacher_token_indices] == action_pred[teacher_token_indices])
            self.per_token_teacher_correct[j] += teacher_correct
            self.per_token_teacher_count[j] += teacher_count
            teacher_running_count += teacher_count
            agent_running_count += count
            self.agent_running_count_long += teacher_count
            self.teacher_running_count_long += teacher_count
        assert np.min(action_step) < len(self.per_token_count), (np.min(action_step), action_step)
        assert np.max(action_step) >= 0, (np.max(action_step), action_step)
        assert np.min(action_pred) < len(self.per_token_count), (np.min(action_pred), action_pred)
        assert np.max(action_pred) >= 0, (np.max(action_pred), action_pred)
        assert np.min(action_teacher_index) < len(self.per_token_count), (
        np.min(action_teacher_index), action_teacher_index)
        assert np.max(action_teacher_index) >= 0, (np.max(action_teacher_index), action_teacher_index)
        assert agent_running_count == teacher_running_count, (agent_running_count, teacher_running_count)
        assert agent_running_count == len(action_step) == len(action_pred), \
            (agent_running_count, len(action_step), len(action_pred))

    def log_final(self):
        log = {}
        log["Entropy"] = float(self.final_entropy / self.args.recurrence)
        log["Loss"] = float(self.final_policy_loss / self.args.recurrence)
        log["Accuracy"] = float(self.accuracy)
        log["Label_A"] = float(self.label_accuracy)
        assert float(self.accuracy) <= 1.0001, "somehow accuracy is greater than 1"
        assert float(self.accuracy) <= 1.0001, float(self.accuracy)
        teacher_numerator = 0
        teacher_denominator = 0
        agent_numerator = 0
        agent_denominator = 0
        for i, (correct, count) in enumerate(zip(self.precision_correct, self.precision_count)):
            if count > 0:
                log[f'Precision{i}'] = correct / count

        for i, (correct, count, teacher_correct, teacher_count) in enumerate(
            zip(self.per_token_correct, self.per_token_count, self.per_token_teacher_correct,
                self.per_token_teacher_count)):
            assert correct <= count, (correct, count)
            assert teacher_correct <= teacher_count, (teacher_correct, teacher_count)
            if count > 0:
                log[f'Accuracy_{i}'] = correct / count
                agent_numerator += correct
                agent_denominator += count
            if teacher_count > 0:
                log[f'TeacherAccuracy_{i}'] = teacher_correct / teacher_count
                teacher_numerator += teacher_correct
                teacher_denominator += teacher_count
        assert agent_denominator == teacher_denominator, (agent_denominator, teacher_count)

        assert agent_denominator == teacher_denominator, (
            agent_denominator, teacher_denominator, self.per_token_count, self.per_token_teacher_count)
        assert abs(float(self.accuracy) - agent_numerator / agent_denominator) < .001, (
            self.accuracy, agent_numerator / agent_denominator)
        log["TeacherAccuracy"] = float(teacher_numerator / teacher_denominator)
        return log

    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            return np.arange(0, num_frames, self.args.recurrence)[:-1]

    def relabel(self, batch, relabel_dict, source):
        if source == 'teacher':
            return batch
        self.set_mode(False)
        obss_list_original, action_true, action_teacher, done, inds_original, mask = batch
        inds = inds_original.copy()

        # All the batch demos are in a single flat vector.
        # Inds holds the start of each demo
        obss_list = [self.preprocess_obs(o, relabel_dict, show_instrs=True) for o in obss_list_original]
        obss = {}
        keys = obss_list[0].keys()
        for k in keys:
            obss[k] = torch.cat([getattr(o, k) for o in obss_list])
        obss = DictList(obss)
        actions = torch.zeros_like(action_true, device=self.device,
                                   dtype=torch.float32 if source == 'agent_probs' else torch.long)
        memory = torch.zeros([len(inds), self.acmodel.memory_size], device=self.device)

        # We're going to loop through each trajectory together.
        # inds holds the current index of each trajectory (initialized to the start of each trajectory)
        while True:
            # taking observations and done located at inds
            num_demos = len(inds)
            obs = obss[inds]
            done_step = done[inds]

            with torch.no_grad():
                # Taking memory up until num_demos, as demos after that have finished
                dist, info = self.acmodel(obs, memory[:num_demos])
                if source == 'agent':
                    action = dist.sample().long()
                elif source == 'agent_argmax':
                    action = dist.probs.max(1, keepdim=False)[1].long()
                elif source == 'agent_probs':
                    action = dist.probs.float()
                new_memory = info['memory']
            actions[inds] = action[:num_demos]
            memory[:num_demos] = new_memory

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            num_trajs_finished = sum(done_step)
            inds = inds[:int(num_demos - num_trajs_finished)]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]
        return obss_list_original, actions, action_teacher, done, inds_original, mask

    def distill(self, demo_batch, is_training=True, source='agent', teachers_dict={}, distill_target='powerset',
                relabel=False, relabel_dict={}):

        preprocessed_batch = self.preprocess_batch(demo_batch, source)
        if relabel:
            preprocessed_batch = self.relabel(preprocessed_batch, relabel_dict, source)

        # Distill to the powerset of distillation types
        keys = [key for key in teachers_dict.keys() if teachers_dict[key]]
        powerset = chain.from_iterable(combinations(keys, r) for r in range(len(keys) + 1))
        logs = {}

        if distill_target == 'powerset':
            for key_set in powerset:
                if len(key_set) > 1:
                    continue
                teacher_subset_dict = {}
                for k in teachers_dict.keys():
                    if k in key_set:
                        teacher_subset_dict[k] = True
                    else:
                        teacher_subset_dict[k] = False
                batch = copy.deepcopy(preprocessed_batch)
                log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training, source=source,
                                                          teacher_dict=teacher_subset_dict)
                logs[key_set] = log
        elif distill_target == 'not_none':
            for key_set in powerset:
                if len(key_set) > 1:
                    continue
                teacher_subset_dict = {}
                if len(key_set) == 0:  # Don't distill to no teacher
                    continue
                for k in teachers_dict.keys():
                    if k in key_set:
                        teacher_subset_dict[k] = True
                    else:
                        teacher_subset_dict[k] = False
                batch = copy.deepcopy(preprocessed_batch)
                log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training, source=source,
                                                          teacher_dict=teacher_subset_dict)
                logs[key_set] = log
        elif distill_target == 'all':
            key_set = tuple(set(keys))
            if len(key_set) > 1:
                raise NotImplementedError
            else:
                teacher_name = key_set[0]
            log = self.run_epoch_recurrence_one_batch(preprocessed_batch, is_training=is_training, source=source,
                                                      teacher_dict=teachers_dict)
            logs[key_set] = log
        elif distill_target == 'none':
            key_set = tuple(set())
            teacher_subset_dict = {}
            for k in teachers_dict.keys():
                teacher_subset_dict[k] = False
            log = self.run_epoch_recurrence_one_batch(preprocessed_batch, is_training=is_training, source=source,
                                                      teacher_dict=teacher_subset_dict)
            logs[key_set] = log

        if is_training:
            for scheduler in self.scheduler_dict.values():
                scheduler.step()
        return logs
