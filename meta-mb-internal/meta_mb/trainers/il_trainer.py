import numpy as np
import torch
import babyai.utils as utils
import logging

logger = logging.getLogger(__name__)


class ImitationLearning(object):
    def __init__(self, model, env, args, distill_with_teacher, reward_predictor=False):
        self.args = args
        self.distill_with_teacher = distill_with_teacher
        self.reward_predictor = reward_predictor

        utils.seed(self.args.seed)
        self.env = env

        # Define actor-critic model
        self.acmodel = model

        self.acmodel.train()
        if torch.cuda.is_available():
            self.acmodel.cuda()

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_batch(self, batch, source):
        obss = batch.obs
        if source == 'teacher':
            action_true = batch.env_infos.teacher_action[:, 0]
        elif source == 'agent':
            action_true = batch.action
        action_true = torch.tensor(action_true, device=self.device, dtype=torch.long)
        action_teacher = batch.env_infos.teacher_action[:, 0]
        done = batch.full_done
        inds = torch.where(done == 1)[0].detach().cpu().numpy() + 1
        done = done.detach().cpu().numpy()
        inds = np.concatenate([[0], inds[:-1]])

        mask = np.ones([len(obss)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        return obss, action_true, action_teacher, done, inds, mask

    def set_mode(self, is_training):
        if is_training:
            self.acmodel.train()
        else:
            self.acmodel.eval()

    def obss_preprocessor(self, obs, device=None):
        obs_arr = np.stack(obs, 0)
        return torch.FloatTensor(obs_arr).to(device)

    def run_epoch_recurrence_one_batch(self, batch, is_training=False, source='agent'):
        self.set_mode(is_training)

        # All the batch demos are in a single flat vector.
        # Inds holds the start of each demo
        obss, action_true, action_teacher, done, inds, mask = self.preprocess_batch(batch, source)
        num_frames = len(obss)

        # Memory to be stored
        memories = torch.zeros([num_frames, self.acmodel.memory_size], device=self.device)
        memory = torch.zeros([len(inds), self.acmodel.memory_size], device=self.device)

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
                obs = self.obss_preprocessor(obs, self.device)
                dist, info = self.acmodel(obs, memory[:num_demos], use_teacher=self.distill_with_teacher)
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
            obs = self.obss_preprocessor(obs, self.device)
            dist, info = self.acmodel(obs, memory * mask_step, use_teacher=self.distill_with_teacher)
            memory = info["memory"]

            # Compute the cross-entropy loss with an entropy bonus
            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()
            loss = policy_loss - self.args.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=False)[1]  # argmax action
            final_loss += loss
            self.log_t(action_pred, action_step, action_teacher, indexes, entropy, policy_loss)
            # Increment indexes to hold the next step for each chunk
            indexes += 1

        # Update the model
        final_loss /= self.args.recurrence
        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

        # Store log info
        log = self.log_final()
        return log

    def initialize_logs(self, indexes):
        self.per_token_correct = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_teacher_correct = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_count = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_teacher_count = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_agent_count = [0, 0, 0, 0, 0, 0, 0]
        self.final_entropy = 0
        self.final_policy_loss = 0
        self.final_value_loss = 0
        self.accuracy = 0
        self.total_frames = len(indexes) * self.args.recurrence
        self.accuracy_list = []
        self.lengths_list = []
        self.agent_running_count_long = 0
        self.teacher_running_count_long = 0

    def log_t(self, action_pred, action_step, action_teacher, indexes, entropy, policy_loss):
        self.accuracy_list.append(float((action_pred == action_step).sum()))
        self.lengths_list.append((action_pred.shape, action_step.shape, indexes.shape))
        self.accuracy += float((action_pred == action_step).sum()) / self.total_frames

        self.final_entropy += entropy
        self.final_policy_loss += policy_loss

        action_step = action_step.detach().cpu().numpy()  # ground truth action
        action_pred = action_pred.detach().cpu().numpy()  # action we took
        agent_running_count = 0
        teacher_running_count = 0
        for j in range(len(self.per_token_count)):
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
        if not float(self.accuracy) <= 1.0001:
            print("?", self.accuracy)
            print("Accuracy List", len(self.accuracy_list), self.total_frames, len(self.indexes))
            print(self.accuracy_list)
            print(self.lengths_list)
            import IPython
            IPython.embed()
        assert float(self.accuracy) <= 1.0001, float(self.accuracy)
        teacher_numerator = 0
        teacher_denominator = 0
        agent_numerator = 0
        agent_denominator = 0
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

    def distill(self, demo_batch, is_training=True, source='agent'):
        # Log is a dictionary with keys entropy, policy_loss, and accuracy
        log = self.run_epoch_recurrence_one_batch(demo_batch, is_training=is_training, source=source)
        if is_training:
            # Learning rate scheduler
            self.scheduler.step()
        return log
