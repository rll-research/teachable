import numpy as np
import torch
import babyai.utils as utils


class ImitationLearning(object):
    def __init__(self, model, env, args, distill_with_teacher, reward_predictor=False, preprocess_obs=lambda x: x,
                 label_weightings=False, instr_dropout_prob=.5, obs_dropout_prob=.3):
        self.args = args
        self.distill_with_teacher = distill_with_teacher
        self.reward_predictor = reward_predictor
        self.preprocess_obs = preprocess_obs
        self.label_weightings = label_weightings
        self.instr_dropout_prob = instr_dropout_prob
        self.obs_dropout_prob = obs_dropout_prob

        utils.seed(self.args.seed)
        self.env = env

        # Define actor-critic model
        self.policy_dict = model
        for policy in model.values():
            if torch.cuda.is_available():
                policy.cuda()

        teachers = list(self.policy_dict.keys())
        # Different optimizers for different models, same optimizer for same model
        first_teacher = teachers[0]
        self.optimizer_dict = {first_teacher: torch.optim.Adam(self.policy_dict[first_teacher].parameters(),
                                                               self.args.lr, eps=self.args.optim_eps)}
        for teacher in teachers[1:]:
            policy = self.policy_dict[teacher]
            if policy is self.policy_dict[first_teacher]:
                self.optimizer_dict[teacher] = self.optimizer_dict[first_teacher]
            else:
                self.optimizer_dict[teacher] = torch.optim.Adam(self.policy_dict[teacher].parameters(),
                                                                self.args.lr, eps=self.args.optim_eps)
        self.scheduler_dict = {
            k: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
            for k, optimizer in self.optimizer_dict.items()
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_batch(self, batch, source):
        obss = batch.obs
        if source == 'teacher':
            action_true = batch.teacher_action
            if len(action_true.shape) == 2 and action_true.shape[1] == 1:
                action_true = action_true[:, 0]
        elif source == 'agent':
            action_true = batch.action
        elif source == 'agent_probs':
            if self.args.discrete == False:
                raise NotImplementedError('Action probs not implemented for continuous envs.')
            action_true = batch.argmax_action
        if not source == 'agent_probs':
            dtype = torch.long if self.args.discrete else torch.float32
            action_true = torch.tensor(action_true, device=self.device, dtype=dtype)
        action_teacher = batch.teacher_action
        if len(action_teacher.shape) == 2 and action_teacher.shape[1] == 1:
            action_teacher = action_teacher[:, 0]
        return obss, action_true, action_teacher

    def set_mode(self, is_training, acmodel):
        if is_training:
            acmodel.train()
        else:
            acmodel.eval()

    def distill_batch(self, batch, is_training=False, teacher_dict={}, backprop=True):
        ### SETUP ###

        # Select which model to train based on which teachers are provided
        active_teachers = [k for k, v in teacher_dict.items() if v]
        teacher_name = 'none' if len(active_teachers) == 0 else active_teachers[0]
        acmodel = self.policy_dict[teacher_name]
        optimizer = self.optimizer_dict[teacher_name]
        self.set_mode(is_training, acmodel)

        # Obtain batch and preprocess
        obss, action_true, action_teacher = batch
        self.initialize_logs(len(action_true))
        # Dropout instructions with probability instr_dropout_prob,
        # unless we have no teachers present in which case we keep the instr.
        instr_dropout_prob = 0 if np.sum(list(teacher_dict.values())) == 0 else self.instr_dropout_prob
        obss = self.preprocess_obs(obss, teacher_dict, show_instrs=np.random.uniform() > instr_dropout_prob)

        ### RUN MODEL, COMPUTE LOSS ###
        dist, info = acmodel(obss)
        kl_loss = info['kl']
        policy_loss = -dist.log_prob(action_true).mean()
        entropy = dist.entropy().mean()

        # Optional loss for reconstructing the feedback (currently, all models all teachers)
        if 'reconstruction' in info:
            # Loss for training the reconstructor
            advice = info['reconstruction']
            full_advice_size = int(len(advice[0]) / 2)
            mean = advice[:, :full_advice_size]
            var = torch.exp(advice[:, full_advice_size:])
            loss_fn = torch.nn.GaussianNLLLoss()
            true_advice = obss.advice
            reconstruction_loss = loss_fn(true_advice, mean, var)
        else:
            reconstruction_loss = torch.FloatTensor([0]).to(self.device)
        loss = policy_loss - self.args.entropy_coef * entropy + self.args.mi_coef * reconstruction_loss + self.args.kl_coef * kl_loss

        ### LOGGING ###
        if self.args.discrete:
            action_pred = dist.probs.max(1, keepdim=False)[1]  # argmax action
            avg_mean_dist = -1
            avg_std = -1
        else:  # Continuous env
            action_pred = dist.mean
            avg_std = dist.scale.mean()
            avg_mean_dist = torch.abs(action_pred - action_true).mean()
        self.log_t(action_pred, action_true, action_teacher, entropy, policy_loss, reconstruction_loss,
                   kl_loss, avg_mean_dist, avg_std, loss)
        # Store log info
        log = self.log_final()

        ### UPDATE ###

        if is_training and backprop:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return log, loss

    def initialize_logs(self, total_frames):
        self.per_token_correct = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_teacher_correct = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_count = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_teacher_count = [0, 0, 0, 0, 0, 0, 0]
        self.per_token_agent_count = [0, 0, 0, 0, 0, 0, 0]
        self.precision_correct = [0, 0, 0, 0, 0, 0, 0]
        self.precision_count = [0, 0, 0, 0, 0, 0, 0]
        self.final_entropy = 0
        self.final_policy_loss = 0
        self.final_reconstruction_loss = 0
        self.final_kl_loss = 0
        self.final_value_loss = 0
        self.accuracy = 0
        self.label_accuracy = 0
        self.total_frames = total_frames
        self.accuracy_list = []
        self.agent_running_count_long = 0
        self.teacher_running_count_long = 0
        self.final_mean_dist = 0
        self.final_std = 0
        self.final_loss = 0

    def log_t(self, action_pred, action_true, action_teacher, entropy, policy_loss, reconstruction_loss,
              kl_loss, avg_mean_dist, avg_std, loss):
        self.accuracy_list.append(float((action_pred == action_true).sum()))
        self.accuracy += float((action_pred == action_true).sum()) / self.total_frames
        if action_true.shape == action_teacher.shape:
            self.label_accuracy += np.sum(action_teacher == action_true.detach().cpu().numpy()) / self.total_frames

        self.final_entropy += entropy
        self.final_policy_loss += policy_loss
        self.final_reconstruction_loss += reconstruction_loss
        self.final_kl_loss += kl_loss
        self.final_mean_dist += avg_mean_dist
        self.final_std += avg_std
        self.final_loss += loss

        action_true = action_true.detach().cpu().numpy()  # ground truth action
        action_pred = action_pred.detach().cpu().numpy()  # action we took
        agent_running_count = 0
        teacher_running_count = 0
        if self.args.discrete:
            for j in range(len(self.per_token_count)):
                # Sort by agent action
                action_taken_indices = np.where(action_pred == j)[0]
                self.precision_correct[j] += np.sum(
                    action_true[action_taken_indices] == action_pred[action_taken_indices])
                self.precision_count[j] += len(action_taken_indices)

                # Sort by label action
                token_indices = np.where(action_true == j)[0]
                count = len(token_indices)
                correct = np.sum(action_true[token_indices] == action_pred[token_indices])
                self.per_token_correct[j] += correct
                self.per_token_count[j] += count

                action_teacher_index = action_teacher
                assert action_teacher_index.shape == action_pred.shape == action_true.shape, (
                    action_teacher_index.shape, action_pred.shape, action_true.shape)
                teacher_token_indices = np.where(action_teacher_index == j)[0]
                teacher_count = len(teacher_token_indices)
                teacher_correct = np.sum(
                    action_teacher_index[teacher_token_indices] == action_pred[teacher_token_indices])
                self.per_token_teacher_correct[j] += teacher_correct
                self.per_token_teacher_count[j] += teacher_count
                teacher_running_count += teacher_count
                agent_running_count += count
                self.agent_running_count_long += teacher_count
                self.teacher_running_count_long += teacher_count
            assert agent_running_count == teacher_running_count, (agent_running_count, teacher_running_count)
            assert agent_running_count == len(action_true) == len(action_pred), \
                (agent_running_count, len(action_true), len(action_pred))

    def log_final(self):
        log = {}
        log["Entropy_Loss"] = float(self.final_entropy * self.args.entropy_coef)
        log["Entropy"] = float(self.final_entropy)
        log["Loss"] = float(self.final_policy_loss)
        log["TotalLoss"] = float(self.final_loss)
        log["Reconstruction_Loss"] = float(self.final_reconstruction_loss * self.args.mi_coef)
        log["KL_Loss"] = float(self.final_kl_loss * self.args.kl_coef)
        log["Accuracy"] = float(self.accuracy)
        log["Mean_Dist"] = float(self.final_mean_dist)
        log["Std"] = float(self.final_std)

        if self.args.discrete:
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

    def distill(self, demo_batch, is_training=True, source='agent', teachers_dict={}, distill_target='single_teachers',
                distill_to_none=True):
        preprocessed_batch = self.preprocess_batch(demo_batch, source)
        if distill_target == 'single_teachers_none' and not distill_to_none:
            distill_target = 'single_teachers'
        if distill_target == 'single_teachers':
            dict_list = [(key,) for key in teachers_dict.keys() if teachers_dict[key]]
        elif distill_target == 'single_teachers_none':
            dict_list = [(key,) for key in teachers_dict.keys() if teachers_dict[key]] + [()]
        elif distill_target == 'no_teachers':
            dict_list = [set()]
        logs = {}
        for keys in dict_list:
            teacher_subset_dict = {k: k in keys for k in teachers_dict.keys()}
            log, _ = self.distill_batch(preprocessed_batch, is_training=is_training,
                                        teacher_dict=teacher_subset_dict)
            logs[keys] = log
        if is_training:
            for scheduler in self.scheduler_dict.values():
                scheduler.step()
        return logs
