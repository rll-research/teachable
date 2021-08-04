from envs.babyai.rl.algos.base import BaseAlgo

class DataCollector(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, collect_policy, envs, args, obs_preprocessor, repeated_seed=None):
        self.discrete = args.discrete

        super().__init__(envs, collect_policy, args.frames_per_proc, args.discount, args.lr, args.gae_lambda, args.entropy_coef,
                         args.value_loss_coef, args.max_grad_norm, obs_preprocessor, None,
                         not args.sequential, args.rollouts_per_meta_task, instr_dropout_prob=args.collect_dropout_prob,
                         repeated_seed=repeated_seed, reset_each_batch=args.reset_each_batch,
                         on_policy=args.algo == 'ppo')

    def update_parameters(self):
        raise NotImplementedError