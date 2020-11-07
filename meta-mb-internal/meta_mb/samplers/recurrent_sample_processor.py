from meta_mb.samplers.base import SampleProcessor
import numpy as np
import copy


class RecurrentSampleProcessor(SampleProcessor):

    def process_samples(self, paths_meta_batch, log=False, log_prefix=''):
        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        assert self.baseline, 'baseline must be specified'

        samples_data_meta_batch = []
        all_paths = []

        for meta_task, paths in paths_meta_batch.items():
            # fits baseline, compute advantages and stack path data
            samples_data, paths = self._compute_samples_data(paths)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

        observations, actions, rewards, dones, returns, advantages, env_infos, agent_infos = \
            self._stack_path_data(copy.deepcopy(samples_data_meta_batch))

        # 8) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )
        return samples_data
