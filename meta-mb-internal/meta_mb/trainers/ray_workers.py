import time, sys
from meta_mb.logger import logger
import multiprocessing
import ray

class Worker(object):
    """
    Abstract class for worker instantiations. 
    """

    def __init__(self):
        pass

    def step(self):
        raise NotImplementedError

    def synch(self, data):
        # default is to do nothing
        pass

@ray.remote(num_cpus=1, num_gpus=0)
class Server(object):
    def __init__(self):
        self.objects = {
                'samples_data': None, 
                'env_sampler': None, 
                'dynamics_model': None, 
                'model_sampler': None, 
                }
    
    # update samples_data
    def push(self, keys, values):
        for key in keys:
            self.objects[key] = value # TODO: value.copy()? value_id or value? 

    # return latest version of samples_data
    def pull(self, keys):
        return [self.objects[key] for key in keys]

@ray.remote(num_cpus=1, num_gpus=0)
class WorkerData(Worker):
    def __init__(self, server, dynamics_sample_processor):
        super().__init__()
        self.server = server
        self.dynamics_sample_processor = dynamics_sample_processor

    def step(self, random=False):
        """
        Uses self.env_sampler which samples data under policy.
        Outcome: generate samples_data.
        """

        time_env_sampling_start = time.time()

        logger.log("Obtaining samples from the environment using the policy...")
        env_sampler = self.server.pull.remote(['env_sampler'])[0]
        env_paths = env_sampler.obtain_samples(log=True, random=random, log_prefix='EnvSampler-')

#        logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)
        logger.log("Processing environment samples...")

        # first processing just for logging purposes
        time_env_samp_proc = time.time()
        samples_data = self.dynamics_sample_processor.process_samples(env_paths, log=True,
                                                                      log_prefix='EnvTrajs-')
#        self.env.log_diagnostics(env_paths, prefix='EnvTrajs-')
#        logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)
        
        self.server.push.remote('samples_data', ray.put(samples_data))
            
@ray.remote(num_cpus=1, num_gpus=0)
class WorkerModel(Worker):
    def __init__(self, server, dynamics_model_max_epochs):
        super().__init__()
        self.server = server
        self.dynamics_model_max_epochs = dynamics_model_max_epochs

    def step(self):
        '''
        In sequential order, is "samples_data" accumulated???
        Outcome: dynamics model is updated with self.samples_data.?
        '''
        time_fit_start = time.time()

        ''' --------------- fit dynamics model --------------- '''

        logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
        dynamics_model, samples_data = self.server.pull.remote(['dynamics_model', 'samples_data'])
        self.dynamics_model.fit(samples_data['observations'],
                                samples_data['actions'],
                                samples_data['next_observations'],
                                epochs=self.dynamics_model_max_epochs, verbose=False, log_tabular=True)

        self.server.push.remote('dynamics_data', ray.put(dynamics_model))
#        logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)


@ray.remote(num_cpus=1, num_gpus=0)
class WorkerPolicy(Worker):
    def __init__(self, model_sample_processor, algo):
        super().__init__()
        self.server = server
        self.model_sample_processor = model_sample_processor
        self.algo = algo

    def step(self):
        """
        Uses self.model_sampler which is asynchrounously updated by worker_model.
        Outcome: policy is updated by PPO on one fictitious trajectory. 
        """

        itr_start_time = time.time()

        """ -------------------- Sampling --------------------------"""

        logger.log("Obtaining samples from the model...")
        time_env_sampling_start = time.time()
        model_sampler = self.server.pull.remote(['model_sampler'])[0]
        paths = model_sampler.obtain_samples(log=True, log_prefix='train-')
        sampling_time = time.time() - time_env_sampling_start

        """ ----------------- Processing Samples ---------------------"""

        logger.log("Processing samples from the model...")
        time_proc_samples_start = time.time()
        samples_data = self.model_sample_processor.process_samples(paths, log='all', log_prefix='train-')
        proc_samples_time = time.time() - time_proc_samples_start

        """ ------------------ Policy Update ---------------------"""

        logger.log("Optimizing policy...")
        # This needs to take all samples_data so that it can construct graph for meta-optimization.
        time_optimization_step_start = time.time()
        self.algo.optimize_policy(samples_data)
        optimization_time = time.time() - time_optimization_step_start

#        times_dyn_sampling.append(sampling_time)
#        times_dyn_sample_processing.append(proc_samples_time)
#        times_optimization.append(optimization_time)
#        times_step.append(time.time() - itr_start_time)

        return None

