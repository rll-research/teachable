import time
from meta_mb.logger import logger
from multiprocessing import Process, Pipe, Queue, Event
from meta_mb.workers.metrpo.worker_data import WorkerData
from meta_mb.workers.metrpo.worker_model import WorkerModel
from meta_mb.workers.metrpo.worker_policy import WorkerPolicy


class ParallelTrainer(object):
    """
    Performs steps for MAML

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            exp_dir,
            algo_str,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dicts,
            n_itr,
            flags_need_query,
            config,
            simulation_sleep,
            initial_random_samples=True,
            start_itr=0,
            sampler_str='metrpo',
    ):
        self.initial_random_samples = initial_random_samples

        worker_instances = [
            WorkerData(simulation_sleep=simulation_sleep),
            WorkerModel(),
            WorkerPolicy(algo_str=algo_str, sampler_str=sampler_str),
        ]
        names = ["Data", "Model", "Policy"]
        # one queue for each worker, tasks assigned by scheduler and previous worker
        queues = [Queue(-1) for _ in range(3)]
        # worker sends task-completed notification and time info to scheduler
        worker_remotes, remotes = zip(*[Pipe() for _ in range(3)])
        # stop condition
        stop_cond = Event()
        # current worker needs query means previous workers does not auto push
        # skipped checking here
        flags_auto_push = [not flags_need_query[1], not flags_need_query[2], not flags_need_query[0]]

        self.ps = [
            Process(
                target=worker_instance,
                name=name,
                args=(
                    exp_dir,
                    policy_pickle,
                    env_pickle,
                    baseline_pickle,
                    dynamics_model_pickle,
                    feed_dict,
                    queue_prev,
                    queue,
                    queue_next,
                    worker_remote,
                    start_itr,
                    n_itr,
                    stop_cond,
                    need_query,
                    auto_push,
                    config,
                ),
            ) for (worker_instance, name, feed_dict,
                   queue_prev, queue, queue_next,
                   worker_remote, need_query, auto_push) in zip(
                worker_instances, names, feed_dicts,
                queues[2:] + queues[:2], queues, queues[1:] + queues[:1],
                worker_remotes, flags_need_query, flags_auto_push,
                )
        ]

        # central scheduler sends command and receives receipts
        self.names = names
        self.queues = queues
        self.remotes = remotes

    def train(self):
        """
        Trains policy on env using algo
        """
        worker_data_queue, worker_model_queue, worker_policy_queue = self.queues
        worker_data_remote, worker_model_remote, worker_policy_remote = self.remotes

        for p in self.ps:
            p.start()

        ''' --------------- worker warm-up --------------- '''

        logger.log('Prepare start...')

        worker_data_remote.send('prepare start')
        worker_data_queue.put(self.initial_random_samples)
        assert worker_data_remote.recv() == 'loop ready'

        worker_model_remote.send('prepare start')
        assert worker_model_remote.recv() == 'loop ready'

        worker_policy_remote.send('prepare start')
        assert worker_policy_remote.recv() == 'loop ready'

        time_total = time.time()

        ''' --------------- worker looping --------------- '''

        logger.log('Start looping...')
        for remote in self.remotes:
            remote.send('start loop')

        ''' --------------- collect info --------------- '''

        for remote in self.remotes:
            assert remote.recv() == 'loop done'
        logger.log('\n------------all workers exit loops -------------')
        for remote in self.remotes:
            assert remote.recv() == 'worker closed'

        for p in self.ps:
            p.terminate()

        logger.logkv('Trainer-TimeTotal', time.time() - time_total)
        logger.dumpkvs()
        logger.log('*****Training finished')
