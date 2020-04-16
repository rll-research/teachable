import time
from meta_mb.logger import logger
from multiprocessing import current_process
from queue import Empty


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(
            self,
            verbose=True,
    ):
        self.verbose = verbose

    def construct_from_feed_dict(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(
            self,
            exp_dir,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
            queue_prev,
            queue,
            queue_next,
            remote,
            start_itr,
            n_itr,
            stop_cond,
            need_query,
            auto_push,
            config,
    ):
        time_start = time.time()

        self.name = current_process().name
        logger.configure(dir=exp_dir + '/' + self.name, format_strs=['csv', 'stdout', 'log'], snapshot_mode='last')

        self.n_itr = n_itr
        self.queue_prev = queue_prev
        self.queue = queue
        self.queue_next = queue_next
        self.stop_cond = stop_cond

        # FIXME: specify CPU/GPU usage here

        import tensorflow as tf

        def _init_vars():
            sess = tf.get_default_session()
            sess.run(tf.initializers.global_variables())

        with tf.Session(config=config).as_default():

            self.construct_from_feed_dict(
                policy_pickle,
                env_pickle,
                baseline_pickle,
                dynamics_model_pickle,
                feed_dict,
            )

            _init_vars()

            # warm up
            self.itr_counter = start_itr
            if self.verbose:
                print('{} waiting for starting msg from trainer...'.format(self.name))
            assert remote.recv() == 'prepare start'
            self.prepare_start()
            remote.send('loop ready')
            logger.dumpkvs()
            logger.log("\n============== {} is ready =============".format(self.name))

            assert remote.recv() == 'start loop'
            total_push, total_synch, total_step = 0, 0, 0
            while not self.stop_cond.is_set():
                if self.verbose:
                    logger.log("\n------------------------- {} starting new loop ------------------".format(self.name))
                if need_query: # poll
                    time_poll = time.time()
                    queue_prev.put('push')
                    time_poll = time.time() - time_poll
                    logger.logkv('{}-TimePoll'.format(self.name), time_poll)
                do_push, do_synch, do_step = self.process_queue()
                # step
                if do_step:
                    self.itr_counter += 1
                    self.step()
                    if auto_push:
                        do_push += 1
                        self.push()
                    # Assuming doing autopush for all
                    assert do_push == 1
                    assert do_step == 1

                total_push += do_push
                total_synch += do_synch
                total_step += do_step
                logger.logkv(self.name+'-TimeSoFar', time.time() - time_start)
                logger.logkv(self.name+'-TotalPush', total_push)
                logger.logkv(self.name+'-TotalSynch', total_synch)
                logger.logkv(self.name+'-TotalStep', total_step)
                if total_synch > 0:
                    logger.logkv(self.name+'-StepPerSynch', total_step/total_synch)
                logger.dumpkvs()
                logger.log("\n========================== {} {}, total {} ===================".format(
                    self.name,
                    (do_push, do_synch, do_step),
                    (total_push, total_synch, total_step),
                ))
                self.set_stop_cond()

            remote.send('loop done')

        logger.log("\n================== {} closed ===================".format(
            self.name
        ))

        remote.send('worker closed')

    def prepare_start(self):
        raise NotImplementedError

    def process_queue(self):
        do_push, do_synch = 0, 0
        data = None

        while True:
            try:
                if self.verbose:
                    logger.log('{} try'.format(self.name))
                new_data = self.queue.get_nowait()
                if new_data == 'push': # only happens when next worker has need_query = True
                    if do_push == 0:  # only push once
                        do_push += 1
                        self.push()
                else:
                    do_synch = 1
                    data = new_data
            except Empty:
                break

        if do_synch:
            self._synch(data)

        do_step = 1 # - do_synch

        if self.verbose:
            logger.log('{} finishes processing queue with {}, {}, {}......'.format(self.name, do_push, do_synch, do_step))

        return do_push, do_synch, do_step

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def _synch(self, *args, **kwargs):
        raise NotImplementedError

    def push(self):
        raise NotImplementedError

    def set_stop_cond(self):
        pass

