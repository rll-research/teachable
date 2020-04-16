from gym.core import Env
from gym.envs.mujoco import MujocoEnv
import numpy as np


class MetaEnv(Env):
    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        return [None] * n_tasks
        # raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        pass
        # raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        return None
        # raise NotImplementedError

    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass

class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction', 'jnt_stiffness']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, *args, rand_params=RAND_PARAMS, **kwargs):
        super(RandomEnv, self).__init__(*args, **kwargs)
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.log_scale_limit = log_scale_limit            
        self.rand_params = rand_params
        self.save_parameters()

    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []

        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts

            new_params = {}

            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

            # stiffness at the model's joints
            if 'jnt_stiffness' in self.rand_params:
                jnt_stiffness_multpliers = np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.jnt_stiffness.shape)
                new_params['jnt_stiffness'] = np.multiply(self.init_params['jnt_stiffness'], jnt_stiffness_multpliers)

            param_sets.append(new_params)

        return param_sets

    def set_task(self, task):
        for param, param_val in task.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            for i in range(param_variable.shape[0]):
                if param == 'body_mass':
                    self.model.body_mass[i] = param_val[i]
                elif param == 'body_inertia':
                    self.model.body_inertia[i] = param_val[i]
                elif param == 'dof_damping':
                    self.model.dof_damping[i] = param_val[i]
                elif param == 'geom_friction':
                    self.model.geom_friction[i] = param_val[i]
                elif param == 'geom_size':
                    self.model.geom_size[i] = param_val[i]
                elif param == "jnt_stiffness":
                    self.model.jnt_stiffness[i] = param_val[i]
                else:
                    setattr(self.model, param, param_val)

        self.cur_params = task

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.model.dof_damping

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.model.geom_friction

        # stiffness at the model's joints 
        if 'jnt_stiffness' in self.rand_params:
            self.init_params['jnt_stiffness'] = self.model.jnt_stiffness

        self.cur_params = self.init_params
