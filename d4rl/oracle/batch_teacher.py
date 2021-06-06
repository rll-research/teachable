class BatchTeacher:
    """
    Batched version of the Teacher class.
    """
    def __init__(self, controller, teachers):
        self.waypoint_controller = controller
        self.teachers = teachers

    def get_action(self, env):
        action, _ = self.waypoint_controller.get_action(env.get_pos(), env.get_vel(), env.get_target())
        return action, self.waypoint_controller.waypoints.copy()

    def step(self, env):
        self.last_action = self.next_action
        self.next_action, self.waypoints = self.get_action(env)
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.step()
        return return_dict

    def give_feedback(self, env):
        return_dict = {}
        for k, v in self.teachers.items():
            advice, advice_given = v.give_feedback(env)
            return_dict[k] = advice
            return_dict['gave_' + k] = advice_given
        return return_dict

    def empty_feedback(self):
        return {k: v.empty_feedback() for k, v in self.teachers.items()}

    def compute_feedback(self, env):
        return {k: v.compute_feedback(env) for k, v in self.teachers.items()}

    def feedback_condition(self):
        return {k: v.feedback_condition() for k, v in self.teachers.items()}

    def reset(self, env):
        self.last_action = -1
        self.next_action, self.waypoints = self.get_action(env)
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.reset()
        return return_dict

    def success_check(self, state, action, oracle):
        # Not implemented for the d4rl envs
        return {k: True for k in self.teacher.keys()}