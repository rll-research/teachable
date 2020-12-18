import numpy as np

class BatchTeacher:
    """
    Batched version of the Teacher class.
    """
    def __init__(self, teachers):
        self.teachers = teachers

    def step(self, action, oracle):
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.step(action, oracle[k])
        return return_dict

    def give_feedback(self, state, oracle):
        return_dict = {}
        for k, v in self.teachers.items():
            advice, advice_given = v.give_feedback(state, oracle[k])
            return_dict[k] = advice
            return_dict['gave_' + k] = advice_given
        return return_dict

    def empty_feedback(self):
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.empty_feedback()
        return return_dict

    # TODO: do we really want null feedback to always be 0?  Maybe it should be noise?  Or some special token?
    def null_feedback(self):
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.empty_feedback() * 0
        return return_dict

    def compute_feedback(self):
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.compute_feedback()
        return return_dict

    def feedback_condition(self):
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.feedback_condition()
        return return_dict

    def set_feedback_type(self, feedback_type):
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.set_feedback_type(feedback_type)
        return return_dict

    def reset(self, oracle):
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.reset(oracle[k])
        return return_dict

    def get_last_step_error(self):
        last_step_error = [t.last_step_error for t in self.teachers.values()]
        last_step_error = np.max(last_step_error)
        return last_step_error

    def success_check(self, state, action, oracle):
        return_dict = {}
        for k, v in self.teachers.items():
            return_dict[k] = v.success_check(state, action, oracle[k])
        return return_dict