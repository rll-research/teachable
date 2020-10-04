import numpy as np

class BatchTeacher:
    """
    Batched version of the Teacher class.
    """
    def __init__(self, teachers):
        self.teachers = teachers

    def step(self, action, oracle):
        return [teacher.step(action, o) for teacher, o in zip(self.teachers, oracle)]

    def give_feedback(self, state):
        return np.concatenate([teacher.give_feedback(state) for teacher in self.teachers])

    def empty_feedback(self):
        return np.concatenate([teacher.empty_feedback() for teacher in self.teachers])

    def compute_feedback(self):
        return np.concatenate([teacher.compute_feedback() for teacher in self.teachers])

    def feedback_condition(self):
        return np.any(np.array([teacher.feedback_condition() for teacher in self.teachers]))

    def set_feedback_type(self, feedback_type):
        return [teacher.set_feedback_type(feedback_type) for teacher in self.teachers]

    def reset(self, oracle):
        return [teacher.reset(o) for teacher, o in zip(self.teachers, oracle)]

    def get_last_step_error(self):
        last_step_error = [t.last_step_error for t in self.teachers]
        last_step_error = np.max(last_step_error)
        return last_step_error