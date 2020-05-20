import numpy as np

class BatchTeacher:
    """
    Batched version of the Teacher class.  # TODO: I think there's a way to make it call the func on each teacher by default.
    """
    def __init__(self, teachers):
        self.teachers = teachers

    def step(self, agent_action):
        [teacher.step(agent_action) for teacher in self.teachers]

    def give_feedback(self, state):
        feedback = [teacher.give_feedback(state) for teacher in self.teachers]
        return feedback

    def empty_feedback(self):
        return [teacher.empty_feedback() for teacher in self.teachers]

    def compute_feedback(self):
        return [teacher.compute_feedback() for teacher in self.teachers]

    def feedback_condition(self):
        return [teacher.feedback_condition() for teacher in self.teachers]

    def set_feedback_type(self, feedback_type):
        return [teacher.set_feedback_type(feedback_type) for teacher in self.teachers]

    def reset(self):
        [teacher.reset() for teacher in self.teachers]