

class BatchTeacher:
    """
    Batched version of the Teacher class.
    """
    def __init__(self, teachers):
        self.teachers = teachers

    def step(self, agent_action_list):
        [teacher.step(action) for teacher, action in zip(self.teachers, agent_action_list)]

    def give_feedback(self, state_list):
        return [teacher.give_feedback(state) for teacher, state in zip(self.teachers, state_list)]

    def empty_feedback(self):
        return [teacher.empty_feedback() for teacher in self.teachers]

    def compute_feedback(self):
        return [teacher.compute_feedback() for teacher in self.teachers]

    def feedback_condition(self):
        return [teacher.feedback_condition() for teacher in self.teachers]
