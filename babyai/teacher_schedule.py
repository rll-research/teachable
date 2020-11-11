import copy


# A distillation scheme is represented by a function which takes in a level number and returns a dictionary specifying
# What teachers the agent should train on and distill to
# For the moment, we hard-code the levels, although later it might be nice to determine which teachers to use
# Based on the success/reward from the past itr

#### NO TEACHER ####
def no_teacher(level):
    return {}, {}


#### SINGLE TEACHER ####
def single_teacher(level, teacher_name):
    if level == -1:  # Generate no_teacher_dict
        return {teacher_name: False}
    teacher_train_dict = {teacher_name: True}
    distillation_dict = copy.deepcopy(teacher_train_dict)
    return teacher_train_dict, distillation_dict


#### SINGLE TEACHER, NO POWERSET ####
def single_teacher_no_powerset(level, teacher_name):
    if level == -1:  # Generate no_teacher_dict
        return {teacher_name: False}
    teacher_train_dict = {teacher_name: True}
    distillation_dict = {teacher_name: False}
    return teacher_train_dict, distillation_dict


### PREACTION TO ONE OTHER ####
# Add in the second teacher ...
def easy_add_harder(level, easy_teacher, harder_teacher, cutoff_level=21):
    if level == -1:  # Generate no_teacher_dict
        return {easy_teacher: False, harder_teacher: False}
    if level < cutoff_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
    else:
        teacher_train_dict = {easy_teacher: True, harder_teacher: True}
    distillation_dict = copy.deepcopy(teacher_train_dict)
    return teacher_train_dict, distillation_dict


### PREACTION TO ONE OTHER, SWAP OUT ####
# Add in the second teacher ...
def easy_replace_harder(level, easy_teacher, harder_teacher, add_hard_level=15, remove_easy_level=21):
    if level == -1:  # Generate no_teacher_dict
        return {easy_teacher: False, harder_teacher: False}
    if level < add_hard_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    elif level < remove_easy_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: True}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
        distillation_dict = {easy_teacher: True, harder_teacher: True}
    return teacher_train_dict, distillation_dict


def make_teacher_schedule(feedback_types, teacher_schedule):
    if teacher_schedule == 'none':
        assert len(feedback_types) == 0
        return no_teacher
    elif teacher_schedule == 'single_teacher':
        assert len(feedback_types) == 1
        return lambda level: single_teacher(level, feedback_types[0])
    elif teacher_schedule == 'single_teacher_no_powerset':
        assert len(feedback_types) == 1
        return lambda level: single_teacher_no_powerset(level, feedback_types[0])
    elif teacher_schedule == 'easy_add_harder':
        assert len(feedback_types) == 2
        return lambda level: easy_add_harder(level, feedback_types[0], feedback_types[1])
    elif teacher_schedule == 'easy_swap_harder':
        assert len(feedback_types) == 2
        return lambda level: easy_add_harder(level, feedback_types[0], feedback_types[1])
    else:
        raise ValueError(f'Unknown distillation scheme {teacher_schedule}, {feedback_types}')
