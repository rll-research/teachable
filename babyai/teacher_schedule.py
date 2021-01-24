import copy


# A distillation scheme is represented by a function which takes in a level number and returns a dictionary specifying
# What teachers the agent should train on and distill to
# For the moment, we hard-code the levels, although later it might be nice to determine which teachers to use
# Based on the success/reward from the past itr

#### NO TEACHER ####
def no_teacher(level, teacher_list):
    no_teacher_dict = {}
    for teacher in teacher_list:
        no_teacher_dict[teacher] = False
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    return no_teacher_dict, copy.deepcopy(no_teacher_dict)


#### SINGLE TEACHER ####
def all_teachers(level, teacher_list):
    no_teacher_dict = {t: False for t in teacher_list}
    teacher_train_dict = {t: True for t in teacher_list}
    distillation_dict = {t: True for t in teacher_list}
    if level == -1:
        return no_teacher_dict, None
    return teacher_train_dict, distillation_dict


#### FIRST TEACHER ####
# Train on the first teacher, distill to all the others
def first_teacher(level, teacher_list):
    no_teacher_dict = {t: False for t in teacher_list}
    first = teacher_list[0]
    teacher_train_dict = {t: t == first for t in teacher_list}
    distillation_dict = {t: not t == first for t in teacher_list}
    if level == -1:
        return no_teacher_dict, None
    return teacher_train_dict, distillation_dict


#### Last TEACHER ####
# Train on the last teacher, distill to all the others
def last_teacher(level, teacher_list):
    no_teacher_dict = {t: False for t in teacher_list}
    last = teacher_list[-1]
    teacher_train_dict = {t: t == last for t in teacher_list}
    distillation_dict = {t: not t == last for t in teacher_list}
    if level == -1:
        return no_teacher_dict, None
    return teacher_train_dict, distillation_dict


#### FIRST TEACHER ####
# Train on the first teacher, distill to first
def train_first_distill_first(level, teacher_list):
    no_teacher_dict = {t: False for t in teacher_list}
    first = teacher_list[0]
    teacher_train_dict = {t: t == first for t in teacher_list}
    distillation_dict = {t: t == first for t in teacher_list}
    if level == -1:
        return no_teacher_dict, None
    return teacher_train_dict, distillation_dict


# Train on the first teacher, distill to the second
def train_first_advance_second(level, easy_teacher, harder_teacher):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    teacher_train_dict = {easy_teacher: True, harder_teacher: False}
    distillation_dict = {easy_teacher: False, harder_teacher: True}
    return teacher_train_dict, distillation_dict


### PREACTION TO ONE OTHER, SWAP OUT ####
# Add in the second teacher ...
def easy_swap_harder(level, easy_teacher, harder_teacher, remove_easy_level=13):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    elif level < remove_easy_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
    distillation_dict = {easy_teacher: False, harder_teacher: True}
    return teacher_train_dict, distillation_dict


def easy_swap_harder_help(level, success_rate, accuracy_rate, easy_teacher, harder_teacher,
                          success_intervention_cutoff=.95, accuracy_intervention_cutoff=.95,
                          remove_easy_level=13):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    elif level < remove_easy_level or (success_rate < success_intervention_cutoff) or \
        (accuracy_rate < accuracy_intervention_cutoff):
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
    distillation_dict = {easy_teacher: False, harder_teacher: True}
    return teacher_train_dict, distillation_dict


def easy_swap_harder_each_time(level, success_rate, accuracy_rate, easy_teacher, harder_teacher,
                               success_intervention_cutoff=1, accuracy_intervention_cutoff=.95):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    # If success rate is bad, re-introduce the easy teacher
    if (success_rate < success_intervention_cutoff) or (accuracy_rate < accuracy_intervention_cutoff):
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
    distillation_dict = {easy_teacher: False, harder_teacher: True}
    return teacher_train_dict, distillation_dict


def make_teacher_schedule(feedback_types, teacher_schedule):
    feedback_types = [teacher for teacher in feedback_types if not teacher == 'None']
    if teacher_schedule == 'none':
        return lambda level, a, b: no_teacher(level, feedback_types)
    elif teacher_schedule == 'all_teachers':
        return lambda level, a, b: all_teachers(level, feedback_types)
    elif teacher_schedule == 'first_teacher':
        return lambda level, a, b: first_teacher(level, feedback_types)
    elif teacher_schedule == 'last_teacher':
        return lambda level, a, b: last_teacher(level, feedback_types)
    elif teacher_schedule == 'first_teacher_both':
        return lambda level, a, b: first_teacher(level, feedback_types)
    elif teacher_schedule == 'train_first_advance_second':
        assert len(feedback_types) == 2
        return lambda level, a, b: train_first_advance_second(level, feedback_types[0], feedback_types[1])
    elif teacher_schedule == 'easy_swap_harder':
        assert len(feedback_types) == 2
        return lambda level, a, b: easy_swap_harder(level, feedback_types[0], feedback_types[1])
    elif teacher_schedule == 'easy_swap_harder_noselfdistill':
        assert len(feedback_types) == 2
        return lambda level, a, b: easy_swap_harder(level, feedback_types[0], feedback_types[1])
    elif teacher_schedule == 'easy_swap_harder_each_time':
        assert len(feedback_types) == 2
        return lambda level, success_rate, accuracy_rate: easy_swap_harder_each_time(level, success_rate, accuracy_rate,
                                                                                     feedback_types[0],
                                                                                     feedback_types[1])
    elif teacher_schedule == 'easy_swap_harder_help':
        assert len(feedback_types) == 2
        return lambda level, success_rate, accuracy_rate: easy_swap_harder_help(level, success_rate, accuracy_rate,
                                                                                feedback_types[0],
                                                                                feedback_types[1])
    else:
        raise ValueError(f'Unknown distillation scheme {teacher_schedule}, {feedback_types}')
