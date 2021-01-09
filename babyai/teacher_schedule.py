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
    return no_teacher_dict, copy.deepcopy(no_teacher_dict), copy.deepcopy(no_teacher_dict)


#### SINGLE TEACHER ####
def all_teachers(level, teacher_list):
    no_teacher_dict = {}
    teacher_train_dict = {}
    for teacher in teacher_list:
        no_teacher_dict[teacher] = False
        teacher_train_dict[teacher] = True
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    distillation_dict = copy.deepcopy(teacher_train_dict)
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


#### FIRST TEACHER ####
# Train on the first teacher, distill to all the others
def first_teacher(level, teacher_list):
    no_teacher_dict = {}
    distillation_dict = {}
    for teacher in teacher_list:
        no_teacher_dict[teacher] = False
        distillation_dict[teacher] = True
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    teacher_train_dict = copy.deepcopy(no_teacher_dict)
    teacher_train_dict[teacher_list[0]] = True
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


#### Last TEACHER ####
# Train on the last teacher, distill to all the others
def last_teacher(level, teacher_list):
    no_teacher_dict = {}
    distillation_dict = {}
    for teacher in teacher_list:
        no_teacher_dict[teacher] = False
        distillation_dict[teacher] = True
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    teacher_train_dict = copy.deepcopy(no_teacher_dict)
    teacher_train_dict[teacher_list[-1]] = True
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


#### FIRST TEACHER ####
# Train on the first teacher, distill to first
def train_first_advance_first(level, teacher_list):
    no_teacher_dict = {}
    for teacher in teacher_list:
        no_teacher_dict[teacher] = False
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    teacher_train_dict = copy.deepcopy(no_teacher_dict)
    teacher_train_dict[teacher_list[0]] = True
    advancement_dict = copy.deepcopy(teacher_train_dict)
    distillation_dict = copy.deepcopy(teacher_train_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


# Train on the first teacher, distill to the second, advance with the second
def train_first_advance_second(level, easy_teacher, harder_teacher):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    teacher_train_dict = {easy_teacher: True, harder_teacher: False}
    advancement_dict = {easy_teacher: False, harder_teacher: True}
    distillation_dict = {easy_teacher: True, harder_teacher: True}
    return teacher_train_dict, distillation_dict, advancement_dict


#### FIRST TEACHER ####
# Train and distill on the first teacher
def first_teacher_both(level, teacher_list):
    no_teacher_dict = {}
    for teacher in teacher_list:
        no_teacher_dict[teacher] = False
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    teacher_train_dict = copy.deepcopy(no_teacher_dict)
    teacher_train_dict[teacher_list[0]] = True
    distillation_dict = copy.deepcopy(teacher_train_dict)
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


#### SINGLE TEACHER, NO POWERSET ####
def single_teacher_no_powerset(level, teacher_name):
    no_teacher_dict = {teacher_name: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    teacher_train_dict = {teacher_name: True}
    distillation_dict = {teacher_name: False}
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


### PREACTION TO ONE OTHER ####
# Add in the second teacher ...
def easy_add_harder(level, easy_teacher, harder_teacher, cutoff_level=8):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    if level < cutoff_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
    else:
        teacher_train_dict = {easy_teacher: True, harder_teacher: True}
    distillation_dict = copy.deepcopy(teacher_train_dict)
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


### PREACTION TO ONE OTHER, SWAP OUT ####
# Add in the second teacher ...
def easy_swap_harder(level, easy_teacher, harder_teacher, add_hard_level=3, remove_easy_level=13):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    if level < add_hard_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    elif level < remove_easy_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: True}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
        distillation_dict = {easy_teacher: True, harder_teacher: True}
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict

def easy_swap_harder_sparse(level, easy_teacher, harder_teacher, add_hard_level=3, remove_easy_level=13):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    if level < add_hard_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    elif level < remove_easy_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = {easy_teacher: True, harder_teacher: True}
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
        distillation_dict = {easy_teacher: False, harder_teacher: True}
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict

# Same as easy_swap_harder, but don't distill to the teacher you're training on
def easy_swap_harder_noselfdistill(level, easy_teacher, harder_teacher, add_hard_level=3, remove_easy_level=13):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    if level < add_hard_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = {easy_teacher: False, harder_teacher: False}
    elif level < remove_easy_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: True}
        distillation_dict = {easy_teacher: False, harder_teacher: True}
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
        distillation_dict = {easy_teacher: False, harder_teacher: False}
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


# Add in the second teacher ...
def easy_swap_harder_advance_harder(level, easy_teacher, harder_teacher, add_hard_level=3, remove_easy_level=13):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    if level < add_hard_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    elif level < remove_easy_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: True}
        distillation_dict = {easy_teacher: False, harder_teacher: True}
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
        distillation_dict = {easy_teacher: False, harder_teacher: True}
    advancement_dict = copy.deepcopy(distillation_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


def easy_swap_harder_help(level, success_rate, accuracy_rate, easy_teacher, harder_teacher,
                          success_intervention_cutoff=.95, accuracy_intervention_cutoff=.75,
                          add_hard_level=3, remove_easy_level=13):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    if level < add_hard_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    elif level < remove_easy_level or (success_rate < success_intervention_cutoff) or \
        (accuracy_rate < accuracy_intervention_cutoff):
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = {easy_teacher: True, harder_teacher: True}
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
        distillation_dict = {easy_teacher: True, harder_teacher: True}
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


def easy_swap_harder_each_time(level, success_rate, accuracy_rate, easy_teacher, harder_teacher,
                               success_intervention_cutoff=.95, accuracy_intervention_cutoff=.75):
    no_teacher_dict = {easy_teacher: False, harder_teacher: False}
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    # If success rate is bad, re-introduce the easy teacher
    if (success_rate < success_intervention_cutoff) or (accuracy_rate < accuracy_intervention_cutoff):
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = {easy_teacher: True, harder_teacher: True}
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
        distillation_dict = {easy_teacher: False, harder_teacher: True}
    advancement_dict = copy.deepcopy(no_teacher_dict)
    return teacher_train_dict, distillation_dict, advancement_dict


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
    elif teacher_schedule == 'train_first_advance_first':
        return lambda level, a, b: train_first_advance_first(level, feedback_types)
    elif teacher_schedule == 'first_teacher_both':
        return lambda level, a, b: first_teacher(level, feedback_types)
    elif teacher_schedule == 'single_teacher_no_powerset':
        assert len(feedback_types) == 1
        return lambda level, a, b: single_teacher_no_powerset(level, feedback_types[0])
    elif teacher_schedule == 'train_first_advance_second':
        assert len(feedback_types) == 2
        return lambda level, a, b: train_first_advance_second(level, feedback_types[0], feedback_types[1])
    elif teacher_schedule == 'easy_add_harder':
        assert len(feedback_types) == 2
        return lambda level, a, b: easy_add_harder(level, feedback_types[0], feedback_types[1])
    elif teacher_schedule == 'easy_swap_harder':
        assert len(feedback_types) == 2
        return lambda level, a, b: easy_swap_harder(level, feedback_types[0], feedback_types[1])
    elif teacher_schedule == 'easy_swap_harder_sparse':
        assert len(feedback_types) == 2
        return lambda level, a, b: easy_swap_harder_sparse(level, feedback_types[0], feedback_types[1])
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
    elif teacher_schedule == 'easy_swap_harder_advance_harder':
        assert len(feedback_types) == 2
        return lambda level: easy_swap_harder_advance_harder(level, feedback_types[0], feedback_types[1])
    else:
        raise ValueError(f'Unknown distillation scheme {teacher_schedule}, {feedback_types}')
