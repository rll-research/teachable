"""
Levels described in the ICLR 2019 submission.
"""
import gym
from .verifier import *
from .levelgen import *
from .teachable_robot_levels import Level_TeachableRobot
from meta_mb.meta_envs.base import MetaEnv
from gym_minigrid.minigrid import MiniGridEnv, Key, Ball, Box

class Level_GoToRedBallGrey(Level_TeachableRobot):
    """
    Go to the red ball, single room, with distractors.
    The distractors are all grey to reduce perceptual complexity.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None, **kwargs):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            num_dists=num_dists,
            seed=seed,
            **kwargs
        )

    def make_mission(self):
        return {
            "task": ["ball", "red"],
            "instrs": GoToInstr(ObjDesc("ball", "red"))
        }

    def add_objs(self, _):
        obj, _ = self.add_object(0, 0, 'ball', 'red')
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        for dist in dists:
            dist.color = 'grey'
        return dists + [obj], obj


class Level_GoToRedBall(Level_TeachableRobot):
    """
    Go to the red ball, single room, with distractors.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None, **kwargs):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def make_mission(self):
        return {
            "task": ["ball", "red"],
            "instrs": GoToInstr(ObjDesc("ball", "red"))
        }

    def add_objs(self, _):
        obj, _ = self.add_object(0, 0, 'ball', 'red')
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        return dists + [obj], obj


class Level_GoToRedBallNoDists(Level_GoToRedBall):
    """
    Go to the red ball. No distractors present.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=8, num_dists=0, seed=seed, **kwargs)


class Level_GoToObj(Level_TeachableRobot):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, seed=None, **kwargs):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def make_mission(self):
        obj_type, obj_color = self.sample_object()
        return {
            "task": (obj_type, obj_color),
            "instrs": GoToInstr(ObjDesc(obj_type, obj_color))
        }

    def add_objs(self, task):
        obj_type, obj_color = task
        obj, _ = self.add_object(0, 0, obj_type, obj_color)
        return [obj], obj


class Level_GoToObjS4(Level_GoToObj):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=4, seed=seed, **kwargs)


class Level_GoToObjS6(Level_GoToObj):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, seed=seed, **kwargs)


class Level_GoToLocal(Level_TeachableRobot):
    """
    Go to an object, inside a single room with no doors, multiple distractors
    """

    def __init__(self, room_size=8, num_dists=8, seed=None, **kwargs):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def make_mission(self):
        obj_type, obj_color = self.sample_object()
        return {
            "task": (obj_type, obj_color),
            "instrs": GoToInstr(ObjDesc(obj_type, obj_color))
        }

    def add_objs(self, task):
        obj_type, obj_color = task
        obj, _ = self.add_object(0, 0, obj_type, obj_color)
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        return dists + [obj], obj


class Level_GoToLocalS5N2(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, num_dists=2, seed=seed, **kwargs)


class Level_GoToLocalS6N2(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, num_dists=2, seed=seed, **kwargs)


class Level_GoToLocalS6N3(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, num_dists=3, seed=seed, **kwargs)


class Level_GoToLocalS6N4(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, num_dists=4, seed=seed, **kwargs)


class Level_GoToLocalS7N4(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=7, num_dists=4, seed=seed, **kwargs)


class Level_GoToLocalS7N5(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=7, num_dists=5, seed=seed, **kwargs)


class Level_GoToLocalS8N2(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=8, num_dists=2, seed=seed, **kwargs)


class Level_GoToLocalS8N3(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=8, num_dists=3, seed=seed, **kwargs)


class Level_GoToLocalS8N4(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=8, num_dists=4, seed=seed, **kwargs)


class Level_GoToLocalS8N5(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=8, num_dists=5, seed=seed, **kwargs)


class Level_GoToLocalS8N6(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=8, num_dists=6, seed=seed, **kwargs)


class Level_GoToLocalS8N7(Level_GoToLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=8, num_dists=7, seed=seed, **kwargs)


class Level_PutNextLocal(Level_TeachableRobot):
    """
    Put an object next to another object, inside a single room
    with no doors, no distractors
    """

    def __init__(self, room_size=8, num_dists=8, seed=None, **kwargs):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def make_mission(self):
        o1_type, o1_color = self.sample_object()
        o2_type, o2_color = o1_type, o1_color
        while (o1_type == o2_type and o1_color == o2_color):
            o2_type, o2_color = self.sample_object()
        return {
            "task": (o1_type, o1_color, o2_type, o2_color),
            "instrs": PutNextInstr(ObjDesc(o1_type, o1_color), ObjDesc(o2_type, o2_color))
        }

    def add_objs(self, task):
        o1_type, o1_color, o2_type, o2_color = task
        obj1, _ = self.add_object(0, 0, o1_type, o1_color)
        obj2, _ = self.add_object(0, 0, o2_type, o2_color)
        dists = self.add_distractors(num_distractors=self.num_dists - 2, all_unique=True)
        return dists + [obj1, obj2], (obj1, obj2)


class Level_PutNextLocalS5N3(Level_PutNextLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, num_dists=3, seed=seed, **kwargs)


class Level_PutNextLocalS6N4(Level_PutNextLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, num_dists=4, seed=seed, **kwargs)


class Level_GoTo(Level_TeachableRobot):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        doors_open=False,
        seed=None,
        **kwargs
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def make_mission(self):
        obj_type, obj_color = self.sample_object()
        return {
            "task": (obj_type, obj_color),
            "instrs": GoToInstr(ObjDesc(obj_type, obj_color))
        }

    def add_objs(self, task):
        self.connect_all()
        obj_type, obj_color = task
        # Choose room
        room_i = self._rand_int(0, self.num_rows)
        room_j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(room_i, room_j, obj_type, obj_color)
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()
        return dists + self.get_doors() + [obj], obj


class Level_GoToOpen(Level_GoTo):
    def __init__(self, seed=None, **kwargs):
        super().__init__(doors_open=True, seed=seed, **kwargs)


class Level_GoToObjMaze(Level_GoTo):
    """
    Go to an object, the object may be in another room. No distractors.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=1, doors_open=False, seed=seed, **kwargs)


class Level_GoToObjMazeOpen(Level_GoTo):
    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=1, doors_open=True, seed=seed, **kwargs)


class Level_GoToObjMazeS4R2(Level_GoTo):
    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=1, room_size=4, num_rows=2, num_cols=2, seed=seed, **kwargs)


class Level_GoToObjMazeS4(Level_GoTo):
    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=1, room_size=4, seed=seed, **kwargs)


class Level_GoToObjMazeS5(Level_GoTo):
    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=1, room_size=5, seed=seed, **kwargs)


class Level_GoToObjMazeS6(Level_GoTo):
    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=1, room_size=6, seed=seed, **kwargs)


class Level_GoToObjMazeS6(Level_GoTo):
    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=1, room_size=7, seed=seed, **kwargs)


class Level_GoToImpUnlock(Level_TeachableRobot):
    """
    Go to an object, which may be in a locked room.
    Competencies: Maze, GoTo, ImpUnlock
    No unblocking.
    """

    def make_mission(self):
        obj_type, obj_color = self.sample_object()
        return {
            "task": (obj_type, obj_color),
            "instrs": GoToInstr(ObjDesc(obj_type, obj_color))
        }

    def add_objs(self, task):

        obj_type, obj_color = task

        while True:
            id = self._rand_int(0, self.num_rows)
            jd = self._rand_int(0, self.num_cols)
            locked_room = self.get_room(id, jd)
            agent_room = self.room_from_pos(*self.agent_pos)
            if not locked_room is agent_room:
                break
        door, pos = self.add_door(id, jd, locked=True)
        door.color = obj_color

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        all_dists = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i is not id or j is not jd:
                    dists = self.add_distractors(
                        i,
                        j,
                        num_distractors=2,
                        all_unique=False
                    )
                    all_dists += dists

        self.check_objs_reachable()

        obj, _ = self.add_object(id, jd, obj_type, obj_color)
        return all_dists + self.get_doors() + [obj], obj


class Level_Pickup(Level_TeachableRobot):
    """
    Pick up an object, the object may be in another room.
    """

    def make_mission(self):
        obj_type, obj_color = self.sample_object()
        return {
            "task": (obj_type, obj_color),
            "instrs": PickupInstr(ObjDesc(obj_type, obj_color))
        }

    def add_objs(self, task):
        self.connect_all()
        room_i = self._rand_int(0, self.num_rows)
        room_j = self._rand_int(0, self.num_cols)
        obj_type, obj_color = task
        obj, _ = self.add_object(room_i, room_j, obj_type, obj_color)
        dists = self.add_distractors(num_distractors=17, all_unique=False)
        self.check_objs_reachable()
        return dists + self.get_doors() + [obj], obj



class Level_UnblockPickup(Level_TeachableRobot):
    """
    Pick up an object, the object may be in another room. The path may
    be blocked by one or more obstructors.
    """
    def make_mission(self):
        obj_type, obj_color = self.sample_object()
        return {
            "task": (obj_type, obj_color),
            "instrs": PickupInstr(ObjDesc(obj_type, obj_color))
        }

    def add_objs(self, task):
        self.connect_all()
        room_i = self._rand_int(0, self.num_rows)
        room_j = self._rand_int(0, self.num_cols)
        obj_type, obj_color = task
        obj, _ = self.add_object(room_i, room_j, obj_type, obj_color)
        dists = self.add_distractors(num_distractors=19, all_unique=False)
        if self.check_objs_reachable(raise_exc=False):
            raise RejectSampling('all objects reachable')
        return dists + self.get_doors() + [obj], obj


class Level_Open(Level_TeachableRobot):
    """
    Open a door, which may be in another room
    """
    def make_mission(self):
        # We only need the color
        _, obj_color = self.sample_object()
        return {
            "task": obj_color,
            "instrs": OpenInstr(ObjDesc("door", obj_color))
        }


    def add_objs(self, task):
        obj_color = task
        self.connect_all()
        dists = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()

        # Make sure at least one door has the required color by randomly setting one door color
        doors = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)

        door = self._rand_elem(doors)
        door.color = obj_color
        return dists + self.get_doors(), door


class Level_Unlock(Level_TeachableRobot):
    """
    Unlock a door.

    Competencies: Maze, Open, Unlock. No unblocking.
    """
    def make_mission(self):
        _, obj_color = self.sample_object()
        return {
            "task": obj_color,
            "instrs": OpenInstr(ObjDesc("door", obj_color))
        }


    def add_objs(self, task):

        obj_color = task

        while True:
            id = self._rand_int(0, self.num_rows)
            jd = self._rand_int(0, self.num_cols)
            locked_room = self.get_room(id, jd)
            agent_room = self.room_from_pos(*self.agent_pos)
            if not locked_room is agent_room:
                break
        door, pos = self.add_door(id, jd, locked=True)
        door.color = obj_color

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        # With 50% probability, ensure that the locked door is the only
        # door of that color
        if self._rand_bool():
            colors = list(filter(lambda c: c is not obj_color, COLOR_NAMES))
            self.connect_all(door_colors=colors)
        else:
            self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        all_dists = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i is not id or j is not jd:
                    dists = self.add_distractors(
                        i,
                        j,
                        num_distractors=3,
                        all_unique=False
                    )
                    all_dists += dists

        self.check_objs_reachable()
        return all_dists + self.get_doors(), door


class Level_PutNext(Level_TeachableRobot):
    """
    Put an object next to another object. Either of these may be in another room.
    """

    def __init__(self, room_size=8, num_objs=8, seed=None, **kwargs):
        self.num_objs = num_objs
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def make_mission(self):
        o1_type, o1_color = self.sample_object()
        o2_type, o2_color = o1_type, o1_color
        while (o1_type == o2_type and o1_color == o2_color):
            o2_type, o2_color = self.sample_object()
        return {
            "task": (o1_type, o1_color, o2_type, o2_color),
            "instrs": PutNextInstr(ObjDesc(o1_type, o1_color), ObjDesc(o2_type, o2_color))
        }

    def add_objs(self, task):
        self.connect_all()
        o1_type, o1_color, o2_type, o2_color = task
        obj1, _ = self.add_object(0, 0, o1_type, o1_color)
        obj2, _ = self.add_object(0, 0, o2_type, o2_color)
        dists = self.add_distractors(num_distractors=16 - 2, all_unique=True)
        self.check_objs_reachable()
        return dists + self.get_doors() + [obj1, obj2], (obj1, obj2)


class Level_PickupLoc(LevelGen):
    """
    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.
    """

    def __init__(self, seed=None, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False,
            **kwargs
        )


class Level_GoToSeq(LevelGen):
    """
    Sequencing of go-to-object commands.

    Competencies: Maze, GoTo, Seq
    No locked room.
    No locations.
    No unblocking.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        seed=None,
        **kwargs
    ):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['goto'],
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            **kwargs
        )


class Level_GoToSeqS5R2(Level_GoToSeq):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, num_rows=2, num_cols=2, num_dists=4, seed=seed, **kwargs)


class Level_Synth(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        seed=None,
        **kwargs
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            instr_kinds=['action'],
            locations=False,
            unblocking=True,
            implicit_unlock=False,
            **kwargs
        )


class Level_SynthS5R2(Level_Synth):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            room_size=5,
            num_rows=2,
            num_cols=2,
            num_dists=7,
            seed=seed,
            **kwargs
        )


class Level_SynthLoc(LevelGen):
    """
    Like Synth, but a significant share of object descriptions involves
    location language like in PickUpLoc. No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc
    """

    def __init__(self, seed=None, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            instr_kinds=['action'],
            locations=True,
            unblocking=True,
            implicit_unlock=False,
            **kwargs
        )


class Level_SynthSeq(LevelGen):
    """
    Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.
    No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq
    """

    def __init__(self, seed=None, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            locations=True,
            unblocking=True,
            implicit_unlock=False,
            **kwargs
        )


class Level_MiniBossLevel(LevelGen):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            seed=seed,
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25,
            **kwargs
        )


class Level_BossLevel(LevelGen):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            seed=seed,
            **kwargs
        )


class Level_BossLevelNoUnlock(LevelGen):
    def __init__(self, seed=None, **kwargs):
        super().__init__(
            seed=seed,
            locked_room_prob=0,
            implicit_unlock=False,
            **kwargs
        )


# Register the levels in this file
register_levels(__name__, globals())
