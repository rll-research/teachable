"""
Levels described in the ICLR 2019 submission.
"""
import gym
from .verifier import *
from .levelgen import *
from .teachable_robot_levels import Level_TeachableRobot
from meta_mb.meta_envs.base import MetaEnv
from gym_minigrid.minigrid import MiniGridEnv, Key, Ball, Box


class Level_IntroPrimitives(Level_TeachableRobot):
    """
    Get a reward for doing what the teacher tells you to do.
    """

    def __init__(self, max_delay=0, room_size=8, strict=False, seed=None, **kwargs):
        self.room_size = room_size
        self.max_delay = max_delay
        self.strict = strict
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            **kwargs
        )

    def make_mission(self):
        action = self.action_space.sample()
        delay = self.np_random.randint(0, self.max_delay + 1)
        return {
            "task": (action, delay),
            "instrs": TakeActionInstr(action, delay, self.strict)
        }

    def add_objs(self, task):
        num_dists = self.np_random.randint(0, min(20, (self.room_size - 3) ** 2))
        dists = self.add_distractors(num_distractors=num_dists, all_unique=False)
        return dists, None


class Level_IntroPrimitivesD0(Level_IntroPrimitives):
    def __init__(self, seed=None, **kwargs):
        super().__init__(max_delay=0, seed=seed, **kwargs)


class Level_IntroPrimitivesD1(Level_IntroPrimitives):
    def __init__(self, seed=None, **kwargs):
        super().__init__(max_delay=1, seed=seed, **kwargs)


class Level_IntroPrimitivesD5(Level_IntroPrimitives):
    def __init__(self, seed=None, **kwargs):
        super().__init__(max_delay=5, seed=seed, **kwargs)


class Level_IntroPrimitivesD10(Level_IntroPrimitives):
    def __init__(self, seed=None, **kwargs):
        super().__init__(max_delay=10, seed=seed, **kwargs)


class Level_IntroPrimitivesD0Strict(Level_IntroPrimitives):
    def __init__(self, seed=None, **kwargs):
        super().__init__(max_delay=0, seed=seed, strict=True, **kwargs)


class Level_IntroPrimitivesD1Strict(Level_IntroPrimitives):
    def __init__(self, seed=None, **kwargs):
        super().__init__(max_delay=1, seed=seed, strict=True, **kwargs)


class Level_IntroPrimitivesD5Strict(Level_IntroPrimitives):
    def __init__(self, seed=None, **kwargs):
        super().__init__(max_delay=5, seed=seed, strict=True, **kwargs)


class Level_IntroPrimitivesD10Strict(Level_IntroPrimitives):
    def __init__(self, seed=None, **kwargs):
        super().__init__(max_delay=10, seed=seed, strict=True, **kwargs)


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
        self.check_objs_reachable()
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
        self.check_objs_reachable()
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


class Level_GoToObjS5(Level_GoToObj):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, seed=seed, **kwargs)


class Level_GoToObjS6(Level_GoToObj):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, seed=seed, **kwargs)


class Level_GoToObjS7(Level_GoToObj):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=7, seed=seed, **kwargs)


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
        self.check_objs_reachable()
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


class Level_PickupLocal(Level_TeachableRobot):
    """
    Pick up an object, inside a single room with no doors, multiple distractors
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
            "instrs": PickupInstr(ObjDesc(obj_type, obj_color))
        }

    def add_objs(self, task):
        obj_type, obj_color = task
        obj, _ = self.add_object(0, 0, obj_type, obj_color)
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        return dists + [obj], obj


class Level_PickupLocalS5N2(Level_PickupLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, num_dists=2, seed=seed, **kwargs)


class Level_PickupLocalS6N3(Level_PickupLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, num_dists=3, seed=seed, **kwargs)


class Level_PickupLocalS7N4(Level_PickupLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=7, num_dists=4, seed=seed, **kwargs)


class Level_PickupLocalS8N7(Level_PickupLocal):
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
        self.check_objs_reachable()
        return dists + [obj1, obj2], (obj1, obj2)


class Level_PutNextLocalS5N3(Level_PutNextLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, num_dists=3, seed=seed, **kwargs)


class Level_PutNextLocalS5N2(Level_PutNextLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, num_dists=2, seed=seed, **kwargs)


class Level_PutNextLocalS6N4(Level_PutNextLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, num_dists=4, seed=seed, **kwargs)


class Level_PutNextLocalS7N4(Level_PutNextLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=7, num_dists=4, seed=seed, **kwargs)


class Level_OpenLocal(Level_TeachableRobot):
    """
    Open a door in the current room (0,0), since that's currently where we initialize the agent.
    """

    def __init__(self, room_size=8, num_dists=8, seed=None, **kwargs):
        self.num_dists = num_dists
        super().__init__(
            room_size=room_size,
            seed=seed,
            **kwargs
        )

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
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Make sure at least one door has the required color by randomly setting one door color
        doors = []
        room = self.get_room(0, 0)
        for door in room.doors:
            if door:
                doors.append(door)
        door = self._rand_elem(doors)
        door.color = obj_color
        self.check_objs_reachable()
        return dists + self.get_doors(), door


class Level_OpenLocalS5N2(Level_OpenLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, num_dists=2, seed=seed, **kwargs)


class Level_OpenLocalS5N3(Level_OpenLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=5, num_dists=3, seed=seed, **kwargs)


class Level_OpenLocalS6N4(Level_OpenLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=6, num_dists=4, seed=seed, **kwargs)


class Level_OpenLocalS7N4(Level_OpenLocal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(room_size=7, num_dists=4, seed=seed, **kwargs)


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
        self.check_objs_reachable()
        return dists + self.get_doors() + [obj], obj


class Level_Seek(Level_GoTo):
    def make_mission(self):
        obj_type, obj_color = self.sample_object()
        return {
            "task": (obj_type, obj_color),
            "instrs": SeekInstr(ObjDesc(obj_type, obj_color))
        }


class Level_GoToHeldout(Level_GoTo):
    def make_mission(self):
        obj_type, obj_color = self.sample_object()
        obj_color = 'yellow'
        return {
            "task": (obj_type, obj_color),
            "instrs": GoToInstr(ObjDesc(obj_type, obj_color))
        }


class Level_GoToGreenBox(Level_GoTo):
    def make_mission(self):
        obj_color = 'green'
        obj_type = 'box'
        return {
            "task": (obj_type, obj_color),
            "instrs": GoToUnknownInstr(ObjDesc(obj_type, obj_color))
        }


class Level_GoToDouble(Level_GoTo):
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
        obj1_type, obj1_color = self.sample_object()
        obj2_type, obj2_color = self.sample_object()
        return {
            "task": (obj1_type, obj1_color, obj2_type, obj2_color),
            "instrs": BeforeInstr(
                GoToInstr(ObjDesc(obj1_type, obj1_color)),
                GoToInstr(ObjDesc(obj2_type, obj2_color))
            )
        }

    def add_objs(self, task):
        self.connect_all()
        obj1_type, obj1_color, obj2_type, obj2_color = task
        # Choose room
        room_i = self._rand_int(0, self.num_rows)
        room_j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(room_i, room_j, obj1_type, obj1_color)
        room_i = self._rand_int(0, self.num_rows)
        room_j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(room_i, room_j, obj2_type, obj2_color)
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()
        self.check_objs_reachable()
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

class Level_GoToObjDistractors(Level_GoTo):
    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=60, room_size=8, seed=seed, **kwargs)

    def add_objs(self, task):
        obj_list, obj = super().add_objs(task)
        obj_type = obj.type
        obj_color = obj.color
        for dist in obj_list[:-1]:
            if dist.type == obj_type and dist.color == obj_color:
                dist.color = self._rand_elem([c for c in COLOR_NAMES if not c == obj_color])
        return obj_list, obj

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
            if not (locked_room is agent_room):
                break
        _, color = self.sample_object()
        door, pos = self.add_door(id, jd, color=color, locked=True)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik == id and jk == jd:
                continue
            key, _ = self.add_object(ik, jk, 'key', door.color)
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
                        all_unique=False,
                    )
                    all_dists += dists
        for dist in all_dists:
            if dist.type == obj_type and dist.color == obj_color:
                dist.color = self._rand_elem([c for c in COLOR_NAMES if not c == obj_color])

        self.check_objs_reachable()

        obj, _ = self.add_object(id, jd, obj_type, obj_color)
        return all_dists + self.get_doors() + [obj, key], obj


class Level_Pickup(Level_TeachableRobot):
    """
    Pick up an object, the object may be in another room.
    """
    def __init__(
        self,
        num_dists=18,
        doors_open=False,
        **kwargs
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(**kwargs)

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

class Level_PickupObjBigger(Level_Pickup):
    def __init__(self, seed=None, **kwargs):
        super().__init__(num_dists=18, room_size=6, num_rows=5, num_cols=5, seed=seed, **kwargs)

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
        dists = self.add_distractors(num_distractors=39, all_unique=False)
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


class Level_OpenDoorsDouble(Level_TeachableRobot):
    """
    Open door X, then open door Y
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """
    def make_mission(self):
        # We only need the color
        colors = self._rand_subset(COLOR_NAMES, 2)
        return {
            "task": colors,
            "instrs": BeforeInstr(
                OpenInstr(ObjDesc('door', colors[0])),
                OpenInstr(ObjDesc('door', colors[1]))
            )
        }

    def add_objs(self, task):
        first_color, second_color = task
        door1, _ = self.add_door(1, 1, 2, color=first_color, locked=False)
        door2, _ = self.add_door(1, 1, 0, color=second_color, locked=False)
        self.connect_all()
        dists = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        return dists + self.get_doors(), (door1, door2)


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
        door, pos = self.add_door(id, jd, color=obj_color, locked=True)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik == id and jk == jd:
                continue
            key, _ = self.add_object(ik, jk, 'key', door.color)
            break

        # Ensure that the locked door is the only
        # door of that color
        colors = list(filter(lambda c: not c == obj_color, COLOR_NAMES))
        self.connect_all(door_colors=colors)

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
        return [key] + all_dists + self.get_doors(), door


class Level_PutNext(Level_TeachableRobot):
    """
    Put an object next to another object. Either of these may be in another room.
    """

    def __init__(self, room_size=8, num_dists=16, seed=None, **kwargs):
        self.num_dists = num_dists
        super().__init__(
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
        i1 = self._rand_int(0, self.num_rows)
        j1 = self._rand_int(0, self.num_cols)
        i2 = self._rand_int(0, self.num_rows)
        j2 = self._rand_int(0, self.num_cols)
        obj1, _ = self.add_object(i1, j1, o1_type, o1_color)
        obj2, _ = self.add_object(i2, j2, o2_type, o2_color)
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        return dists + self.get_doors() + [obj1, obj2], (obj1, obj2)


class Level_PutNextSameColor(Level_PutNext):
    def make_mission(self):
        o1_type, o1_color = self.sample_object()
        o2_type, o2_color = o1_type, o1_color
        while o1_type == o2_type:
            o2_type, o2_color = self.sample_object()
        return {
            "task": (o1_type, o1_color, o2_type, o1_color),
            "instrs": PutNextSameColorInstr(ObjDesc(o1_type, o1_color), ObjDesc(o2_type, o1_color))
        }


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
