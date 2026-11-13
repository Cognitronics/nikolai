"""A small, real text world – fast enough for a neural brain to practise in.

Five places, real things, real rules:
                 Forest (tree: branches, bast)
                    |
 Orchard ——— Hut ——— Workshop (workbench)
 (apple tree)       |
                 River (riverbank: stones)

Hunger grows over time; an apple (about 80 kcal) eases it a little. From a branch, a stone and bast you build a stone
axe at the workbench (as people did in the Stone Age). An apple tree doesn't bear without end: after picking, a new
apple grows slowly.

Version 6 – hidden knowledge: the tree alone is no longer enough. Now and then a windfall apple lies somewhere (in the
forest, at the river, in the workshop or in the hut) – each time somewhere else. The creature only sees it when it is
there. The teacher knows where it lies: whoever listens (or asks) finds food faster.

Version 10 – a friend: two creatures (NikolAI and Kogli) can share a world. Each has its own body (place, hunger,
backpack); ground, apple tree and windfall belong to both. What one puts down, the other can take; whoever finds the
windfall apple first gets it.

Version 11 – rest days: on some days the apple tree bears nothing at all (like a bad year). Then there are only
windfall apples – and whoever listens, asks or tells the friend gets full. Only for practice; the tests run without.

Version 17 – sizes: apples, branches and stones come big or small. A big apple fills more (0.5 instead of 0.3).
NikolAI sees how many big and small ones lie around and how many it carries. Its hands can choose: take, put down or
eat "any", "a big one" or "a small one" (a third part of each action, next to verb and thing). If the one it wants
isn't there, it takes the other – the choice only matters when both are there.

The creature perceives the world through its senses (a vector of numbers: where am I, what do I see, what do I carry,
how hungry am I, did my last action work?). At first it doesn't know what these numbers mean.
"""

import random

PLACES = ["hut", "orchard", "forest", "river", "workshop"]
PATHS = {"hut": {"west": "orchard", "north": "forest", "south": "river", "east": "workshop"},
        "orchard": {"east": "hut"}, "forest": {"south": "hut"}, "river": {"north": "hut"},
        "workshop": {"west": "hut"}}
DIRECTIONS = ["north", "south", "east", "west"]
THINGS = ["apple", "branch", "bast", "stone", "stone_axe"]
SOURCES = {"orchard": ["apple"], "forest": ["branch", "bast"], "river": ["stone"]}       # what can be fetched there
STATIONS = {"orchard": "apple_tree", "forest": "tree", "river": "riverbank", "workshop": "workbench", "hut": "bed"}
RECIPE = {"stone_axe": {"branch": 1, "stone": 1, "bast": 1}}                             # at the workbench
VERBS = ["go", "take", "eat", "put", "build", "wait", "say"]     # "say": say words instead of acting
OBJECTS = DIRECTIONS + THINGS
HUNGER_PER_STEP = 1 / 120
APPLE_FILLS = 0.4
APPLE_REGROWS = 45         # steps until an apple hangs on the tree again (the tree alone is barely enough)
WINDFALL_PLACES = ["forest", "river", "workshop", "hut"]
WINDFALL_REGROWS = 25      # steps until an apple lies somewhere else again
REST_DAY = 0.4              # how often the tree bears nothing for a whole day (practice only)
MAX_CARRY = 3             # at most this many of each thing
VARIANTS = ["apple", "branch", "stone"]       # things that come big or small (version 17)
SIZES = ["big", "small"]
MODS = ["any", "big", "small"]                # which one the hands choose
APPLE_FILLS_BY_SIZE = {"big": 0.5, "small": 0.3}

SENSES = len(PLACES) + len(PLACES) + len(THINGS) + len(THINGS) + 2 + 3     # place, station, ground, backpack, hunger, outcome
SENSES += 4 * len(VARIANTS) + 1                 # version 17: big/small on the ground and in the backpack, big apple on the tree


class Environment:
    """What doesn't belong to a body: ground, apple tree, windfall. Two creatures can share one."""

    def __init__(self, z):
        self.ground = {o: {d: 0 for d in THINGS} for o in PLACES}
        self.ground_big = {o: {d: 0 for d in VARIANTS} for o in PLACES}   # how many of them are big
        self.tree_big = z.random() < 0.5    # the size of the apple hanging on the tree
        self.windfall_big = False
        self.apple_ready = 0          # steps until the next apple
        self.windfall = None           # where a windfall apple lies right now
        self.windfall_timer = z.randint(0, 10)


def _env_field(name):
    return property(lambda self: getattr(self.u, name), lambda self, v: setattr(self.u, name, v))


class World:
    """A world with one creature (or one body in a shared world, see share_with).
    step() gets (verb, object) as numbers and returns what happened."""

    ground = _env_field("ground")
    ground_big = _env_field("ground_big")
    tree_big = _env_field("tree_big")
    windfall_big = _env_field("windfall_big")
    apple_ready = _env_field("apple_ready")
    windfall = _env_field("windfall")
    windfall_timer = _env_field("windfall_timer")

    def __init__(self, rng=None, rest_days=False):
        self.z = rng or random.Random()
        self.rest_days = rest_days
        self.fresh()

    def share_with(self, other):
        """From now on this body lives in the world of `other` (time there runs through `other`)."""
        self.u = other.u
        self.ticks = False

    def fresh(self):
        self.u = Environment(self.z)
        if self.rest_days and self.z.random() < REST_DAY:
            self.u.apple_ready = 10 ** 9      # no apple on the tree today
        self.ticks = True              # only one body makes shared time run (tree, windfall)
        self.place = "hut"
        self.backpack = {d: 0 for d in THINGS}
        self.backpack_big = {d: 0 for d in VARIANTS}
        self.hunger = self.z.uniform(0.1, 0.5)
        self.outcome = "nothing"       # ok / failure / nothing
        self.had_choice = False
        self.steps = 0
        self.eaten = 0
        self.built = 0

    # ------------------------------------------------------------ senses
    def senses(self):
        v = [0.0] * SENSES
        i = 0
        v[i + PLACES.index(self.place)] = 1
        i += len(PLACES)
        v[i + PLACES.index(self.place)] = 1 if self.place in STATIONS else 0       # the station of this place
        i += len(PLACES)
        for k, d in enumerate(THINGS):
            v[i + k] = min(self.ground[self.place][d], 3) / 3
        i += len(THINGS)
        for k, d in enumerate(THINGS):
            v[i + k] = min(self.backpack[d], 3) / 3
        i += len(THINGS)
        v[i] = self.hunger
        v[i + 1] = 1.0 if self.place == "orchard" and self.apple_ready == 0 else 0.0   # sees an apple hanging
        i += 2
        v[i + ["ok", "failure", "nothing"].index(self.outcome)] = 1
        i += 3
        for d in VARIANTS:                                       # version 17: sizes
            big, all_ = self.ground_big[self.place][d], self.ground[self.place][d]
            v[i], v[i + 1] = min(big, 3) / 3, min(all_ - big, 3) / 3
            big, all_ = self.backpack_big[d], self.backpack[d]
            v[i + 2], v[i + 3] = min(big, 3) / 3, min(all_ - big, 3) / 3
            i += 4
        v[i] = 1.0 if self.place == "orchard" and self.apple_ready == 0 and self.tree_big else 0.0
        return v

    # ------------------------------------------------------------ sizes (version 17)
    def _remove(self, total, big, thing, mod):
        """Takes one thing out of a store (ground of a place or backpack). mod: any/big/small. Returns its size or
        None if there is none of that kind."""
        n = total[thing]
        if n <= 0:
            return None
        self.had_choice = False
        if thing not in VARIANTS:
            total[thing] -= 1
            return ""
        nb = big[thing]
        self.had_choice = nb > 0 and n - nb > 0          # were both sizes there?
        if mod == "any":
            size = "big" if self.z.random() < nb / n else "small"
        else:
            size = mod
        if size == "big" and nb == 0:                 # wanted a big one, there is only a small one: takes that
            size = "small"
        elif size == "small" and n - nb == 0:
            size = "big"
        total[thing] -= 1
        if size == "big":
            big[thing] -= 1
        return size

    def _add(self, total, big, thing, size):
        total[thing] += 1
        if thing in VARIANTS and size == "big":
            big[thing] += 1

    def random_size(self):
        return self.z.choice(SIZES)

    def shuffle_sizes(self):
        """For tests that put things somewhere by hand: give every thing a random size."""
        for o in PLACES:
            for d in VARIANTS:
                self.ground_big[o][d] = sum(self.z.random() < 0.5 for _ in range(self.ground[o][d]))
        for d in VARIANTS:
            self.backpack_big[d] = sum(self.z.random() < 0.5 for _ in range(self.backpack[d]))

    def place_thing(self, place, thing, size=None):
        """Put a thing on the ground (for tests and the teacher)."""
        self._add(self.ground[place], self.ground_big[place], thing, size or self.random_size())

    def drop_windfall(self, place):
        """An apple falls (blown by the wind, dropped by a bird) – this is where it lies now."""
        if self.windfall:
            g = self.ground[self.windfall]
            if g["apple"] > 0:
                gb = self.ground_big[self.windfall]
                if self.windfall_big and gb["apple"] > 0 or g["apple"] - gb["apple"] == 0:
                    gb["apple"] -= 1
                g["apple"] -= 1
        self.windfall = place
        self.windfall_big = self.z.random() < 0.5
        self._add(self.ground[place], self.ground_big[place], "apple", "big" if self.windfall_big else "small")

    def give(self, thing):
        """Someone gives the creature something (the teacher)."""
        if self.backpack[thing] < MAX_CARRY:
            self._add(self.backpack, self.backpack_big, thing, self.random_size())
            return True
        return False

    # ------------------------------------------------------------ acting
    def step(self, verb_i, obj_i, mod_i=0):
        """Returns (reward from the body, event). The event describes what really happened.
        mod_i (version 17): which one the hands choose – any, a big one, a small one."""
        verb, obj, mod = VERBS[verb_i], OBJECTS[obj_i], MODS[mod_i]
        self.steps += 1
        hunger_before = self.hunger
        self.hunger = min(1.0, self.hunger + HUNGER_PER_STEP)
        if self.ticks and self.apple_ready > 0:
            self.apple_ready -= 1
        if self.ticks and self.windfall is None:
            self.windfall_timer -= 1
            if self.windfall_timer <= 0:
                self.drop_windfall(self.z.choice([o for o in WINDFALL_PLACES if o != self.place]))
        ev = {"verb": verb, "obj": obj, "ok": False, "place": self.place}
        if verb == "go" and obj in DIRECTIONS:
            target = PATHS[self.place].get(obj)
            if target:
                self.place = target
                ev.update(ok=True, target=target)
        elif verb == "take" and obj in THINGS:
            if self.backpack[obj] < MAX_CARRY:
                size = self._remove(self.ground[self.place], self.ground_big[self.place], obj, mod)
                if size is not None:
                    self._add(self.backpack, self.backpack_big, obj, size)
                    ev.update(ok=True, size=size, choice=self.had_choice)
                    if obj == "apple" and self.place == self.windfall:
                        self.windfall, self.windfall_timer = None, WINDFALL_REGROWS
                        ev["windfall"] = True
                elif obj in SOURCES.get(self.place, []) and (obj != "apple" or self.apple_ready == 0):
                    if obj == "apple":
                        size = "big" if self.tree_big else "small"
                    else:
                        size = self.random_size() if mod == "any" else mod      # branches and stones: plenty of both
                    self._add(self.backpack, self.backpack_big, obj, size)
                    ev.update(ok=True, size=size, choice=obj != "apple")     # branches, stones: both there
                    if obj == "apple":
                        self.apple_ready = APPLE_REGROWS
                        self.tree_big = self.z.random() < 0.5
        elif verb == "eat" and obj in THINGS:
            size = self._remove(self.backpack, self.backpack_big, "apple", mod) if obj == "apple" else None
            if size is not None:
                ev.update(size=size, choice=self.had_choice)
                self.hunger = max(0.0, self.hunger - APPLE_FILLS_BY_SIZE[size])
                self.eaten += 1
                ev["ok"] = True
        elif verb == "put" and obj in THINGS:
            size = self._remove(self.backpack, self.backpack_big, obj, mod)
            if size is not None:
                self._add(self.ground[self.place], self.ground_big[self.place], obj, size)
                ev.update(ok=True, size=size, choice=self.had_choice)
        elif verb == "build" and obj in RECIPE:
            needs = RECIPE[obj]
            if self.place == "workshop" and all(self.backpack[d] >= n for d, n in needs.items()) \
                    and self.backpack[obj] < MAX_CARRY:
                for d, n in needs.items():
                    for _ in range(n):
                        self._remove(self.backpack, self.backpack_big, d, "any")
                self.backpack[obj] += 1
                self.built += 1
                ev["ok"] = True
        elif verb in ("wait", "say"):
            ev["ok"] = True
        self.outcome = "nothing" if verb in ("wait", "say") else ("ok" if ev["ok"] else "failure")
        # what the body reports: hunger hurts, getting full feels good
        body = -0.02 * self.hunger ** 2 + max(0.0, hunger_before - self.hunger) * 1.5
        ev["place_after"] = self.place
        return body, ev
