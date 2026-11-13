"""A small but real text world – fast enough for a neural brain to practise in.

Five places, real things, real rules:
                 Forest (tree: branches, bast)
                    |
 Orchard ——— Hut ——— Workshop (workbench)
 (apple tree)        |
                 River (bank: stones)

Hunger grows over time; an apple (about 80 kcal) eases it a little. From a branch, a stone and some bast you build a
stone axe at the workbench (as people did in the Stone Age). An apple tree doesn't bear without limit: after picking,
a new apple grows back slowly.

Version 6 – hidden knowledge: the tree alone is no longer enough. Now and then a windfall apple lies somewhere
(in the forest, at the river, in the workshop or in the hut) – each time somewhere else. The agent only sees it when
it is there. The teacher knows where it lies: whoever listens (or asks) finds food faster.

Version 10 – a friend: two agents (NikolAI and Kogli) can share one world. Each has its own body (place, hunger,
backpack); the ground, the apple tree and the windfall apple belong to both. What one puts down, the other can take;
whoever finds the windfall apple first gets it.

Version 11 – rest days: on some days the apple tree bears nothing at all (as in a bad year). Then there are only
windfall apples – and whoever listens, asks, or tells the friend gets full. Practice only; the tests run without.

The agent perceives the world through its senses (a vector of numbers: where am I, what do I see, what do I carry,
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
VERBS = ["go", "take", "eat", "put", "build", "wait", "say"]     # "say": saying words instead of acting
OBJECTS = DIRECTIONS + THINGS
HUNGER_PER_STEP = 1 / 120
APPLE_FILLS = 0.4
APPLE_REGROWS = 45         # steps until an apple hangs on the tree again (the tree alone is barely enough – more apples only elsewhere)
WINDFALL_PLACES = ["forest", "river", "workshop", "hut"]
WINDFALL_REGROWS = 25      # steps until an apple lies somewhere else again
REST_DAY = 0.4              # how often the tree bears nothing for a whole day (practice only)
MAX_CARRY = 3             # at most this many of each thing

SENSES = len(PLACES) + len(PLACES) + len(THINGS) + len(THINGS) + 2 + 3     # place, station, ground, backpack, hunger, outcome


class Environment:
    """What doesn't belong to a body: ground, apple tree, windfall apple. Two agents can share one."""

    def __init__(self, z):
        self.ground = {o: {d: 0 for d in THINGS} for o in PLACES}
        self.apple_ready = 0          # steps until the next apple
        self.windfall = None           # where a windfall apple lies right now
        self.windfall_timer = z.randint(0, 10)


def _env_field(name):
    return property(lambda self: getattr(self.u, name), lambda self, v: setattr(self.u, name, v))


class World:
    """A world with one agent (or one body in a shared world, see share_with).
    step() takes (verb, object) as numbers and returns the result."""

    ground = _env_field("ground")
    apple_ready = _env_field("apple_ready")
    windfall = _env_field("windfall")
    windfall_timer = _env_field("windfall_timer")

    def __init__(self, rng=None, rest_days=False):
        self.z = rng or random.Random()
        self.rest_days = rest_days
        self.reset()

    def share_with(self, other):
        """From now on this body lives in the world of `other` (time there is driven by `other`)."""
        self.u = other.u
        self.ticks = False

    def reset(self):
        self.u = Environment(self.z)
        if self.rest_days and self.z.random() < REST_DAY:
            self.u.apple_ready = 10 ** 9      # no apple on the tree today
        self.ticks = True              # only one body drives the shared time (tree, windfall)
        self.place = "hut"
        self.backpack = {d: 0 for d in THINGS}
        self.hunger = self.z.uniform(0.1, 0.5)
        self.outcome = "nothing"       # ok / failed / nothing
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
        v[i + ["ok", "failed", "nothing"].index(self.outcome)] = 1
        return v

    def place_windfall(self, place):
        """An apple falls (blown by the wind, dropped by a bird) – this is where it lies now."""
        if self.windfall:
            self.ground[self.windfall]["apple"] = max(0, self.ground[self.windfall]["apple"] - 1)
        self.windfall = place
        self.ground[place]["apple"] += 1

    def give(self, thing):
        """Someone gives the agent something (the teacher)."""
        if self.backpack[thing] < MAX_CARRY:
            self.backpack[thing] += 1
            return True
        return False

    # ------------------------------------------------------------ acting
    def step(self, verb_i, obj_i):
        """Returns (reward from the body, event). The event describes what really happened."""
        verb, obj = VERBS[verb_i], OBJECTS[obj_i]
        self.steps += 1
        hunger_before = self.hunger
        self.hunger = min(1.0, self.hunger + HUNGER_PER_STEP)
        if self.ticks and self.apple_ready > 0:
            self.apple_ready -= 1
        if self.ticks and self.windfall is None:
            self.windfall_timer -= 1
            if self.windfall_timer <= 0:
                self.place_windfall(self.z.choice([o for o in WINDFALL_PLACES if o != self.place]))
        ev = {"verb": verb, "obj": obj, "ok": False, "place": self.place}
        if verb == "go" and obj in DIRECTIONS:
            goal = PATHS[self.place].get(obj)
            if goal:
                self.place = goal
                ev.update(ok=True, goal=goal)
        elif verb == "take" and obj in THINGS:
            if self.backpack[obj] < MAX_CARRY:
                if self.ground[self.place][obj] > 0:
                    self.ground[self.place][obj] -= 1
                    self.backpack[obj] += 1
                    ev["ok"] = True
                    if obj == "apple" and self.place == self.windfall:
                        self.windfall, self.windfall_timer = None, WINDFALL_REGROWS
                        ev["windfall"] = True
                elif obj in SOURCES.get(self.place, []) and (obj != "apple" or self.apple_ready == 0):
                    self.backpack[obj] += 1
                    ev["ok"] = True
                    if obj == "apple":
                        self.apple_ready = APPLE_REGROWS
        elif verb == "eat" and obj in THINGS:
            if obj == "apple" and self.backpack["apple"] > 0:
                self.backpack["apple"] -= 1
                self.hunger = max(0.0, self.hunger - APPLE_FILLS)
                self.eaten += 1
                ev["ok"] = True
        elif verb == "put" and obj in THINGS:
            if self.backpack[obj] > 0:
                self.backpack[obj] -= 1
                self.ground[self.place][obj] += 1
                ev["ok"] = True
        elif verb == "build" and obj in RECIPE:
            needs = RECIPE[obj]
            if self.place == "workshop" and all(self.backpack[d] >= n for d, n in needs.items()) \
                    and self.backpack[obj] < MAX_CARRY:
                for d, n in needs.items():
                    self.backpack[d] -= n
                self.backpack[obj] += 1
                self.built += 1
                ev["ok"] = True
        elif verb in ("wait", "say"):
            ev["ok"] = True
        self.outcome = "nothing" if verb in ("wait", "say") else ("ok" if ev["ok"] else "failed")
        # what the body reports: hunger hurts, getting full feels good
        body = -0.02 * self.hunger ** 2 + max(0.0, hunger_before - self.hunger) * 1.5
        ev["place_after"] = self.place
        return body, ev
