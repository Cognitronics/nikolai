"""Imagination: what NikolAI expects a thing to be like – and how sure it is.

Every thing in the world has real, perceivable properties (colour, size, shape, surface, parts, weight, taste). Apples
vary (red, green, yellow), stones too (big, small, round, angular). NikolAI only finds them out by doing things:

* seeing  – it is where the thing is (on the ground, on the tree, at its source): colour, size, rough shape
* taking  – shape, surface, weight, hardness, stem, leaf
* carrying – while it carries something it looks at it: fine details (spots, stem, leaf, parts like a handle)
* eating  – taste (only apples can be eaten)

A part of its brain learns from the HEARD WORD ("apple") to predict these properties – each with an uncertainty
(a Gaussian: mean and spread; trained with its negative log-likelihood). Where it has experience, the spread shrinks
to how much the thing really varies; where it has none, it stays wide: fog. Words that name nothing it ever touched
("dragon", "house") are taught to stay fog. This is the idea of the brain as a prediction machine (predictive
coding; e.g. Rao & Ballard 1999, Friston 2010) – and the picture on the dashboard is drawn from samples of these
expectations: certain = sharp, uncertain = blurred, varying = flickering between possibilities.

Version 17: apples, branches and stones are big or small. A big one is bigger and heavier. When it hears "big apple"
it should expect a bigger one than for "small apple" – and for "big stone", which it never heard, too (Phase 10b).

The paint box (circle, line, colour, blur) is given, like crayons for a child. What it paints, and how exactly,
comes only from its experience.
"""

import random

import torch

# The properties (all between 0 and 1)
PROPERTIES = ["red", "green", "blue", "size", "long", "rough", "weight", "hard", "stem", "leaf", "spots",
                 "sweet", "sour", "handle", "blade", "binding"]
P = len(PROPERTIES)
IDX = {n: i for i, n in enumerate(PROPERTIES)}
VISIBLE = ["red", "green", "blue", "size", "long", "rough", "stem", "leaf", "spots", "handle", "blade", "binding"]

# Which sense reveals what
SENSE = {
    "seeing": ["red", "green", "blue", "size", "long"],
    "taking": ["size", "long", "rough", "weight", "hard", "stem", "leaf"],
    "carrying": ["rough", "stem", "leaf", "spots", "handle", "blade", "binding"],
    "eating": ["sweet", "sour"],
}

SIZED_THINGS = ("apple", "branch", "stone")
SIZE_OF = {("apple", "big"): (0.38, 0.03), ("apple", "small"): (0.22, 0.03), ("branch", "big"): (0.78, 0.06),
           ("branch", "small"): (0.42, 0.06), ("stone", "big"): (0.52, 0.05), ("stone", "small"): (0.2, 0.05)}
THING_WORD = {"apple": "apple", "branch": "branch", "bast": "bast", "stone": "stone", "stone_axe": "stone axe"}


def _n(z, m, s):
    return min(1.0, max(0.0, z.gauss(m, s)))


def sample_thing(thing, z, size=None):
    """One real thing of this kind, with all its properties (a fresh one each time – things vary).
    size (version 17): big or small for apples, branches and stones (random if not given)."""
    x = dict.fromkeys(PROPERTIES, 0.0)
    if thing in SIZED_THINGS and not size:
        size = z.choice(["big", "small"])
    if thing == "apple":
        colour = z.choices([(0.85, 0.15, 0.12), (0.45, 0.8, 0.25), (0.95, 0.82, 0.2)], [0.5, 0.3, 0.2])[0]
        x.update(red=_n(z, colour[0], 0.05), green=_n(z, colour[1], 0.05), blue=_n(z, colour[2], 0.04),
                 size=_n(z, 0.3, 0.05), long=_n(z, 0.1, 0.05), rough=_n(z, 0.1, 0.05), weight=_n(z, 0.2, 0.05),
                 hard=_n(z, 0.3, 0.05), stem=float(z.random() < 0.9), leaf=float(z.random() < 0.4),
                 spots=_n(z, 0.3, 0.15), sweet=_n(z, 0.7, 0.15), sour=_n(z, 0.3, 0.15))
    elif thing == "branch":
        x.update(red=_n(z, 0.45, 0.05), green=_n(z, 0.3, 0.05), blue=_n(z, 0.15, 0.04), size=_n(z, 0.6, 0.15),
                 long=_n(z, 0.9, 0.05), rough=_n(z, 0.7, 0.1), weight=_n(z, 0.25, 0.08), hard=_n(z, 0.6, 0.1),
                 leaf=float(z.random() < 0.3), spots=_n(z, 0.1, 0.05))
    elif thing == "stone":
        g = _n(z, 0.55, 0.1)
        x.update(red=g, green=g, blue=_n(z, g + 0.03, 0.02), size=_n(z, 0.35, 0.15), long=_n(z, 0.3, 0.2),
                 rough=_n(z, 0.5, 0.2), weight=_n(z, 0.8, 0.1), hard=1.0, spots=_n(z, 0.5, 0.25))
    elif thing == "bast":
        x.update(red=_n(z, 0.8, 0.05), green=_n(z, 0.7, 0.05), blue=_n(z, 0.45, 0.05), size=_n(z, 0.5, 0.1),
                 long=_n(z, 0.95, 0.03), rough=_n(z, 0.6, 0.1), weight=_n(z, 0.05, 0.03), hard=_n(z, 0.1, 0.05),
                 spots=_n(z, 0.2, 0.1))
    elif thing == "stone_axe":
        x.update(red=_n(z, 0.5, 0.05), green=_n(z, 0.45, 0.05), blue=_n(z, 0.4, 0.05), size=_n(z, 0.6, 0.08),
                 long=_n(z, 0.7, 0.05), rough=_n(z, 0.4, 0.1), weight=_n(z, 0.7, 0.08), hard=_n(z, 0.9, 0.05),
                 handle=1.0, blade=1.0, binding=1.0, spots=_n(z, 0.3, 0.1))
    if thing in SIZED_THINGS:                          # version 17: big or small – bigger is heavier
        m, s = SIZE_OF[(thing, size)]
        x["size"] = _n(z, m, s)
        x["weight"] = _n(z, x["weight"] * (1.25 if size == "big" else 0.75), 0.03)
    return x


def perceive(thing, sense, z, size=None):
    """(properties as a tensor, mask: which ones this sense reveals). Seeing is a little blurry."""
    x = sample_thing(thing, z, size)
    noise = 0.04 if sense == "seeing" else 0.0
    vals = torch.tensor([min(1.0, max(0.0, x[n] + z.gauss(0, noise))) for n in PROPERTIES])
    mask = torch.zeros(P)
    for n in SENSE[sense]:
        mask[IDX[n]] = 1.0
    return vals, mask


def true_distribution(thing, n=2000, seed=0, size=None):
    """What this kind of thing is really like on average, and how much it varies (for the tests)."""
    z = random.Random(seed)
    xs = torch.tensor([[sample_thing(thing, z, size)[e] for e in PROPERTIES] for _ in range(n)])
    return xs.mean(0), xs.std(0)


def size_in(total, big, thing, z):
    """The size of a random one of these things in a store (ground or backpack), or None."""
    if thing not in SIZED_THINGS or total[thing] <= 0:
        return None
    return "big" if z.random() < big[thing] / total[thing] else "small"


def visible_things(world):
    """Which things NikolAI can see right now (on the ground, on the tree, at their source): [(thing, size)]."""
    from .world import SOURCES
    z = world.z
    here = world.place
    visible_now = [(d, size_in(world.ground[here], world.ground_big[here], d, z))
                   for d, n in world.ground[here].items() if n > 0]
    for d in SOURCES.get(here, []):
        if d == "apple" and world.apple_ready == 0:
            visible_now.append((d, "big" if world.tree_big else "small"))
        elif d != "apple":
            visible_now.append((d, z.choice(["big", "small"])))
    return visible_now


UNKNOWN = ["dragon", "house", "sun", "moon", "ship", "castle", "flower", "bird", "fish", "cat", "dog", "horse",
             "car", "door", "king", "queen", "cloud", "star", "boat", "bread"]       # named, never touched
