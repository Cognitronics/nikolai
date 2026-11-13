"""The teacher: talks to NikolAI about what it is really doing right now – like a parent with a toddler.

* Describes actions ("You take an apple.") and names things ("This is the apple tree.").
* Gives instructions ("Go to the forest!") and praises NikolAI when it follows them.
* Some combinations are never given as instructions during training (e.g. "Put down the apple!"). They only appear
  in tests: that shows whether NikolAI really understood the words – or only memorised whole sentences.
* Listens: if NikolAI says the word for something it really needs right now ("apple" when hungry, "bast" at the
  workbench when bast is the only thing missing for a stone axe), the teacher hands it over and models the full
  sentence in first person ("Say: I want an apple.").
* Does not give everything: apples only now and then (NikolAI still has to pick them itself), a stone axe never
  (it has to build one), branch, stone and bast only for an instruction, and nothing it doesn't need. Receiving
  something is never rewarded in itself – only what NikolAI achieves with it counts (otherwise it learns to trick
  the teacher, as happened in version 3).

Version 6:
* Knows where the windfall apple lies – NikolAI doesn't. When NikolAI is hungry, the teacher sometimes says where it
  is ("There is an apple at the river.") and sometimes models the question ("Hungry? Say: where is an apple?").
  If NikolAI asks "where … apple", the teacher answers. Listening (or asking) now really pays off.
* Points at the windfall apple when NikolAI arrives where it lies ("Look, there is an apple here. Take an apple!"):
  at first NikolAI only knows apples from the tree. Hindsight then links this to "There is an apple in the forest."
* Gives instructions against habit: what NikolAI does on its own anyway is asked for less often, and nothing about
  apples while it is hungry anyway. Otherwise it can earn praise without listening.
* hindsight(): after the fact, what instruction would this action have fulfilled? ("You took a stone" → "Take a
  stone!"). Practising uses this to learn from every action, not only from the rare real instructions
  (hindsight relabeling: Andrychowicz et al. 2017; for language: Chan et al. 2019).

Version 8 – conversation: now and then the teacher asks NikolAI about itself ("Where are you?", "How are you?",
"What do you have?", "What do you want?", "What did you do?", "Who are you?"). antwort() is the true answer in its
world right now – NikolAI learns to give it (with the speaking part it got from reading).

Version 10 – a friend: NikolAI and Kogli share a world and hear what the other says. When the friend is hungry
and this one stands right where an apple lies, the teacher whispers: "Kogli is hungry. Say: there is an apple in the
forest." – the same sentence the teacher uses as a hint, which the friend already understands. Whether passing it on
is worth it, they find out themselves: each also feels it when the other gets full (a shared reward).

Everything the teacher says is correct English and true in the world. It knows nothing about the brain's insides.
"""

import re

from .welt import REZEPT

ORT_ZIEL = {"huette": "to the hut", "obstgarten": "to the orchard", "wald": "to the forest", "fluss": "to the river",
            "werkstatt": "to the workshop"}
A = {"apfel": "an apple", "ast": "a branch", "bast": "some bast", "stein": "a stone", "steinbeil": "a stone axe"}
THE = {"apfel": "the apple", "ast": "the branch", "bast": "the bast", "stein": "the stone", "steinbeil": "the stone axe"}
NO = {"apfel": "no apple", "ast": "no branch", "bast": "no bast", "stein": "no stone", "steinbeil": "no stone axe"}
STATION_NAME = {"obstgarten": "This is the apple tree.", "wald": "This is a tree.", "fluss": "This is the riverbank.",
                "werkstatt": "This is the workbench.", "huette": "This is your bed."}
WORD = {"apfel": "apple", "ast": "branch", "bast": "bast", "stein": "stone", "steinbeil": "stone axe"}
WORD_TO_THING = {"apple": "apfel", "branch": "ast", "bast": "bast", "stone": "stein", "axe": "steinbeil"}
RICHTUNGEN = ("norden", "sueden", "osten", "westen")
ORT_BEI = {"huette": "in the hut", "obstgarten": "in the orchard", "wald": "in the forest", "fluss": "at the river",
           "werkstatt": "in the workshop"}
FRAGEN = {"wie": "How are you?", "wo": "Where are you?", "hast": "What do you have?", "willst": "What do you want?",
          "tat": "What did you do?", "wer": "Who are you?"}
FRAGE_RATE = 0.10         # chance per step to ask something when no instruction is running
WORT_ORT = {"huette": "hut", "obstgarten": "orchard", "wald": "forest", "fluss": "river", "werkstatt": "workshop"}
TIPP_PAUSE = 20           # steps between two hints / modelled questions

# Instructions: (verb, object) → sentence. For "gehe" the object is the destination; for "sage" the words to say.
AUFTRAEGE = {("nimm", d): f"Take {A[d]}!" for d in ("apfel", "ast", "bast", "stein")}
AUFTRAEGE.update({("gehe", o): f"Go {ORT_ZIEL[o]}!" for o in ORT_ZIEL})
AUFTRAEGE.update({("lege", d): f"Put down {THE[d]}!" for d in ("apfel", "ast", "bast", "stein")})
AUFTRAEGE[("iss", "apfel")] = "Eat an apple!"
AUFTRAEGE[("baue", "steinbeil")] = "Build a stone axe!"
AUFTRAEGE.update({("sage", WORD[d]): f"Say “{WORD[d]}”!" for d in WORD})                       # repeating
AUFTRAEGE.update({("sage", f"i want {A[d]}"): f"Say “I want {A[d]}”!" for d in ("apfel", "stein", "bast")})
APFEL_PAUSE = 150         # steps between two apples from the teacher (rare: begging alone must not feed it)
GEBEN_PAUSE = 8
NIE_GEUEBT = {("lege", "apfel"), ("nimm", "bast")}      # test only – NikolAI knows the words from other sentences
FRIST = 30                                              # steps NikolAI has to follow an instruction


def worte(satz):
    return re.findall(r"[a-z]+", (satz or "").lower())


def vorgesagt(satz):
    """What the teacher asks NikolAI to repeat ("Say: I want an apple." / "Say “stone”!") – or None."""
    m = re.search(r"Say:? “?([^“”!.]+)", satz or "")
    return worte(m.group(1)) if m else None


def beschreiben(ev, welt):
    """What the teacher says about an action (or None)."""
    v, o, ok = ev["verb"], ev["obj"], ev["ok"]
    if v in ("warte", "sage"):
        return None
    if v == "gehe":
        if o not in RICHTUNGEN:
            return "You can walk north, south, east or west."
        return f"You go {ORT_ZIEL[ev['ziel']]}." if ok else "There is no path that way."
    if o in RICHTUNGEN:
        return "That doesn't work."
    if v == "nimm":
        if ok:
            return f"You take {A[o]}."
        return f"There is {NO[o]} here." if welt.rucksack[o] < 3 else "You can't carry any more."
    if v == "iss":
        if o != "apfel":
            return "You can't eat that."
        return "You eat an apple. Yummy!" if ok else "You have no apple."
    if v == "lege":
        return f"You put down {THE[o]}." if ok else f"You have {NO[o]}."
    if v == "baue":
        if o not in REZEPT:
            return "That can't be built."
        if ok:
            return "You build a stone axe!"
        if ev["ort"] != "werkstatt":
            return "You can build things at the workbench."
        return "For that you need a branch, a stone and some bast."
    return None


def antwort(typ, welt, ev=None, name="nikolai"):
    """The true answer to a question about itself, right now, in first person."""
    r = welt.rucksack
    if typ == "wie":
        return "i am hungry" if welt.hunger > 0.6 else "i am full" if welt.hunger < 0.25 else "i am fine"
    if typ == "wo":
        return f"i am {ORT_BEI[welt.ort]}"
    if typ == "hast":
        dinge = [A[d] for d in ("apfel", "ast", "bast", "stein", "steinbeil") if r[d] > 0]
        return "i have " + " and ".join(dinge[:2]) if dinge else "i have nothing"
    if typ == "willst":
        if welt.hunger > 0.5 and r["apfel"] == 0:
            return "i want an apple"
        fehlt = [d for d in ("ast", "stein", "bast") if r[d] == 0]
        if r["steinbeil"] == 0 and 0 < len(fehlt) < 3:
            return f"i want {A[fehlt[0]]}"
        return "i want to play"
    if typ == "tat":
        if not ev or not ev["ok"] or ev["verb"] in ("warte", "sage"):
            return "i did nothing"
        v, o = ev["verb"], ev["obj"]
        return {"gehe": lambda: f"i went {ORT_ZIEL.get(ev.get('ziel'), '')}", "nimm": lambda: f"i took {A[o]}",
                "iss": lambda: "i ate an apple", "lege": lambda: f"i put down {THE[o]}",
                "baue": lambda: "i built a stone axe"}[v]()
    return f"i am {name}"


def frage_aus_text(text):
    """Which of the teacher's questions is this? (also when a human asks it)"""
    t = _norm(text)
    for typ, satz in FRAGEN.items():
        if t == _norm(satz):
            return typ
    return None


def wo_satz(ort):
    return f"There is an apple {ORT_BEI[ort]}."


def handlung(ev):
    """Which instruction (verb, obj) does this successful action fulfil? Or None."""
    if not ev or not ev["ok"]:
        return None
    a = ("gehe", ev["ziel"]) if ev["verb"] == "gehe" and "ziel" in ev else (ev["verb"], ev["obj"])
    return a if a in AUFTRAEGE else None


def hindsight(ev):
    """Sentences that would have been true instructions or hints for what NikolAI just did (never test-only ones)."""
    out = []
    a = handlung(ev)
    if a and a not in NIE_GEUEBT:
        out.append(AUFTRAEGE[a])
    if ev and ev.get("fallobst"):
        out.append(wo_satz(ev["ort"]))
    return out


def erfuellt(auftrag, ev):
    v, o = auftrag
    if v == "sage":
        return bool(ev.get("worte")) and all(x in ev["worte"] for x in o.split())
    if v == "gehe":
        return ev["ort_nachher"] == o and ev["verb"] == "gehe" and ev["ok"]
    return ev["verb"] == v and ev["obj"] == o and ev["ok"]


def ding_aus(worte):
    """Which thing does an utterance ask for? ("stone axe" counts as the axe, not the stone.)"""
    if "axe" in worte:
        return "steinbeil"
    return next((WORD_TO_THING[w] for w in worte if w in WORD_TO_THING), None)


class Lehrer:
    """A teacher for one world. sprich() returns what NikolAI hears this step, and the reward."""

    def __init__(self, zufall, auftraege=True, geuebt_nur=True, beschreiben_rate=0.6, name="nikolai"):
        self.z = zufall
        self.name = name
        self.freund = None                                   # (world of the friend, its name) – version 10
        self.freund_sagte = []                               # what the friend just said (NikolAI heard it too)
        self.auftraege_an = auftraege
        self.geuebt_nur = geuebt_nur
        self.rate = beschreiben_rate
        self.auftrag = None
        self.seit = 0
        self.pause = 0
        self.erfolge = 0
        self.versuche = 0
        self.seit_geben = 99
        self.seit_apfel = 99
        self.bitten = {"verstanden": 0, "ich_form": 0, "gegeben": 0, "wo": 0}
        self.seit_tipp = 99
        self.frage = None                                    # the question just asked (NikolAI answers next step)
        self.gewohnt = {}                                    # what NikolAI does on its own (no instruction)
        self._b0 = 0
        self._w0 = 0

    def neuer_auftrag(self, welt, auftrag=None):
        if auftrag is None:
            moeglich = [a for a in AUFTRAEGE if not (self.geuebt_nur and a in NIE_GEUEBT)]
            moeglich = [a for a in moeglich if not (a[0] == "gehe" and a[1] == welt.ort)]
            moeglich = [a for a in moeglich if a[0] != "lege" or welt.rucksack[a[1]] > 0] or moeglich
            if welt.hunger > 0.5:                            # it would fetch and eat apples anyway
                moeglich = [a for a in moeglich if a[1] != "apfel"] or moeglich
            gewichte = [1.0 / (1.0 + self.gewohnt.get(a, 0)) for a in moeglich]
            auftrag = self.z.choices(moeglich, gewichte)[0]
            self.gewohnt = {k: v / 2 for k, v in self.gewohnt.items()}
        self.auftrag, self.seit = auftrag, 0
        self.versuche += 1
        return AUFTRAEGE[auftrag]

    # ------------------------------------------------------------ listening
    def braucht(self, welt, ding):
        """Does NikolAI really need this thing right now? (Only then does the teacher hand it over.)"""
        r = welt.rucksack
        if ding == "apfel":
            return welt.hunger > 0.4 and r["apfel"] == 0 and self.seit_apfel >= APFEL_PAUSE
        if ding == "steinbeil":
            return False
        if r[ding] > 0:
            return False
        if self.auftrag in (("nimm", ding), ("lege", ding)):
            return True
        if self.auftrag != ("baue", "steinbeil"):         # only for a real purpose
            return False
        andere = [d for d in ("ast", "stein", "bast") if d != ding]
        return all(r[d] > 0 for d in andere)              # the only thing still missing for a stone axe

    def hoere_wesen(self, welt, worte):
        """NikolAI said something. Returns (answer, reward)."""
        if "where" in worte:                               # a question, not a request
            self.bitten["wo"] += 1
            self.seit_tipp = 0
            if welt.fallobst:
                return wo_satz(welt.fallobst), 0.0
            return "No apple is lying around. Look at the apple tree.", 0.0
        x = ding_aus(worte)
        if not x:
            return ("Hm?" if worte and self.z.random() < 0.3 else ""), 0.0
        ich = "i" in worte and "want" in worte
        self.bitten["verstanden"] += 1
        if x == "steinbeil" and welt.rucksack["steinbeil"] == 0:
            return "You have to build a stone axe yourself: from a branch, a stone and some bast.", 0.0
        if self.seit_geben < GEBEN_PAUSE or not self.braucht(welt, x):
            if welt.rucksack[x] > 0:
                return f"You already have {A[x]}.", 0.0
            return f"{WORD[x].capitalize()}? Not now.", 0.0
        welt.geben(x)
        self.seit_geben = 0
        if x == "apfel":
            self.seit_apfel = 0
        self.bitten["gegeben"] += 1
        if ich:                                            # no reward for receiving – only what it achieves counts
            self.bitten["ich_form"] += 1
            return f"Nicely said! Here is {A[x]}.", 0.0
        return f"You want {A[x]}? Here is {A[x]}. Say: I want {A[x]}.", 0.0

    def sprich(self, welt, ev):
        """(sentence heard or "", reward from the teacher, description or None, instruction done?)"""
        teile, lohn, fertig = [], 0.0, False
        self.frage = None
        self.seit_geben += 1
        self.seit_apfel += 1
        self.seit_tipp += 1
        if ev and ev.get("worte"):
            antwort, lohn = self.hoere_wesen(welt, ev["worte"])
            if antwort:
                teile.append(antwort)
        if ev and ev["verb"] == "gehe" and ev["ok"] and ev.get("ziel") == welt.fallobst and not self.auftrag \
                and welt.rucksack["apfel"] < 2:
            teile.append("Look, there is an apple here.")       # scaffolding: points at what it can't connect yet
            teile.append(self.neuer_auftrag(welt, ("nimm", "apfel")))
        if not self.auftrag:
            a = handlung(ev)
            if a:
                self.gewohnt[a] = self.gewohnt.get(a, 0) + 1
        if self.freund and self.seit_tipp >= TIPP_PAUSE and welt.fallobst == welt.ort:
            fw, fname = self.freund                          # it sees an apple – and its friend needs one
            if fw.rucksack["apfel"] == 0 and fw.ort != welt.ort:
                if "where" in self.freund_sagte:             # the friend asked – NikolAI heard that itself
                    self.seit_tipp = 0
                    teile.append(f"{fname.capitalize()} asks. Say: {wo_satz(welt.ort).lower()}")
                elif fw.hunger > 0.45:
                    self.seit_tipp = 0
                    teile.append(f"{fname.capitalize()} is hungry. Say: {wo_satz(welt.ort).lower()}")
        if welt.fallobst and welt.hunger > 0.35 and welt.rucksack["apfel"] == 0 and self.seit_tipp >= TIPP_PAUSE \
                and welt.ort != welt.fallobst:
            self.seit_tipp = 0
            direkt = 0.2 if self.freund else 0.5             # with a friend around, the friend should tell
            teile.append(wo_satz(welt.fallobst) if self.z.random() < direkt else "Hungry? Say: where is an apple?")
        beschr = beschreiben(ev, welt) if ev else None
        name = STATION_NAME.get(ev["ziel"]) if ev and ev["verb"] == "gehe" and ev["ok"] and self.z.random() < 0.3 else None
        if beschr and self.z.random() < self.rate:
            teile.append(beschr)
        else:
            beschr = None
        if name:
            teile.append(name)
        if self.auftrag:
            self.seit += 1
            if ev and erfuellt(self.auftrag, ev):
                teile.append("Well done!")
                lohn, fertig = lohn + 1.0, True
                self.erfolge += 1
                self.auftrag, self.pause = None, self.z.randint(5, 25)
            elif self.seit >= FRIST:
                self.auftrag, self.pause = None, self.z.randint(5, 25)
            elif self.seit % 10 == 0:
                teile.append(AUFTRAEGE[self.auftrag])            # say it again
        elif self.auftraege_an:
            self.pause -= 1
            if self.pause <= 0:
                teile.append(self.neuer_auftrag(welt))
            elif self.z.random() < FRAGE_RATE:            # a little conversation
                # (they share one brain – it cannot tell which body it is, so only NikolAI is asked its name)
                self.frage = self.z.choice([f for f in FRAGEN if f != "wer" or self.name == "nikolai"])
                teile.append(FRAGEN[self.frage])
        return " ".join(teile), lohn, beschr, fertig


def alle_saetze():
    """Everything the teacher can say (for the reading vocabulary: its words must all be known)."""
    out = list(AUFTRAEGE.values()) + list(STATION_NAME.values()) + [wo_satz(o) for o in ORT_BEI]
    out += ["You can walk north, south, east or west.", "There is no path that way.", "That doesn't work.",
            "You can't carry any more.", "You can't eat that.", "You eat an apple. Yummy!", "You have no apple.",
            "That can't be built.", "You build a stone axe!", "You can build things at the workbench.",
            "For that you need a branch, a stone and some bast.", "Well done!", "Hm?", "Nicely said!",
            "Hungry? Say: where is an apple?", "Look, there is an apple here.",
            "No apple is lying around. Look at the apple tree.",
            "You have to build a stone axe yourself: from a branch, a stone and some bast."]
    out += list(FRAGEN.values()) + ["i am nikolai", "i am kogli", "i did nothing", "i want to play", "i have nothing",
                                     "Nikolai is hungry.", "Kogli is hungry.", "Nikolai asks.", "Kogli asks."]
    for d in A:
        out += [f"You take {A[d]}.", f"There is {NO[d]} here.", f"You put down {THE[d]}.", f"You have {NO[d]}.",
                f"You already have {A[d]}.", f"{WORD[d].capitalize()}? Not now.", f"Here is {A[d]}.",
                f"You want {A[d]}? Here is {A[d]}. Say: I want {A[d]}."]
    out += [f"You go {z}." for z in ORT_ZIEL.values()]
    return out


def _norm(t):
    return " ".join(worte(t))


def auftrag_aus_text(text):
    """Is this one of the teacher's instructions? Then the human becomes the teacher. Returns (verb, obj) or None."""
    t = _norm(text)
    for a, satz in AUFTRAEGE.items():
        if t == _norm(satz):
            return a
    return None
