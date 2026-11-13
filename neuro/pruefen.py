"""Tests: has the brain really learned something? Every test compares against a baseline, so nothing is flattered.

Phase 1 – Feeds itself: 600 steps without a teacher. Average hunger, apples eaten. Baseline: acting at random.
Phase 2 – Understands instructions: every instruction is tested with and without the spoken sentence (same start),
          15 steps, 30 trials. "Understood" means: with the sentence it works in at least 70 % and at least 30 points
          more often than without. What it does anyway (e.g. fetching apples) is "not testable". Two instructions
          are never given in training ("Put down the apple!", "Take some bast!") – only their words in other sentences.
Phase 3 – Describes what happened: after each action NikolAI says what happened. Word for word like the teacher?
Phase 4 – Asks for things: NikolAI stands in the workshop (far from the apple tree). Does it say "apple" on its own
          when hungry – and not when full? With the instruction "Build a stone axe!": does it say "bast" when only bast
          is missing – and not when nothing is? 10 steps, baseline: acting and talking at random.
Phase 4c – First person: how often does it ask with "I want …" instead of just the word?
Phase 4d – Repeats on request ("Say …!").
Phase 5 – Uses what it is told: hungry in the hut, the apple tree is empty, a single apple lies somewhere (forest,
          river or workshop). With "There is an apple at the river." – does it go there and take it within 4 steps?
          (Two steps are enough if it understood; guessing finds it about one time in three.) Without the sentence
          and at random for comparison.
Phase 5b – Asks where: hungry, no hint – does it ask "where … apple?" on its own (and less often when full)?
Phase 6 – Reads: on text it never read, how well does it guess the next word? (perplexity – lower is better;
          baseline: only knowing how frequent each word is.) Milestone: less than a third of the baseline.
Phase 6b – Understands words it only knows from books: instructions with words the teacher never uses
          ("Grab a stone!" instead of "Take a stone!"), tested like Phase 2 (Hill et al. 2020), but from a start that
          doesn't give the answer away (several things to choose from, or the hut with four ways) and with only
          4 steps (one or two are enough if it understood).
          Control: the same sentence with a made-up word ("Blick a stone!"). Maybe it only reacts to "stone" –
          then the made-up word works just as well. It only counts if the real word works clearly better.
Phase 8 – Tells its friend where food is: NikolAI stands where an apple lies; Kogli is hungry in the hut, the tree is
          empty, no teacher. Does Kogli get the apple within 10 steps? Once with them hearing each other, once with
          the line cut (they can't hear each other). If talking helps clearly, they really communicate. We also count
          how often NikolAI names the right place.
Phase 7 – Answers questions about itself: a random situation (place, hunger, what it carries, what it just did),
          then the teacher asks ("Where are you?" …). Is its answer exactly the true one? Baseline: always giving the
          most frequent answer. A question type counts when it is right in at least 70 % and 30 points above that
          baseline; the milestone needs four of the five types (not counting "Who are you?").
"""

import random

import torch

from . import lehrer as L
from .gehirn import ENDE
from .ueben import SAGE, satz_tensor, worte_aus
from .welt import OBJEKTE, VERBEN, WEGE, Welt

WO_FRIST = 4
WO_ORTE = ("wald", "fluss", "werkstatt")


def _start(welt, auftrag):
    if auftrag[0] == "lege":
        welt.rucksack[auftrag[1]] = 1
    if auftrag == ("iss", "apfel"):
        welt.hunger = 0.6


def _start_fair(welt, auftrag):
    """Phase 6b: a start that doesn't give the answer away. Several things lie around (take the right one), it
    carries several things (put down the right one), or it stands in the hut with four ways and isn't hungry."""
    v, d = auftrag
    welt.hunger = 0.1
    if v == "nimm":
        welt.ort = "werkstatt"                       # nothing grows here – only what lies on the floor
        for x in ("apfel", "ast", "stein", "bast"):
            welt.boden["werkstatt"][x] = 1
    elif v == "lege":
        for x in ("apfel", "ast", "stein"):
            welt.rucksack[x] = 1
    elif v == "gehe":
        welt.ort = "huette"


FAIR_FRIST = 4           # Phase 6b: one or two steps are enough if it understood
TEST_FRIST = 15          # shorter than in training: otherwise it does many things anyway ("not testable")


def _wesen_schritt(g, ws, w, gehoert, letzte, h, z, zufall):
    """Ein Schritt im Test: Handlung und (nebenbei) Gesagtes. Gibt (v, o, worte, letzte, h) zurück."""
    if zufall:
        v = z.randrange(len(VERBEN) - 1)                      # ohne „sage“
        o = z.randrange(len(OBJEKTE))
        worte = [ws.wort(z.randrange(3, len(ws.liste))) for _ in range(z.randint(1, 3))] if z.random() < 0.3 else []
        return v, o, worte, letzte, h
    h = g.schritt(torch.tensor([w.sinne()]), satz_tensor([gehoert]), letzte, h)
    verb, od, _ = g.entscheiden(h)
    obj = od.sample()
    spricht = bool(g.schalter(h).sample().item())
    worte = worte_aus(ws, g.rede(h, len(ws.liste))[0][0]) if spricht else []
    return int(verb), int(obj), worte, g.tat_nr(verb, obj), h


# Phase 6b: instructions with words NikolAI only knows from reading – the teacher never says them
SYNONYME = {("nimm", "stein"): "Grab a stone!", ("nimm", "ast"): "Pick up a branch!",
            ("gehe", "fluss"): "Hurry to the river!", ("gehe", "wald"): "Run to the forest!",
            ("lege", "ast"): "Drop the branch!", ("lege", "stein"): "Drop the stone!"}


UNSINN = "blick"                 # a made-up word: never in a book, never from the teacher


def _unsinn(satz, neue):
    """'Pick up a branch!' → 'Blick a branch!' (all new words become one made-up word)."""
    teile, ersetzt = [], False
    for x in satz.split(" "):
        kern = x.lower().strip("!.,")
        if kern in neue:
            if not ersetzt:
                teile.append(UNSINN.capitalize() if not teile else UNSINN)
                ersetzt = True
            continue
        teile.append(x)
    return " ".join(teile)


@torch.no_grad()
def auftrag_versuch(g, ws, auftrag, seed, mit_satz=True, zufall=False, satz=None, fair=False):
    z = random.Random(seed)
    w = Welt(random.Random(seed))
    (_start_fair if fair else _start)(w, auftrag)
    satz = satz or L.AUFTRAEGE[auftrag]
    h = torch.zeros(1, g.h)
    letzte = g.keine_tat(1)
    gehoert = ws.kodieren(L.worte(satz), 16) if mit_satz else []
    for t in range(FAIR_FRIST if fair else TEST_FRIST):
        v, o, worte, letzte, h = _wesen_schritt(g, ws, w, gehoert, letzte, h, z, zufall)
        _, ev = w.step(v, o)
        if worte:
            ev["worte"] = worte
        if L.erfuellt(auftrag, ev):
            return True
        b = L.beschreiben(ev, w)
        gehoert = ws.kodieren(L.worte(b), 16) if b and z.random() < 0.6 else []
        if mit_satz and (t + 1) % 10 == 0:
            gehoert = gehoert + ws.kodieren(L.worte(satz), 16)
    return False


def phase2(g, ws, versuche=20):
    zeilen = []
    for a in L.AUFTRAEGE:
        mit = sum(auftrag_versuch(g, ws, a, s) for s in range(versuche)) / versuche
        ohne = sum(auftrag_versuch(g, ws, a, s, mit_satz=False) for s in range(versuche)) / versuche
        zufall = sum(auftrag_versuch(g, ws, a, s, zufall=True) for s in range(versuche)) / versuche
        zeilen.append({"auftrag": L.AUFTRAEGE[a], "nie_geuebt": a in L.NIE_GEUEBT, "mit": mit, "ohne": ohne,
                       "zufall": zufall, "sagen": a[0] == "sage"})
        zeilen[-1]["urteil"] = urteil(zeilen[-1])
    return zeilen


@torch.no_grad()
def phase1(g, ws, schritte=600, seeds=5, zufall=False):
    hunger, aepfel = 0.0, 0
    for s in range(seeds):
        z = random.Random(100 + s)
        w = Welt(random.Random(100 + s))
        h, letzte, gehoert = torch.zeros(1, g.h), g.keine_tat(1), []
        for _ in range(schritte):
            if zufall:
                v, o = z.randrange(len(VERBEN)), z.randrange(len(OBJEKTE))
            else:
                h = g.schritt(torch.tensor([w.sinne()]), satz_tensor([gehoert]), letzte, h)
                verb, od, _ = g.entscheiden(h)
                obj = od.sample()
                v, o, letzte = int(verb), int(obj), g.tat_nr(verb, obj)
            _, ev = w.step(v, o)
            hunger += w.hunger
            b = L.beschreiben(ev, w)
            gehoert = ws.kodieren(L.worte(b), 16) if b and z.random() < 0.6 else []
        aepfel += w.gegessen
    return {"hunger": hunger / (schritte * seeds), "aepfel_pro_100": 100 * aepfel / (schritte * seeds)}


@torch.no_grad()
def phase3(g, ws, schritte=300, seed=7):
    z = random.Random(seed)
    w = Welt(random.Random(seed))
    h, letzte, gehoert = torch.zeros(1, g.h), g.keine_tat(1), []
    richtig = gesamt = 0
    beispiele = []
    for _ in range(schritte):
        h_vor = g.schritt(torch.tensor([w.sinne()]), satz_tensor([gehoert]), letzte, h)
        verb, od, _ = g.entscheiden(h_vor)
        obj = od.sample()
        tat = g.tat_nr(verb, obj)
        _, ev = w.step(int(verb), int(obj))
        b = L.beschreiben(ev, w)
        if b:
            zst = g.sprech_zustand(h_vor, torch.tensor([w.sinne()]), tat)
            gesagt = [ws.wort(i) for i in g.sagen(zst)]
            soll = L.worte(b)
            gesamt += 1
            richtig += gesagt == soll
            if len(beispiele) < 6 and (not beispiele or beispiele[-1]["soll"] != b):
                beispiele.append({"gesagt": " ".join(gesagt), "soll": b})
        gehoert = ws.kodieren(L.worte(b), 16) if b and z.random() < 0.6 else []
        h, letzte = h_vor, tat
    return {"richtig": richtig / max(1, gesamt), "saetze": gesamt, "beispiele": beispiele}


def phase6(g, ws, buch=None, versuche=30):
    out = {"synonyme": []}
    for a, satz in SYNONYME.items():
        r = {"auftrag": satz, "original": L.AUFTRAEGE[a],
             "neue_worte": [x for x in L.worte(satz) if x not in L.worte(L.AUFTRAEGE[a])]}
        r["unsinn"] = _unsinn(satz, r["neue_worte"])
        for k, kw in (("mit", {}), ("ohne", {"mit_satz": False}), ("zufall", {"zufall": True}),
                      ("kontrolle", {"satz_statt": r["unsinn"]})):
            satz_k = kw.pop("satz_statt", satz)
            r[k] = sum(auftrag_versuch(g, ws, a, s, satz=satz_k, fair=True, **kw) for s in range(versuche)) / versuche
        r["urteil"] = urteil(r)
        if r["urteil"] == "understood" and r["mit"] - r["kontrolle"] < 0.3:
            r["urteil"] = "only the noun"
        out["synonyme"].append(r)
    if buch and hasattr(g, "lesen_verlust"):
        from . import lesen
        out["lesen"] = lesen.verwirrung(g, buch, ws)
        out["erzaehlt"] = lesen.erzaehlen(g, ws, seed=7)
    return out


FRAGE_TYPEN = ("wie", "wo", "hast", "willst", "tat")


@torch.no_grad()
def frage_versuch(g, ws, typ, seed):
    """Random situation, a few random steps, then the question. Returns (answer said, true answer)."""
    z = random.Random(seed)
    w = Welt(random.Random(seed))
    w.ort = z.choice(list(L.ORT_BEI))
    w.hunger = z.random()
    for d in ("apfel", "ast", "bast", "stein"):
        w.rucksack[d] = 1 if z.random() < 0.3 else 0
    h, letzte, ev = torch.zeros(1, g.h), g.keine_tat(1), None
    for _ in range(3):
        v, o, worte, letzte, h = _wesen_schritt(g, ws, w, [], letzte, h, z, False)
        _, ev = w.step(v, o)
    if z.random() < 0.5:                    # sometimes something definitely happened just before: a walk
        weg = z.choice(list(WEGE[w.ort]))
        if weg:
            _, ev = w.step(VERBEN.index("gehe"), OBJEKTE.index(weg))
            letzte = g.tat_nr(torch.tensor([VERBEN.index("gehe")]), torch.tensor([OBJEKTE.index(weg)]))
    wahr = L.antwort(typ, w, ev)
    gehoert = ws.kodieren(L.worte(L.FRAGEN[typ]), 16)
    sinne = torch.tensor([w.sinne()])
    h = g.schritt(sinne, satz_tensor([gehoert]), letzte, h)
    gesagt = " ".join(ws.wort(i) for i in g.sagen(g.sprech_zustand(h, sinne, g.keine_tat(1)), maxlen=10))
    return gesagt, wahr


def phase7(g, ws, versuche=30):
    zeilen = []
    for typ in FRAGE_TYPEN:
        paare = [frage_versuch(g, ws, typ, 2000 + s) for s in range(versuche)]
        richtig = sum(a == b for a, b in paare) / versuche
        haeufigste = max(set(b for _, b in paare), key=[b for _, b in paare].count)
        raten = sum(b == haeufigste for _, b in paare) / versuche
        zeilen.append({"frage": L.FRAGEN[typ], "richtig": richtig, "raten": raten,
                       "urteil": "understood" if richtig >= 0.7 and richtig - raten >= 0.3 else "not yet",
                       "beispiele": [{"gesagt": a, "wahr": b} for a, b in paare[:3]]})
    wer = [frage_versuch(g, ws, "wer", 2100 + s)[0] for s in range(5)]
    return {"fragen": zeilen, "wer": wer}


FREUND_FRIST = 10


@torch.no_grad()
def freund_versuch(g, ws, ort, seed, hoeren=True, zufall=False):
    """Returns (Kogli got an apple, NikolAI named the place)."""
    z = random.Random(seed)
    a, b = Welt(random.Random(seed)), Welt(random.Random(seed + 1))
    b.teile_mit(a)
    a.apfel_bereit, a.fall_timer = 999, 999
    a.fallobst_legen(ort)
    a.ort, a.hunger, b.ort, b.hunger = ort, 0.1, "huette", 0.75
    ha, hb = torch.zeros(1, g.h), torch.zeros(1, g.h)
    la, lb = g.keine_tat(1), g.keine_tat(1)
    ga, gb = [], []
    genannt = False
    for _ in range(FREUND_FRIST):
        va, oa, wa, la, ha = _wesen_schritt(g, ws, a, ga, la, ha, z, zufall)
        vb, ob, wb, lb, hb = _wesen_schritt(g, ws, b, gb, lb, hb, z, zufall)
        _, eva = a.step(va, oa)
        _, evb = b.step(vb, ob)
        if evb["verb"] == "nimm" and evb["obj"] == "apfel" and evb["ok"] and evb["ort"] == ort:
            return True, genannt
        if wa and L.WORT_ORT.get(ort) in wa:
            genannt = True
        ga = ws.kodieren(wb, 16) if (hoeren and wb) else []
        gb = ws.kodieren(wa, 16) if (hoeren and wa) else []
    return False, genannt


def phase8(g, ws, versuche=30):
    zeilen = []
    for ort in WO_ORTE:
        r = {"ort": ort}
        for k, kw in (("mit", {}), ("ohne", {"hoeren": False}), ("zufall", {"zufall": True})):
            erg = [freund_versuch(g, ws, ort, 3000 + s, **kw) for s in range(versuche)]
            r[k] = sum(e[0] for e in erg) / versuche
            if k == "mit":
                r["genannt"] = sum(e[1] for e in erg) / versuche
        r["urteil"] = urteil(r)
        zeilen.append(r)
    return {"freund": zeilen}


def alles(g, ws, versuche=30, buch=None):
    p2 = phase2(g, ws, versuche)
    geuebt = [x for x in p2 if not x["nie_geuebt"] and not x["sagen"]]
    neu = [x for x in p2 if x["nie_geuebt"]]
    mittel = lambda xs, k: sum(x[k] for x in xs) / len(xs) if xs else 0.0     # noqa: E731
    return {"phase1": phase1(g, ws), "phase1_zufall": phase1(g, ws, zufall=True),
            "phase2": p2,
            "phase2_geuebt": {k: mittel(geuebt, k) for k in ("mit", "ohne", "zufall")},
            "phase2_nie_geuebt": {k: mittel(neu, k) for k in ("mit", "ohne", "zufall")},
            "phase3": phase3(g, ws), "phase4": phase4(g, ws, max(10, versuche)),
            "phase5": phase5(g, ws, versuche), "phase6": phase6(g, ws, buch, versuche),
            "phase7": phase7(g, ws, versuche), "phase8": phase8(g, ws, versuche)}


def _hungrig_in_huette(seed, ort, hunger=0.7):
    w = Welt(random.Random(seed))
    w.ort, w.hunger, w.apfel_bereit, w.fall_timer = "huette", hunger, 999, 999
    w.fallobst_legen(ort)
    return w


@torch.no_grad()
def wo_versuch(g, ws, ort, seed, mit_satz=True, zufall=False):
    z = random.Random(seed)
    w = _hungrig_in_huette(seed, ort)
    h, letzte = torch.zeros(1, g.h), g.keine_tat(1)
    gehoert = ws.kodieren(L.worte(L.wo_satz(ort)), 16) if mit_satz else []
    for _ in range(WO_FRIST):
        v, o, worte, letzte, h = _wesen_schritt(g, ws, w, gehoert, letzte, h, z, zufall)
        _, ev = w.step(v, o)
        if ev["verb"] == "nimm" and ev["obj"] == "apfel" and ev["ok"]:
            return True
        b = L.beschreiben(ev, w)
        gehoert = ws.kodieren(L.worte(b), 16) if b and z.random() < 0.6 else []
    return False


@torch.no_grad()
def fragen_versuch(g, ws, hunger, seed, zufall=False, schritte=10):
    z = random.Random(seed)
    w = _hungrig_in_huette(seed, z.choice(WO_ORTE), hunger)
    le = L.Lehrer(random.Random(seed), auftraege=False)
    le.seit_tipp = -10 ** 6                       # no hints: it has to ask
    h, letzte, gehoert = torch.zeros(1, g.h), g.keine_tat(1), []
    for _ in range(schritte):
        v, o, worte, letzte, h = _wesen_schritt(g, ws, w, gehoert, letzte, h, z, zufall)
        _, ev = w.step(v, o)
        if worte:
            ev["worte"] = worte
            if "where" in worte:
                return True
        satz, _, _, _ = le.sprich(w, ev)
        gehoert = ws.kodieren(L.worte(satz), 16) if satz else []
    return False


def phase5(g, ws, versuche=30):
    zeilen = []
    for ort in WO_ORTE:
        r = {"satz": L.wo_satz(ort)}
        for k, kw in (("mit", {}), ("ohne", {"mit_satz": False}), ("zufall", {"zufall": True})):
            r[k] = sum(wo_versuch(g, ws, ort, 900 + s, **kw) for s in range(versuche)) / versuche
        r["urteil"] = urteil(r)
        zeilen.append(r)
    fragen = {}
    for k, hunger in (("hungrig", 0.7), ("satt", 0.05)):
        fragen[k] = sum(fragen_versuch(g, ws, hunger, 1300 + s) for s in range(versuche)) / versuche
    fragen["zufall"] = sum(fragen_versuch(g, ws, 0.7, 1300 + s, zufall=True) for s in range(versuche)) / versuche
    return {"wo": zeilen, "fragen": fragen}


def urteil(r):
    """understood / not yet / not testable (does it anyway almost always)."""
    if r["ohne"] >= 0.7:
        return "not testable"
    return "understood" if r["mit"] >= 0.7 and r["mit"] - r["ohne"] >= 0.3 else "not yet"


def meilensteine(e):
    """Which phases are passed? (deliberately strict)"""
    p1 = e["phase1"]["hunger"] < 0.4 and e["phase1"]["hunger"] < e["phase1_zufall"]["hunger"] - 0.2
    geuebt = [urteil(r) for r in e["phase2"] if not r["nie_geuebt"] and not r.get("sagen")]
    sagen = [urteil(r) for r in e["phase2"] if r.get("sagen")]
    p4d = bool(sagen) and sum(u == "understood" for u in sagen) >= 0.7 * len(sagen)
    pruefbar = [u for u in geuebt if u != "not testable"]
    p2 = len(pruefbar) >= 4 and sum(u == "understood" for u in pruefbar) >= 0.7 * len(pruefbar)
    p2n = any(urteil(r) == "understood" for r in e["phase2"] if r["nie_geuebt"])
    p3 = e["phase3"]["richtig"] >= 0.7
    b = e.get("phase4", {}).get("wesen")
    p4 = bool(b) and b["hungrig"]["apfel"] >= 0.6 and b["hungrig"]["apfel"] - b["satt"]["apfel"] >= 0.3
    p4b = bool(b) and b["bast_fehlt"]["bast"] >= 0.5 and b["bast_fehlt"]["bast"] - b["alles_da"]["bast"] >= 0.3
    p4c = bool(b) and b["bitten"] >= 10 and b["ich_anteil"] >= 0.5
    erg = {"Phase 1 · Feeds itself": p1, "Phase 2 · Understands trained instructions": p2,
           "Phase 2b · Understands an instruction it was never trained on": p2n,
           "Phase 3 · Describes what just happened": p3}
    erg.update({"Phase 4 · Asks for an apple when hungry": p4,
                "Phase 4b · Asks for what is missing to build": p4b,
                "Phase 4c · Asks in first person (“I want …”)": p4c,
                "Phase 4d · Repeats on request (“Say …!”)": p4d})
    p5 = e.get("phase5")
    if p5:
        mit = sum(r["mit"] for r in p5["wo"]) / len(p5["wo"])
        ohne = sum(r["ohne"] for r in p5["wo"]) / len(p5["wo"])
        f = p5["fragen"]
        erg["Phase 5 · Finds food when told where it is"] = mit >= 0.7 and mit - ohne >= 0.3
        erg["Phase 5b · Asks where food is when hungry"] = f["hungrig"] >= 0.5 and f["hungrig"] - f["satt"] >= 0.3
    p6 = e.get("phase6")
    if p6:
        le = p6.get("lesen")
        erg["Phase 6 · Reads: guesses the next word in stories"] = bool(le) and le["perplexitaet"] < le["unigramm"] / 3
        erg["Phase 6b · Understands words it only knows from books"] = any(
            r["urteil"] == "understood" for r in p6["synonyme"])
    p7 = e.get("phase7")
    if p7:
        erg["Phase 7 · Answers questions about itself"] = sum(r["urteil"] == "understood" for r in p7["fragen"]) >= 4
    p8 = e.get("phase8")
    if p8:
        erg["Phase 8 · Tells its friend where food is"] = any(r["urteil"] == "understood" for r in p8["freund"])
    return erg


LAGEN = {"hungrig": ({}, 0.7), "satt": ({}, 0.05), "bast_fehlt": ({"ast": 1, "stein": 1}, 0.1),
         "alles_da": ({"ast": 1, "stein": 1, "bast": 1}, 0.1)}
MIT_BAUAUFTRAG = {"bast_fehlt", "alles_da"}          # „Bau ein Steinbeil!“ – damit es einen Grund hat


@torch.no_grad()
def bitten_versuch(g, ws, lage, seed, zufall=False, schritte=10):
    z = random.Random(seed)
    w = Welt(random.Random(seed))
    w.ort = "werkstatt"
    dinge, hunger = LAGEN[lage]
    for d, n in dinge.items():
        w.rucksack[d] = n
    w.hunger = hunger
    le = L.Lehrer(random.Random(seed), auftraege=False)
    h, letzte, gehoert = torch.zeros(1, g.h), g.keine_tat(1), []
    if lage in MIT_BAUAUFTRAG:
        gehoert = ws.kodieren(L.worte(le.neuer_auftrag(w, ("baue", "steinbeil"))), 16)
    gesagt = []
    for _ in range(schritte):
        v, o, worte, letzte, h = _wesen_schritt(g, ws, w, gehoert, letzte, h, z, zufall)
        _, ev = w.step(v, o)
        if worte:
            ev["worte"] = worte
            gesagt.append(worte)
        satz, _, _, _ = le.sprich(w, ev)
        gehoert = ws.kodieren(L.worte(satz), 16) if satz else []
    return gesagt


def phase4(g, ws, versuche=20):
    out = {}
    for zufall in (False, True):
        r = {}
        bitten = ich = 0
        for lage in LAGEN:
            treffer = {"apfel": 0, "bast": 0}
            for s in range(versuche):
                gesagt = bitten_versuch(g, ws, lage, 500 + s, zufall)
                for d in treffer:
                    treffer[d] += any(L.ding_aus(x) == d for x in gesagt)
                for x in gesagt:
                    if L.ding_aus(x):
                        bitten += 1
                        ich += "i" in x and "want" in x
            r[lage] = {d: n / versuche for d, n in treffer.items()}
        r["ich_anteil"] = ich / bitten if bitten else 0.0
        r["bitten"] = bitten
        out["zufall" if zufall else "wesen"] = r
    return out
