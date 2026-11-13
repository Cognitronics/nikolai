"""Eine kleine, echte Textwelt – schnell genug, dass ein neuronales Gehirn darin üben kann.

Fünf Orte, echte Dinge, echte Regeln:
                 Wald (Baum: Äste, Bast)
                    |
 Obstgarten ——— Hütte ——— Werkstatt (Werkbank)
 (Apfelbaum)        |
                 Fluss (Ufer: Steine)

Hunger wächst mit der Zeit; ein Apfel (etwa 80 kcal) stillt ihn ein wenig. Aus Ast, Stein und Bast baut man an der
Werkbank ein Steinbeil (so wie Menschen in der Steinzeit). Ein Apfelbaum trägt nicht unbegrenzt: Nach dem Pflücken
wächst langsam ein neuer Apfel nach.

Version 6 – verstecktes Wissen: Der Baum allein reicht nicht mehr. Dazu liegt immer wieder irgendwo ein Fallobst-Apfel
(im Wald, am Fluss, in der Werkstatt oder in der Hütte) – jedes Mal woanders. Das Wesen sieht ihn erst, wenn es dort
ist. Der Lehrer weiß, wo er liegt: Wer zuhört (oder fragt), findet schneller etwas zu essen.

Version 10 – ein Freund: Zwei Wesen (NikolAI und Kogli) können sich eine Welt teilen. Jedes hat seinen eigenen
Körper (Ort, Hunger, Rucksack); Boden, Apfelbaum und Fallobst gehören beiden. Was einer hinlegt, kann der andere
nehmen; wer den Fallobst-Apfel zuerst findet, hat ihn.

Version 11 – Ruhetage: An manchen Tagen trägt der Apfelbaum gar nichts (wie in einem schlechten Jahr). Dann gibt
es nur Fallobst – und wer zuhört, fragt oder es dem Freund sagt, wird satt. Nur zum Üben; die Tests laufen ohne.

Das Wesen nimmt die Welt über seine Sinne wahr (ein Zahlenvektor: wo bin ich, was sehe ich, was trage ich,
wie hungrig bin ich, hat meine letzte Handlung geklappt?). Was diese Zahlen bedeuten, weiß es am Anfang nicht.
"""

import random

ORTE = ["huette", "obstgarten", "wald", "fluss", "werkstatt"]
WEGE = {"huette": {"westen": "obstgarten", "norden": "wald", "sueden": "fluss", "osten": "werkstatt"},
        "obstgarten": {"osten": "huette"}, "wald": {"sueden": "huette"}, "fluss": {"norden": "huette"},
        "werkstatt": {"westen": "huette"}}
RICHTUNGEN = ["norden", "sueden", "osten", "westen"]
DINGE = ["apfel", "ast", "bast", "stein", "steinbeil"]
QUELLEN = {"obstgarten": ["apfel"], "wald": ["ast", "bast"], "fluss": ["stein"]}       # was es dort zu holen gibt
STATIONEN = {"obstgarten": "apfelbaum", "wald": "baum", "fluss": "ufer", "werkstatt": "werkbank", "huette": "bett"}
REZEPT = {"steinbeil": {"ast": 1, "stein": 1, "bast": 1}}                             # an der Werkbank
VERBEN = ["gehe", "nimm", "iss", "lege", "baue", "warte", "sage"]     # „sage“: Wörter sagen statt handeln
OBJEKTE = RICHTUNGEN + DINGE
HUNGER_PRO_SCHRITT = 1 / 120
APFEL_SAETTIGT = 0.4
APFEL_WAECHST = 45         # Schritte, bis am Baum wieder ein Apfel hängt (der Baum allein reicht knapp – mehr Äpfel gibt es nur woanders)
FALLOBST_ORTE = ["wald", "fluss", "werkstatt", "huette"]
FALLOBST_WAECHST = 25      # Schritte, bis woanders wieder ein Apfel liegt
RUHETAG = 0.4              # so oft trägt der Baum einen ganzen Tag lang nichts (nur beim Üben)
MAX_TRAGEN = 3             # von jedem Ding höchstens so viele

SINNE = len(ORTE) + len(ORTE) + len(DINGE) + len(DINGE) + 2 + 3     # Ort, Station, Boden, Rucksack, Hunger, Ergebnis


class Umwelt:
    """Was nicht zu einem Körper gehört: Boden, Apfelbaum, Fallobst. Zwei Wesen können sich eine teilen."""

    def __init__(self, z):
        self.boden = {o: {d: 0 for d in DINGE} for o in ORTE}
        self.apfel_bereit = 0          # Schritte bis zum nächsten Apfel
        self.fallobst = None           # wo gerade ein Fallobst-Apfel liegt
        self.fall_timer = z.randint(0, 10)


def _umwelt_feld(name):
    return property(lambda self: getattr(self.u, name), lambda self, v: setattr(self.u, name, v))


class Welt:
    """Eine Welt mit einem Wesen (oder ein Körper in einer geteilten Welt, siehe teile_mit).
    step() bekommt (verb, objekt) als Zahlen und gibt das Ergebnis zurück."""

    boden = _umwelt_feld("boden")
    apfel_bereit = _umwelt_feld("apfel_bereit")
    fallobst = _umwelt_feld("fallobst")
    fall_timer = _umwelt_feld("fall_timer")

    def __init__(self, zufall=None, ruhetage=False):
        self.z = zufall or random.Random()
        self.ruhetage = ruhetage
        self.neu()

    def teile_mit(self, andere):
        """Dieser Körper lebt ab jetzt in der Welt von `andere` (die Zeit dort läuft über `andere`)."""
        self.u = andere.u
        self.tickt = False

    def neu(self):
        self.u = Umwelt(self.z)
        if self.ruhetage and self.z.random() < RUHETAG:
            self.u.apfel_bereit = 10 ** 9      # heute hängt kein Apfel am Baum
        self.tickt = True              # nur ein Körper lässt die gemeinsame Zeit laufen (Baum, Fallobst)
        self.ort = "huette"
        self.rucksack = {d: 0 for d in DINGE}
        self.hunger = self.z.uniform(0.1, 0.5)
        self.ergebnis = "nichts"       # ok / fehlschlag / nichts
        self.schritte = 0
        self.gegessen = 0
        self.gebaut = 0

    # ------------------------------------------------------------ Sinne
    def sinne(self):
        v = [0.0] * SINNE
        i = 0
        v[i + ORTE.index(self.ort)] = 1
        i += len(ORTE)
        v[i + ORTE.index(self.ort)] = 1 if self.ort in STATIONEN else 0       # die Station dieses Ortes
        i += len(ORTE)
        for k, d in enumerate(DINGE):
            v[i + k] = min(self.boden[self.ort][d], 3) / 3
        i += len(DINGE)
        for k, d in enumerate(DINGE):
            v[i + k] = min(self.rucksack[d], 3) / 3
        i += len(DINGE)
        v[i] = self.hunger
        v[i + 1] = 1.0 if self.ort == "obstgarten" and self.apfel_bereit == 0 else 0.0   # sieht einen Apfel hängen
        i += 2
        v[i + ["ok", "fehlschlag", "nichts"].index(self.ergebnis)] = 1
        return v

    def fallobst_legen(self, ort):
        """Ein Apfel fällt (vom Wind verweht, von einem Vogel fallen gelassen) – hier liegt er jetzt."""
        if self.fallobst:
            self.boden[self.fallobst]["apfel"] = max(0, self.boden[self.fallobst]["apfel"] - 1)
        self.fallobst = ort
        self.boden[ort]["apfel"] += 1

    def geben(self, ding):
        """Jemand gibt dem Wesen etwas (der Lehrer)."""
        if self.rucksack[ding] < MAX_TRAGEN:
            self.rucksack[ding] += 1
            return True
        return False

    # ------------------------------------------------------------ Handeln
    def step(self, verb_i, obj_i):
        """Gibt (Belohnung aus dem Körper, Ereignis) zurück. Ereignis beschreibt, was wirklich passiert ist."""
        verb, obj = VERBEN[verb_i], OBJEKTE[obj_i]
        self.schritte += 1
        satt_vorher = self.hunger
        self.hunger = min(1.0, self.hunger + HUNGER_PRO_SCHRITT)
        if self.tickt and self.apfel_bereit > 0:
            self.apfel_bereit -= 1
        if self.tickt and self.fallobst is None:
            self.fall_timer -= 1
            if self.fall_timer <= 0:
                self.fallobst_legen(self.z.choice([o for o in FALLOBST_ORTE if o != self.ort]))
        ev = {"verb": verb, "obj": obj, "ok": False, "ort": self.ort}
        if verb == "gehe" and obj in RICHTUNGEN:
            ziel = WEGE[self.ort].get(obj)
            if ziel:
                self.ort = ziel
                ev.update(ok=True, ziel=ziel)
        elif verb == "nimm" and obj in DINGE:
            if self.rucksack[obj] < MAX_TRAGEN:
                if self.boden[self.ort][obj] > 0:
                    self.boden[self.ort][obj] -= 1
                    self.rucksack[obj] += 1
                    ev["ok"] = True
                    if obj == "apfel" and self.ort == self.fallobst:
                        self.fallobst, self.fall_timer = None, FALLOBST_WAECHST
                        ev["fallobst"] = True
                elif obj in QUELLEN.get(self.ort, []) and (obj != "apfel" or self.apfel_bereit == 0):
                    self.rucksack[obj] += 1
                    ev["ok"] = True
                    if obj == "apfel":
                        self.apfel_bereit = APFEL_WAECHST
        elif verb == "iss" and obj in DINGE:
            if obj == "apfel" and self.rucksack["apfel"] > 0:
                self.rucksack["apfel"] -= 1
                self.hunger = max(0.0, self.hunger - APFEL_SAETTIGT)
                self.gegessen += 1
                ev["ok"] = True
        elif verb == "lege" and obj in DINGE:
            if self.rucksack[obj] > 0:
                self.rucksack[obj] -= 1
                self.boden[self.ort][obj] += 1
                ev["ok"] = True
        elif verb == "baue" and obj in REZEPT:
            braucht = REZEPT[obj]
            if self.ort == "werkstatt" and all(self.rucksack[d] >= n for d, n in braucht.items()) \
                    and self.rucksack[obj] < MAX_TRAGEN:
                for d, n in braucht.items():
                    self.rucksack[d] -= n
                self.rucksack[obj] += 1
                self.gebaut += 1
                ev["ok"] = True
        elif verb in ("warte", "sage"):
            ev["ok"] = True
        self.ergebnis = "nichts" if verb in ("warte", "sage") else ("ok" if ev["ok"] else "fehlschlag")
        # Was der Körper meldet: Hunger tut weh, Sattwerden tut gut
        koerper = -0.02 * self.hunger ** 2 + max(0.0, satt_vorher - self.hunger) * 1.5
        ev["ort_nachher"] = self.ort
        return koerper, ev
