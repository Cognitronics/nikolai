"""Üben: Das Gehirn lebt gleichzeitig in mehreren Kopien der Welt (wie viele Übungsstunden auf einmal) und lernt.

Lernsignale:
  * Körper: Hunger tut weh, Sattwerden tut gut.
  * Neugier: Was das Weltmodell nicht vorhergesagt hat, ist spannend (Belohnung = Überraschung, Pathak u. a. 2017).
    Je besser das Weltmodell wird, desto langweiliger wird das Bekannte – das Wesen sucht Neues.
  * Lehrer: Lob, wenn ein Auftrag erfüllt ist.
  * Sprechen: Das Gehirn versucht vorherzusagen, was der Lehrer gleich über seine Handlung sagen wird.
  * Hindsight (Version 6): Nach jeder gelungenen Handlung fragt das Üben: Welcher Auftrag wäre damit erfüllt gewesen?
    („Du hast einen Stein genommen“ → „Take a stone!“). Dann spielt es die letzten Schritte noch einmal durch, als
    hätte es diesen Satz am Anfang gehört, und lernt: Wer diesen Satz hört, tut das. So wird jede Handlung zu einem
    Lernbeispiel (Andrychowicz u. a. 2017, „Hindsight Experience Replay“; Ghosh u. a. 2019, GCSL; Chan u. a. 2019).
  * Antworten (Version 8): Fragt der Lehrer „Where are you?“, lernt das Sprechen im nächsten Schritt die wahre
    Antwort („i am in the forest“) – aus dem, was es gerade gehört hat und wie es ihm geht.
  * Freunde (Version 10): Die Welten sind paarweise geteilt – NikolAI und Kogli (ein Gehirn, zwei Körper, wie
    Zwillinge). Jeder hört, was der andere sagt, und spürt es mit, wenn der andere satt wird (geteilte Belohnung,
    wie bei kooperativen Agenten in der Forschung, z. B. Foerster u. a. 2016, Lowe u. a. 2017).
  * Lesen (Version 7): Nebenbei liest es weiter in seinem Buch (nächstes Wort erraten). So vergisst das Hören nicht,
    was es aus Büchern weiß, während es lernt, worauf sich die Wörter in seiner Welt beziehen.
Verfahren: Advantage Actor-Critic (A2C) mit GAE, Weltmodell und Sprechen werden gleichzeitig mittrainiert.
"""

import random
import time

import torch
import torch.nn.functional as F

from . import lehrer as L
from . import lesen
from .gehirn import ENDE, PAD, REDE_MAX, Gehirn, Wortschatz
from .welt import VERBEN, Welt

SAGE = VERBEN.index("sage")


def worte_aus(ws, zeile):
    return [ws.wort(int(i)) for i in zeile if int(i) not in (PAD, ENDE)]

GAMMA, LAMBDA = 0.97, 0.95
NEUGIER = 0.05
EPOCHEN, CLIP = 3, 0.2
HINDSIGHT_MAX = 5               # so viele Schritte vor der Handlung wird der Satz „nachträglich“ eingesetzt
HINDSIGHT_GEWICHT = 0.3
LESE_GEWICHT = 0.5
TEAM = 0.5                      # wie sehr es mitfühlt, wenn der Freund satt wird
SPRACHTEILE = ("wort.", "hoeren.", "lese_aus.", "hoer_proj.", "sprechen.", "sprech_aus.", "sprech_start.",
               "rede_zelle.", "rede_aus.")
REDE_KOSTEN = 0.04              # Reden kostet ein bisschen Mühe – sonst plappert es ununterbrochen          # PPO: dreimal über dieselbe Erfahrung, Änderung pro Schritt begrenzt


def satz_tensor(listen, maxlen=24):
    n = max([len(x) for x in listen] + [0])
    t = torch.zeros(len(listen), min(n, maxlen), dtype=torch.long)
    for i, x in enumerate(listen):
        x = x[:maxlen]
        if x:
            t[i, :len(x)] = torch.tensor(x)
    return t


class Uebung:
    def __init__(self, welten=32, seed=0, gehirn=None, wortschatz=None, lr=3e-4, rollout=20, leser=None, buch=None):
        self.z = random.Random(seed)
        self.lr = lr
        import os
        self.buch = lesen.Buch(buch, random.Random(seed)) if buch and os.path.exists(buch) else None
        self.mit_leser = False
        self.neuer_leser = False
        self.leser_id = None
        self.gelesen = 0
        if leser:                                    # hearing and speaking from the reading brain
            d = torch.load(leser, weights_only=False)
            wortschatz = Wortschatz(d["groesse"], d["woerter"])
            gehirn = Gehirn(vokab=wortschatz.groesse)
            gehirn.load_state_dict({k: v for k, v in d["gehirn"].items() if k.startswith(("wort.", "hoeren.", "lese_aus."))},
                                   strict=False)
            gehirn.vom_leser()
            self.mit_leser = self.neuer_leser = True
            self.leser_id = d.get("leser_id")
            self.gelesen = d.get("gelesen", 0)
        self.n = welten
        self.welten = [Welt(random.Random(seed * 1000 + i), ruhetage=True) for i in range(welten)]
        self.lehrer = [L.Lehrer(random.Random(seed * 7777 + i), name="kogli" if i % 2 else "nikolai")
                       for i in range(welten)]
        self.partner = [i + 1 if i % 2 == 0 else i - 1 for i in range(welten)] if welten % 2 == 0 else None
        self._paaren()
        self.ws = wortschatz or Wortschatz()
        self.gehirn = gehirn or Gehirn(vokab=self.ws.groesse)
        self.opt = torch.optim.Adam(self.gehirn.parameters(), lr=lr)
        self.T = rollout
        self.h = torch.zeros(welten, self.gehirn.h)
        self.letzte_tat = self.gehirn.keine_tat(welten)
        self.gehoert = [[] for _ in range(welten)]
        self.nachsprechen = [[] for _ in range(welten)]
        self.antworten = [[] for _ in range(welten)]
        self.sinne = torch.tensor([w.sinne() for w in self.welten])
        self.schritte = 0
        self.statistik = {"gesagt": 0, "belohnung": 0.0, "neugier": 0.0, "auftraege": 0, "erfuellt": 0, "gegessen": 0, "hunger": 0.0,
                          "sprech_fehler": 0.0, "welt_fehler": 0.0, "nachsprech_fehler": 0.0, "hindsight_fehler": 0.0, "lese_fehler": 0.0, "antwort_fehler": 0.0, "n": 0}
        self.neugier_skala = 1.0

    # ------------------------------------------------------------ ein Übungsabschnitt
    def handeln(self, h, bekannt):
        """Entscheiden (ohne Lernen): Handlung, ob es nebenbei redet, was es sagt, und wie wahrscheinlich das war."""
        g = self.gehirn
        verb, od, vd = g.entscheiden(h)
        obj = od.sample()
        sd = g.schalter(h)
        spricht = sd.sample().bool()
        worte, rede_lp, _ = g.rede(h, bekannt)
        logp = vd.log_prob(verb) + od.log_prob(obj) + sd.log_prob(spricht.float()) + torch.where(
            spricht, rede_lp, torch.zeros_like(rede_lp))
        return verb, obj, spricht, worte, logp

    def _neu_bewerten(self, h, s):
        g = self.gehirn
        _, od, vd = g.entscheiden(h, s["verb"])
        sd = g.schalter(h)
        _, rede_lp, rede_ent = g.rede(h, s["bekannt"], s["worte"])
        spricht = s["spricht"]
        logp = vd.log_prob(s["verb"]) + od.log_prob(s["obj"]) + sd.log_prob(spricht.float()) + torch.where(
            spricht, rede_lp, torch.zeros_like(rede_lp))
        ent = vd.entropy() + od.entropy() + sd.entropy() + 0.3 * torch.where(spricht, rede_ent, torch.zeros_like(rede_ent))
        return logp, ent

    def _paaren(self):
        """NikolAI (gerade Nummern) und Kogli (ungerade) teilen sich jeweils eine Welt."""
        if not self.partner:
            return
        for i in range(0, self.n, 2):
            a, b = self.welten[i], self.welten[i + 1]
            if b.u is not a.u:
                b.teile_mit(a)
            self.lehrer[i].freund = (b, "kogli")
            self.lehrer[i + 1].freund = (a, "nikolai")

    def abschnitt(self):
        g = self.gehirn
        h0 = self.h.detach()
        speicher = []
        h = h0
        with torch.no_grad():
            for t in range(self.T):
                saetze = satz_tensor(self.gehoert)
                h = g.schritt(self.sinne, saetze, self.letzte_tat, h)
                bekannt = len(self.ws.liste)
                verb, obj, spricht, rede_worte, logp_alt = self.handeln(h, bekannt)
                wert = g.wert(h).squeeze(1)
                tat = g.tat_nr(verb, obj)
                erwartet = g.erwartung(h, tat)
                lohn, neu_gehoert, beschr, nach, hs, oks, ant = [], [], [], [], [], [], []
                koerper, evs = [], []
                for i, w in enumerate(self.welten):          # erst handeln alle …
                    k, ev = w.step(int(verb[i]), int(obj[i]))
                    if spricht[i]:
                        ev["worte"] = worte_aus(self.ws, rede_worte[i])
                        self.statistik["gesagt"] += 1
                    koerper.append(k)
                    evs.append(ev)
                for i, (w, le) in enumerate(zip(self.welten, self.lehrer)):   # … dann hören und lernen sie
                    ev, k = evs[i], koerper[i]
                    j = self.partner[i] if self.partner else None
                    le.freund_sagte = evs[j].get("worte", []) if j is not None else []
                    satz, l_lohn, b, _ = le.sprich(w, ev)
                    ant.append(self.ws.kodieren(L.worte(L.antwort(le.frage, w, ev, le.name))) + [ENDE]
                               if le.frage else [])
                    oks.append(bool(ev["ok"]) and ev["verb"] != "warte")
                    rueck = L.hindsight(ev)
                    hs.append(self.z.choice(rueck) if rueck else None)
                    team = TEAM * max(0.0, koerper[j]) if j is not None else 0.0     # mitfreuen, nicht mitleiden
                    lohn.append(k + team + l_lohn - (REDE_KOSTEN if spricht[i] else 0.0))
                    if ev["verb"] == "iss" and ev["ok"]:
                        self.statistik["gegessen"] += 1
                    self.statistik["erfuellt"] += l_lohn >= 1.0
                    vom_freund = self.ws.kodieren(le.freund_sagte) if le.freund_sagte else []
                    neu_gehoert.append(vom_freund + self.ws.kodieren(L.worte(satz)))
                    beschr.append(self.ws.kodieren(L.worte(b)) + [ENDE] if b else [])
                    vor = L.vorgesagt(satz)
                    nach.append(self.ws.kodieren(vor) + [ENDE] if vor else [])
                    if w.schritte >= 2000:          # ein „Tag“ ist um: frisch anfangen (Gedächtnis bleibt im Netz)
                        w.neu()
                self._paaren()
                sinne_neu = torch.tensor([w.sinne() for w in self.welten])
                fehler = ((erwartet - sinne_neu) ** 2).mean(1)
                neugier = (fehler / self.neugier_skala).clamp(max=3.0) * NEUGIER
                self.neugier_skala = 0.99 * self.neugier_skala + 0.01 * float(fehler.mean() + 1e-6)
                r = torch.tensor(lohn) + neugier
                speicher.append({"sinne": self.sinne, "saetze": saetze, "letzte": self.letzte_tat, "verb": verb,
                                 "obj": obj, "tat": tat, "r": r, "sinne_neu": sinne_neu, "beschr": beschr,
                                 "worte": rede_worte, "spricht": spricht, "bekannt": bekannt, "logp": logp_alt,
                                 "wert": wert, "nachsprechen": self.nachsprechen, "hs": hs, "ok": oks,
                                 "antworten": self.antworten})
                self.statistik["belohnung"] += float(torch.tensor(lohn).mean())
                self.statistik["neugier"] += float(neugier.mean())
                self.statistik["hunger"] += sum(w.hunger for w in self.welten) / self.n
                self.statistik["n"] += 1
                self.sinne, self.letzte_tat, self.gehoert, self.nachsprechen = sinne_neu, tat, neu_gehoert, nach
                self.antworten = ant
            h_end = g.schritt(self.sinne, satz_tensor(self.gehoert), self.letzte_tat, h)
            wert_end = g.wert(h_end).squeeze(1)
        # ---------- Vorteile (GAE) aus den Werten beim Erleben
        werte_alt = torch.stack([s["wert"] for s in speicher])
        r = torch.stack([s["r"] for s in speicher])
        vorteil = torch.zeros_like(r)
        gae = torch.zeros(self.n)
        naechster = wert_end
        for t in reversed(range(self.T)):
            delta = r[t] + GAMMA * naechster - werte_alt[t]
            gae = delta + GAMMA * LAMBDA * gae
            vorteil[t] = gae
            naechster = werte_alt[t]
        ziel = vorteil + werte_alt
        adv = (vorteil - vorteil.mean()) / (vorteil.std() + 1e-6)
        logp_alt = torch.stack([s["logp"] for s in speicher])
        # ---------- lernen (PPO): mehrmals über dieselbe Erfahrung, aber nie zu weit weg vom alten Verhalten
        for epoche in range(EPOCHEN):
            h = h0
            logps, ents, werte, welt_v, sprech_v, nach_v, antwort_v = [], [], [], [], [], [], []
            h_vor = []
            for s in speicher:
                h_vor.append(h)
                h = g.schritt(s["sinne"], s["saetze"], s["letzte"], h)
                lp, en = self._neu_bewerten(h, s)
                logps.append(lp)
                ents.append(en)
                werte.append(g.wert(h).squeeze(1))
                if epoche == 0:
                    welt_v.append(F.mse_loss(g.erwartung(h, s["tat"]), s["sinne_neu"]))
                    mit = [i for i, b in enumerate(s["beschr"]) if b]
                    if mit:
                        z = g.sprech_zustand(h[mit], s["sinne_neu"][mit], s["tat"][mit])
                        sprech_v.append(g.sprech_verlust(z, satz_tensor([s["beschr"][i] for i in mit])))
                    fr = [i for i, x in enumerate(s["antworten"]) if x]
                    if fr:                                    # answering: the true answer, from what it just heard
                        z = g.sprech_zustand(h[fr], s["sinne"][fr], g.keine_tat(len(fr)))
                        antwort_v.append(g.sprech_verlust(z, satz_tensor([s["antworten"][i] for i in fr])))
                    vor = [i for i, x in enumerate(s["nachsprechen"]) if x]
                    if vor:                                   # Nachsprechen: das Vorgesagte selbst sagen
                        ziel_w = torch.zeros(len(vor), REDE_MAX, dtype=torch.long)
                        for j, i in enumerate(vor):
                            x = s["nachsprechen"][i][:REDE_MAX]
                            ziel_w[j, :len(x)] = torch.tensor(x)
                        _, lp_n, _ = g.rede(h[vor], s["bekannt"], ziel_w)
                        an = g.sprich_schalter(h[vor]).squeeze(1)
                        nach_v.append(-lp_n.mean() + F.binary_cross_entropy_with_logits(an, torch.ones_like(an)))
            lp_t = torch.stack(logps)
            ratio = torch.exp(lp_t - logp_alt)
            pol = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()
            wert_v = F.mse_loss(torch.stack(werte), ziel)
            ent = torch.stack(ents).mean()
            verlust = pol + 0.5 * wert_v - 0.01 * ent
            if epoche == 0:
                welt_l = torch.stack(welt_v).mean()
                sprech_l = torch.stack(sprech_v).mean() if sprech_v else torch.tensor(0.0)
                nach_l = torch.stack(nach_v).mean() if nach_v else torch.tensor(0.0)
                hind_l = self._hindsight(speicher, h_vor)
                antwort_l = torch.stack(antwort_v).mean() if antwort_v else torch.tensor(0.0)
                verlust = verlust + welt_l + 0.5 * sprech_l + 0.3 * nach_l + HINDSIGHT_GEWICHT * hind_l + 1.0 * antwort_l
                lese_l = torch.tensor(0.0)
                if self.buch:                         # a few lines of its book
                    x = lesen.stapel(self.buch, self.ws, 16, 64)
                    lese_l = g.lesen_verlust(x)
                    self.gelesen += x.numel()
                    verlust = verlust + LESE_GEWICHT * lese_l
            self.opt.zero_grad()
            verlust.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 0.5)
            self.opt.step()
        self.h = h.detach()
        self.schritte += self.T * self.n
        self.statistik["sprech_fehler"] += float(sprech_l.detach())
        self.statistik["welt_fehler"] += float(welt_l.detach())
        self.statistik["nachsprech_fehler"] += float(nach_l.detach())
        self.statistik["hindsight_fehler"] += float(hind_l.detach())
        self.statistik["lese_fehler"] += float(lese_l.detach())
        self.statistik["antwort_fehler"] += float(antwort_l.detach())
        return float(verlust.detach())

    def _hindsight(self, speicher, h_vor):
        """Pro Welt eine gelungene Handlung: die Schritte davor noch einmal, mit dem passenden Satz am Anfang."""
        g = self.gehirn
        proben = []
        for i in range(self.n):
            treffer = [t for t, s in enumerate(speicher) if s["hs"][i]]
            if treffer:
                e = self.z.choice(treffer)
                proben.append((i, max(0, e - self.z.randint(0, HINDSIGHT_MAX - 1)), e, speicher[e]["hs"][i]))
        if not proben:
            return torch.tensor(0.0)
        h = torch.stack([h_vor[s0][i] for i, s0, _, _ in proben]).detach()
        summe, anzahl = torch.tensor(0.0), 0
        for k in range(max(e - s0 for _, s0, e, _ in proben) + 1):
            ts = [min(s0 + k, e) for _, s0, e, _ in proben]
            # only steps that worked are imitated (failed attempts on the way are not "how to do it")
            aktiv = torch.tensor([s0 + k <= e and speicher[min(s0 + k, e)]["ok"][i] for i, s0, e, _ in proben],
                                 dtype=torch.float)
            saetze = []
            for (i, s0, e, satz), t in zip(proben, ts):
                gehoert = [int(x) for x in speicher[t]["saetze"][i] if int(x)] if speicher[t]["saetze"].shape[1] else []
                saetze.append(self.ws.kodieren(L.worte(satz)) + gehoert if k == 0 else gehoert)
            sinne = torch.stack([speicher[t]["sinne"][i] for (i, _, _, _), t in zip(proben, ts)])
            letzte = torch.stack([speicher[t]["letzte"][i] for (i, _, _, _), t in zip(proben, ts)])
            verb = torch.stack([speicher[t]["verb"][i] for (i, _, _, _), t in zip(proben, ts)])
            obj = torch.stack([speicher[t]["obj"][i] for (i, _, _, _), t in zip(proben, ts)])
            h = g.schritt(sinne, satz_tensor(saetze), letzte, h)
            _, od, vd = g.entscheiden(h, verb)
            lp = vd.log_prob(verb) + od.log_prob(obj)
            summe = summe - (lp * aktiv).sum()
            anzahl += int(aktiv.sum())
        return summe / max(1, anzahl)

    # ------------------------------------------------------------ Speichern
    def speichern(self, pfad, extra=None):
        torch.save({"gehirn": self.gehirn.state_dict(), "opt": self.opt.state_dict(), "woerter": self.ws.liste,
                    "groesse": self.ws.groesse, "gelesen": self.gelesen, "mit_leser": self.mit_leser,
                    "leser_id": self.leser_id,
                    "schritte": self.schritte, "extra": extra or {}}, pfad + ".tmp")
        import os
        os.replace(pfad + ".tmp", pfad)

    def laden(self, pfad):
        d = torch.load(pfad, weights_only=False)
        if self.neuer_leser:          # a new reading brain moves in: keep body, senses, actions; language from reading
            alt = {k: v for k, v in d["gehirn"].items() if not k.startswith(SPRACHTEILE)}
            self.gehirn.uebernehmen(alt)
            self.opt = torch.optim.Adam(self.gehirn.parameters(), lr=self.lr)
            self.schritte = d["schritte"]
            return d.get("extra", {})
        if d.get("mit_leser"):                                  # a brain that already reads: take it as it is
            self.ws = Wortschatz(d["groesse"], d["woerter"])
            self.gehirn = Gehirn(vokab=self.ws.groesse)
            self.gehirn.load_state_dict(d["gehirn"])
            self.opt = torch.optim.Adam(self.gehirn.parameters(), lr=self.lr)
            try:
                self.opt.load_state_dict(d["opt"])
            except (ValueError, KeyError):
                pass
            self.schritte, self.gelesen, self.mit_leser = d["schritte"], d.get("gelesen", 0), True
            self.leser_id = d.get("leser_id")
            return d.get("extra", {})
        if d.get("extra", {}).get("version", 1) >= 5:          # from version 5 on NikolAI learns English
            self.ws = Wortschatz(self.ws.groesse, d["woerter"])
        # an older (German) brain keeps what it knows about the world, but starts with an empty vocabulary
        neu_gebaut = self.gehirn.uebernehmen(d["gehirn"])
        if not neu_gebaut:
            try:
                self.opt.load_state_dict(d["opt"])
            except (ValueError, KeyError):
                pass                                   # neue Teile im Gehirn: Optimierer fängt frisch an
        self.schritte = d["schritte"]
        return d.get("extra", {})

    def bericht(self):
        s = self.statistik
        n = max(1, s["n"])
        out = {"schritte": self.schritte, "belohnung": s["belohnung"] / n, "neugier": s["neugier"] / n,
               "hunger": s["hunger"] / n, "gegessen_pro_1000": 1000 * s["gegessen"] / (n * self.n),
               "auftraege_erfuellt_pro_1000": 1000 * s["erfuellt"] / (n * self.n),
               "sprech_fehler": s["sprech_fehler"] / max(1, n / self.T),
               "welt_fehler": s["welt_fehler"] / max(1, n / self.T), "woerter": len(self.ws.liste) - 3,
               "gesagt_pro_1000": 1000 * s["gesagt"] / (n * self.n),
               "nachsprech_fehler": s["nachsprech_fehler"] / max(1, n / self.T),
               "hindsight_fehler": s["hindsight_fehler"] / max(1, n / self.T),
               "lese_fehler": s["lese_fehler"] / max(1, n / self.T), "gelesen": self.gelesen,
               "antwort_fehler": s["antwort_fehler"] / max(1, n / self.T),
               "wo_gefragt_pro_1000": 1000 * sum(le.bitten["wo"] - le._w0 for le in self.lehrer) / (n * self.n),
               "gegeben_pro_1000": 1000 * sum(le.bitten["gegeben"] - le._b0 for le in self.lehrer) / (n * self.n)}
        for le in self.lehrer:
            le._b0 = le.bitten["gegeben"]
            le._w0 = le.bitten["wo"]
        for k in s:
            s[k] = 0 if isinstance(s[k], int) else 0.0
        return out


def lauf(minuten=5, welten=32, seed=0, bericht_alle=2000):
    torch.manual_seed(seed)
    u = Uebung(welten=welten, seed=seed)
    t0, i = time.time(), 0
    while time.time() - t0 < minuten * 60:
        u.abschnitt()
        i += 1
        if i % (bericht_alle // u.T) == 0:
            b = u.bericht()
            print(f"{int(time.time() - t0):5d}s {b['schritte']:9d} Schritte | Hunger {b['hunger']:.2f} | "
                  f"Äpfel/1000 {b['gegessen_pro_1000']:.1f} | Aufträge/1000 {b['auftraege_erfuellt_pro_1000']:.1f} | "
                  f"Sprechfehler {b['sprech_fehler']:.2f} | Weltfehler {b['welt_fehler']:.4f}", flush=True)
    return u
