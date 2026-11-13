"""Das neuronale Gehirn. Am Anfang sind alle Verbindungen zufällig – es weiß nichts, kennt kein Wort.

Aufbau (alles lernt gemeinsam, nichts ist vorprogrammiert außer den Sinnen):
  Sinne ──► Wahrnehmung ─┐
  Gehörtes ─► Hören ─────┼─► Gedächtnis (GRU) ─► Handeln (Verb, Ding) und Erwartung (wie gut ist die Lage?)
  letzte Handlung ───────┘          │
                                    ├─► Weltmodell: „Was werde ich gleich spüren?“ – der Fehler ist die Neugier
                                    ├─► Sprechen: sagt in Worten, was gerade passiert ist
                                    └─► Reden (Version 3): ein Schalter entscheidet in jedem Schritt, ob es nebenbei
                                        etwas sagt (bis zu fünf Wörter) – Reden und Handeln gleichzeitig, wie bei Kindern.
                                        Gelernt wird es auf zwei Wegen: Nachsprechen, was der Lehrer vorsagt
                                        („Sag: Ich will einen Apfel“), und Belohnung, wenn Gesagtes etwas bewirkt.

Version 7 – Lesen: Hören und Sprechen bekommen ein Sprachgedächtnis aus Büchern (lesen.py). Das Hören ist ein
Sprachmodell (GRU), das beim Lesen gelernt hat, das nächste Wort vorherzusagen. Sprechen und Reden starten mit
denselben Gewichten – NikolAI kann also schon „erzählen“, bevor es weiß, worauf sich die Wörter beziehen.

Wörter sind für das Gehirn am Anfang nur Nummern. Was „Apfel“ bedeutet, muss es selbst herausfinden –
aus dem Zusammenhang von Gehörtem, Gesehenem, Getanem und Belohnung.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .welt import OBJEKTE, SINNE, VERBEN

PAD, START, ENDE = 0, 1, 2
REDE_MAX = 8            # bis zu 7 Wörter + Ende ("there is an apple in the forest")
WORT_DIM = 128            # wie viele Zahlen ein Wort im Kopf hat
LESE_H = 256              # Gedächtnis beim Lesen (gleich groß wie das Hauptgedächtnis, damit Sprechen es übernehmen kann)


class Wortschatz:
    """Ordnet gehörten Wörtern Nummern zu – in der Reihenfolge, in der sie zum ersten Mal gehört werden."""

    def __init__(self, groesse=400, liste=None):
        self.groesse = groesse
        self.liste = list(liste or ["<pad>", "<start>", "<ende>"])
        self.nr = {w: i for i, w in enumerate(self.liste)}

    def kodieren(self, worte, maxlen=16):
        out = []
        for w in worte[:maxlen]:
            if w not in self.nr and len(self.liste) < self.groesse:
                self.nr[w] = len(self.liste)
                self.liste.append(w)
            if w in self.nr:
                out.append(self.nr[w])
        return out

    def wort(self, i):
        return self.liste[i] if i < len(self.liste) else "?"


class Gehirn(nn.Module):
    def __init__(self, vokab=400, h=256):
        super().__init__()
        self.h = h
        nv, no = len(VERBEN), len(OBJEKTE)
        self.nv, self.no = nv, no
        self.wahrnehmung = nn.Sequential(nn.Linear(SINNE, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.wort = nn.Embedding(vokab, WORT_DIM, padding_idx=PAD)
        self.hoeren = nn.GRU(WORT_DIM, LESE_H, batch_first=True)      # = der Leser
        self.lese_aus = nn.Linear(LESE_H, vokab)                    # Lesen: welches Wort kommt als Nächstes?
        self.hoer_proj = nn.Linear(LESE_H, 128)
        self.tat = nn.Embedding(nv * no + 1, 32)                 # letzte Handlung (+1: noch keine)
        self.gedaechtnis = nn.GRUCell(128 + 128 + 32, h)
        self.verb_kopf = nn.Linear(h, nv)
        self.verb_emb = nn.Embedding(nv, 32)
        self.obj_kopf = nn.Sequential(nn.Linear(h + 32, 128), nn.ReLU(), nn.Linear(128, no))
        self.wert = nn.Linear(h, 1)
        self.weltmodell = nn.Sequential(nn.Linear(h + 32, 256), nn.ReLU(), nn.Linear(256, SINNE))
        self.sprech_start = nn.Linear(h + 128 + 32, h)
        self.sprechen = nn.GRU(WORT_DIM, h, batch_first=True)
        self.sprech_aus = nn.Linear(h, vokab)
        self.vokab = vokab
        self.rede_zelle = nn.GRUCell(WORT_DIM, h)
        self.rede_aus = nn.Linear(h, vokab)
        self.sprich_schalter = nn.Linear(h, 1)
        nn.init.constant_(self.sprich_schalter.bias, -2.0)       # anfangs eher still

    def keine_tat(self, b):
        return torch.full((b,), self.nv * self.no, dtype=torch.long)

    def hoer(self, saetze):
        """saetze: (B, L) Wortnummern, 0 = nichts. Gibt (B, 128) zurück; Stille = Nullen."""
        if saetze.shape[1] == 0:
            return torch.zeros(saetze.shape[0], 128)
        laengen = (saetze != PAD).sum(1)
        aus, _ = self.hoeren(self.wort(saetze))
        idx = (laengen - 1).clamp(min=0)
        letzt = torch.tanh(self.hoer_proj(aus[torch.arange(saetze.shape[0]), idx]))
        return letzt * (laengen > 0).float().unsqueeze(1)

    # ------------------------------------------------------------ Lesen
    def leser_teile(self):
        return list(self.wort.parameters()) + list(self.hoeren.parameters()) + list(self.lese_aus.parameters())

    def lesen_verlust(self, x):
        """x: (B, L) Wortnummern aus einem Buch. Fehler beim Erraten des jeweils nächsten Wortes."""
        aus, _ = self.hoeren(self.wort(x[:, :-1]))
        logits = self.lese_aus(aus)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=PAD)

    @torch.no_grad()
    def weiterlesen(self, ids, n=40, temperatur=0.7, generator=None):
        """Erzählt eine Geschichte weiter (nur mit dem Leser)."""
        aus, hz = self.hoeren(self.wort(torch.tensor([ids])))
        logits = self.lese_aus(aus[:, -1])
        out = []
        for _ in range(n):
            logits[:, :4] = -1e9                                  # keine Sonderzeichen
            nxt = torch.multinomial(torch.softmax(logits / temperatur, -1), 1, generator=generator)
            out.append(int(nxt))
            aus, hz = self.hoeren(self.wort(nxt), hz)
            logits = self.lese_aus(aus[:, -1])
        return out

    def vom_leser(self):
        """Sprechen und Reden starten mit dem, was das Lesen gelernt hat."""
        with torch.no_grad():
            for ziel in (self.sprechen,):
                for k, v in self.hoeren.state_dict().items():
                    ziel.state_dict()[k].copy_(v)
            for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(self.rede_zelle, k).copy_(getattr(self.hoeren, k + "_l0"))
            for ziel in (self.sprech_aus, self.rede_aus):
                ziel.weight.copy_(self.lese_aus.weight)
                ziel.bias.copy_(self.lese_aus.bias)

    def schritt(self, sinne, saetze, letzte_tat, h):
        x = torch.cat([self.wahrnehmung(sinne), self.hoer(saetze), self.tat(letzte_tat)], 1)
        return self.gedaechtnis(x, h)

    def entscheiden(self, h, verb=None):
        """Gibt (verb, obj, logp, entropie) zurück. Mit verb=None wird gewürfelt."""
        vl = self.verb_kopf(h)
        if self.nv > 6:
            vl = vl.clone()
            vl[:, 6] = -1e9                     # „sage“ als eigene Handlung gibt es nicht mehr: Reden geht nebenbei
        vd = torch.distributions.Categorical(logits=vl)
        if verb is None:
            verb = vd.sample()
        ol = self.obj_kopf(torch.cat([h, self.verb_emb(verb)], 1))
        od = torch.distributions.Categorical(logits=ol)
        return verb, od, vd

    def rede(self, h, bekannt, worte=None):
        """Reden als Handlung: würfelt bis zu REDE_MAX Wörter (nur gehörte). Gibt (Wörter, logp, Entropie) zurück.
        Mit worte=(B, REDE_MAX) wird nur die Wahrscheinlichkeit dieser Wörter berechnet (fürs Lernen)."""
        b = h.shape[0]
        maske = torch.full((self.vokab,), -1e9)
        maske[4:max(5, bekannt)] = 0                     # 3 = <unk>: nie sagen
        maske[ENDE] = 0
        z = h
        eingabe = torch.full((b,), START, dtype=torch.long)
        aktiv = torch.ones(b, dtype=torch.bool)
        logp, ent = torch.zeros(b), torch.zeros(b)
        out = torch.full((b, REDE_MAX), PAD, dtype=torch.long)
        for k in range(REDE_MAX):
            z = self.rede_zelle(self.wort(eingabe), z)
            logits = self.rede_aus(z) + maske
            if k == 0:
                logits = logits.clone()
                logits[:, ENDE] = -1e9                     # mindestens ein Wort
            d = torch.distributions.Categorical(logits=logits)
            w = d.sample() if worte is None else torch.where(aktiv, worte[:, k], torch.full_like(worte[:, k], ENDE))
            logp = logp + torch.where(aktiv, d.log_prob(w), torch.zeros(b))
            ent = ent + torch.where(aktiv, d.entropy(), torch.zeros(b))
            out[:, k] = torch.where(aktiv, w, torch.full_like(w, PAD))
            aktiv = aktiv & (w != ENDE)
            eingabe = torch.where(aktiv, w, torch.full_like(w, ENDE))
        return out, logp, ent

    def schalter(self, h):
        return torch.distributions.Bernoulli(logits=self.sprich_schalter(h).squeeze(1))

    def tat_nr(self, verb, obj):
        return verb * self.no + obj

    def erwartung(self, h, tat):
        return self.weltmodell(torch.cat([h, self.tat(tat)], 1))

    def sprech_zustand(self, h, sinne_nachher, tat):
        return torch.tanh(self.sprech_start(torch.cat([h, self.wahrnehmung(sinne_nachher), self.tat(tat)], 1)))

    def sprech_verlust(self, z, ziel):
        """z: (B, h) Startzustand; ziel: (B, L) Wortnummern ohne START, mit ENDE. Mittlerer Fehler pro Wort."""
        ein = torch.cat([torch.full((ziel.shape[0], 1), START, dtype=torch.long), ziel[:, :-1]], 1)
        aus, _ = self.sprechen(self.wort(ein), z.unsqueeze(0))
        logits = self.sprech_aus(aus)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ziel.reshape(-1), ignore_index=PAD)

    def uebernehmen(self, alt):
        """Ein älteres Gehirn übernehmen: alles Gelernte bleibt, neue Teile starten zufällig."""
        neu = self.state_dict()
        nv_alt = alt["verb_kopf.weight"].shape[0]
        no = self.no
        for k, v in alt.items():
            if k not in neu:
                continue
            if neu[k].shape == v.shape:
                neu[k] = v
            elif k in ("verb_kopf.weight", "verb_kopf.bias", "verb_emb.weight"):
                neu[k][:v.shape[0]] = v
            elif k == "tat.weight":
                neu[k][:nv_alt * no] = v[:nv_alt * no]
                neu[k][self.nv * no] = v[nv_alt * no]          # „noch keine Handlung“ sitzt jetzt weiter hinten
        fehlt = [k for k in neu if k not in alt]
        self.load_state_dict(neu)
        return nv_alt != self.nv or bool(fehlt)

    @torch.no_grad()
    def sagen(self, z, maxlen=12, temperatur=0.0):
        """Spricht aus dem Zustand z (1, h) – Wort für Wort, bis zum Ende-Zeichen."""
        w = torch.tensor([[START]])
        hz = z.unsqueeze(0)
        out = []
        for _ in range(maxlen):
            aus, hz = self.sprechen(self.wort(w), hz)
            logits = self.sprech_aus(aus[:, -1])
            logits[:, [PAD, START, 3]] = -1e9                    # 3 = <unk> (with reading) – never said
            if temperatur > 0:
                nxt = torch.distributions.Categorical(logits=logits / temperatur).sample()
            else:
                nxt = logits.argmax(-1)
            n = int(nxt)
            if n == ENDE:
                break
            out.append(n)
            w = nxt.view(1, 1)
        return out
