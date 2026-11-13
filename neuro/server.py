"""Continuous operation: the brain keeps practising, NikolAI lives in its own world in real time, the page shows it all.

* Practice worlds: the brain practises in several copies of the world at once (on a CPU it would take months otherwise).
* Live world: NikolAI lives here in real time (2 steps per second) with its current brain. You can talk to it: an
  instruction like "Take a stone!" makes you the teacher; anything else it simply hears.
* Every 20 minutes the tests in pruefen.py run on a copy of the brain. Every 10 minutes the brain is saved.
* Version 8: talk to it. Whatever you write that isn't an instruction, NikolAI hears in its world and answers with
  its speaking part (from reading, trained on the teacher's questions).
* Version 7: with --leser the brain gets hearing and speaking from reading (lesen.py); with --buch it keeps reading
  a little alongside.
"""

import copy
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from . import englisch as E      # names of places and things
from . import lehrer as L
from . import lesen
from . import pruefen
from .ueben import SAGE, Uebung, satz_tensor, worte_aus
from .welt import OBJEKTE, VERBEN, Welt

ORT_NAME = E.ORT_NAME_EN
DING_NAME = {d: n for d, n in E.DING_EN.items()}
TAT_TEXT = {"gehe": "walks", "nimm": "takes", "iss": "eats", "lege": "puts down", "baue": "builds", "warte": "waits",
            "sage": "says"}
RICHTUNG_NAME = E.RICHTUNG_EN


class Betrieb:
    def __init__(self, speicher, welten=32, threads=6, name="NikolAI", oeffentlich=False, leser=None, buch=None):
        self.oeffentlich = oeffentlich
        torch.set_num_threads(threads)
        self.speicher = speicher
        self.name = name
        self.buch_pfad = buch if buch and os.path.exists(buch) else None
        self.lock = threading.Lock()
        alt = torch.load(speicher, weights_only=False) if os.path.exists(speicher) else {}
        neu_leser = None
        if leser and os.path.exists(leser):
            leser_id = torch.load(leser, weights_only=False).get("leser_id")
            if not alt.get("mit_leser") or alt.get("leser_id") != leser_id:    # new reading brain → moves in
                neu_leser = leser
        del alt
        self.u = Uebung(welten=welten, seed=int(time.time()) % 100000, leser=neu_leser, buch=buch)
        self.lesen_seit = None
        self.log = []
        self.kurven, self.pruefung, self.meilensteine, self.laufzeit = [], None, {}, 0.0
        self.erreicht = {}
        self.m_verlauf = []
        self.englisch_seit = 0
        if os.path.exists(speicher):
            extra = self.u.laden(speicher)
            self.laufzeit = extra.get("laufzeit", 0.0)
            if extra.get("version", 1) >= 5:
                self.kurven = extra.get("kurven", [])
                self.pruefung = extra.get("pruefung")
                self.meilensteine = extra.get("meilensteine", {})
                self.erreicht = extra.get("erreicht", {})
                self.m_verlauf = extra.get("m_verlauf", [])
                self.englisch_seit = extra.get("englisch_seit", 0)
            else:                                   # a German brain: English starts now, curves and milestones anew
                self.englisch_seit = self.u.schritte
            self.lesen_seit = extra.get("lesen_seit")
        if neu_leser:                               # the reading brain moves in now
            self.lesen_seit = self.u.schritte
            self.notiz("NikolAI now hears and speaks with what it learned from reading.", "meilenstein")
        self.live_neu()
        self.user_saetze = []

    # ------------------------------------------------------------ Live-Welt
    def live_neu(self):
        self.lw = Welt(random.Random(), ruhetage=True)
        self.ll = L.Lehrer(random.Random(), auftraege=True)
        self.lh = torch.zeros(1, self.u.gehirn.h)
        self.l_letzte = self.u.gehirn.keine_tat(1)
        self.l_gehoert = []
        self.l_schritt = 0
        self.l_frage = False
        self.l_ev = None
        # Kogli: NikolAI's friend in the same world (same brain, its own body and memory)
        self.kw = Welt(random.Random())
        self.kw.teile_mit(self.lw)
        self.kl = L.Lehrer(random.Random(), auftraege=True, name="kogli")
        self.ll.freund, self.kl.freund = (self.kw, "kogli"), (self.lw, "nikolai")
        self.kh = torch.zeros(1, self.u.gehirn.h)
        self.k_letzte = self.u.gehirn.keine_tat(1)
        self.k_gehoert = []
        self.k_sagte, self.l_sagte = [], []

    def notiz(self, text, art="tat", gesprochen=False):
        """Log entry. gesprochen: something said out loud (shown as a quote)."""
        self.log.append({"t": time.strftime("%H:%M:%S"), "art": art, "text": text, "gesprochen": gesprochen})
        self.log = self.log[-120:]

    def live_schritt(self):
        g, w = self.u.gehirn, self.lw
        with torch.no_grad():                   # ohne Sperre: liest nur das Gehirn, während es weiterübt
            if self.user_saetze:
                self.l_gehoert = self.l_gehoert + self.u.ws.kodieren(L.worte(" ".join(self.user_saetze)), 16)
                self.user_saetze = []
            h = g.schritt(torch.tensor([w.sinne()]), satz_tensor([self.l_gehoert]), self.l_letzte, self.lh)
            if self.l_frage:                                  # the teacher just asked something: answer
                self.l_frage = False
                a = self._antwort(h, w)
                if a:
                    self.notiz(a, "sagt", True)
            verb, od, _ = g.entscheiden(h)
            obj = od.sample()
            tat = g.tat_nr(verb, obj)
            _, ev = w.step(int(verb), int(obj))
            self.l_ev = ev
            if bool(g.schalter(h).sample().item()):          # redet nebenbei
                ev["worte"] = worte_aus(self.u.ws, g.rede(h, len(self.u.ws.liste))[0][0])
                gesagt = " ".join(ev["worte"])
                self.notiz(gesagt, "sagt", True)
            self.l_sagte = ev.get("worte", [])
            self._kogli_schritt()
            self.ll.freund_sagte = self.k_sagte
            satz, lohn, beschr, fertig = self.ll.sprich(w, ev)
            self.l_frage = bool(self.ll.frage)
            self.l_schritt += 1
            v, o = VERBEN[int(verb)], OBJEKTE[int(obj)]
            if v not in ("warte", "sage"):
                was = RICHTUNG_NAME.get(o) or DING_NAME.get(o, o)
                zusatz = f" → {ORT_NAME[ev['ziel']]}" if v == "gehe" and ev["ok"] else ""
                self.notiz(f"{TAT_TEXT[v]} {was}{zusatz} {'✓' if ev['ok'] else '✗'}", "tat")
            if satz:
                self.notiz(satz, "lehrer", True)
            if v not in ("warte", "sage") and self.l_schritt % 3 == 0:
                zst = g.sprech_zustand(h, torch.tensor([w.sinne()]), tat)
                gesagt = " ".join(self.u.ws.wort(i) for i in g.sagen(zst))
                if gesagt:
                    self.notiz(gesagt[:1].upper() + gesagt[1:], "beschreibt", True)
            self.lh, self.l_letzte = h, tat
            self.l_gehoert = (self.u.ws.kodieren(self.k_sagte) if self.k_sagte else []) + \
                (self.u.ws.kodieren(L.worte(satz), 16) if satz else [])
            if w.schritte >= 2000:
                w.neu()
                self.kw.neu()
                self.kw.teile_mit(w)

    def _kogli_schritt(self):
        """Kogli lives one step too: hears NikolAI and its own teacher, acts, maybe talks."""
        g, w = self.u.gehirn, self.kw
        h = g.schritt(torch.tensor([w.sinne()]), satz_tensor([self.k_gehoert]), self.k_letzte, self.kh)
        verb, od, _ = g.entscheiden(h)
        obj = od.sample()
        tat = g.tat_nr(verb, obj)
        _, ev = w.step(int(verb), int(obj))
        self.k_sagte = []
        if bool(g.schalter(h).sample().item()):
            self.k_sagte = worte_aus(self.u.ws, g.rede(h, len(self.u.ws.liste))[0][0])
            ev["worte"] = self.k_sagte
            if self.k_sagte:
                self.notiz(" ".join(self.k_sagte), "kogli", True)
        if ev["verb"] == "nimm" and ev["obj"] == "apfel" and ev["ok"] and ev.get("fallobst"):
            self.notiz(f"Kogli finds the apple in the {ORT_NAME[w.ort]}", "kogli_tat")
        self.kl.freund_sagte = self.l_sagte
        satz, _, _, _ = self.kl.sprich(w, ev)
        if satz and "Say:" in satz:
            self.notiz(satz, "lehrer_kogli", True)
        self.kh, self.k_letzte = h, tat
        self.k_gehoert = (self.u.ws.kodieren(self.l_sagte) if self.l_sagte else []) + \
            (self.u.ws.kodieren(L.worte(satz), 16) if satz else [])

    def _antwort(self, h, w, temperatur=0.0):
        g = self.u.gehirn
        sinne = torch.tensor([w.sinne()])
        ids = g.sagen(g.sprech_zustand(h, sinne, g.keine_tat(1)), maxlen=20, temperatur=temperatur)
        text = " ".join(self.u.ws.wort(i) for i in ids).replace(" .", ".").strip()
        return text[:1].upper() + text[1:] if text else ""

    def sagen(self, text):
        if self.oeffentlich:
            return {"auftrag": False, "fehler": "read-only"}
        a = L.auftrag_aus_text(text)
        self.notiz(text, "du", True)
        if a:                                  # an instruction: the human is the teacher now
            self.ll.neuer_auftrag(self.lw, a)
            self.user_saetze.append(text)
            return {"auftrag": True}
        with torch.no_grad():                  # anything else: it hears it and answers
            g, w = self.u.gehirn, self.lw
            frage = L.frage_aus_text(text)
            wahr = L.antwort(frage, w, self.l_ev) if frage else None      # same moment as what it senses
            sinne = torch.tensor([w.sinne()])
            gehoert = self.u.ws.kodieren(L.worte(text), 16)
            h = g.schritt(sinne, satz_tensor([gehoert]), self.l_letzte, self.lh)
            ids = g.sagen(g.sprech_zustand(h, sinne, g.keine_tat(1)), maxlen=20, temperatur=0.0 if frage else 0.7)
            antwort = " ".join(self.u.ws.wort(i) for i in ids).replace(" .", ".").strip()
            antwort = antwort[:1].upper() + antwort[1:]
            self.lh = h
        if antwort:
            self.notiz(antwort, "sagt", True)
        return {"auftrag": False, "antwort": antwort, "wahr": wahr}

    # ------------------------------------------------------------ Üben, Prüfen, Speichern
    def ueben(self):
        t_kurve = t_pruef = t_speichern = time.time()
        letzte = time.time()
        while True:
            with self.lock:
                self.u.abschnitt()
            time.sleep(0.003)                  # kurz loslassen: Live-Welt und Webseite kommen dran
            jetzt = time.time()
            self.laufzeit += jetzt - letzte
            letzte = jetzt
            if jetzt - t_kurve >= 60:
                t_kurve = jetzt
                b = self.u.bericht()
                b["laufzeit_h"] = round(self.laufzeit / 3600, 3)
                self.kurven = (self.kurven + [b])[-2000:]
            if jetzt - t_pruef >= 20 * 60 or self.pruefung is None:
                t_pruef = jetzt
                with self.lock:
                    g, ws = copy.deepcopy(self.u.gehirn), copy.deepcopy(self.u.ws)
                threading.Thread(target=self._pruefen, args=(g, ws), daemon=True).start()
            if jetzt - t_speichern >= 600:
                t_speichern = jetzt
                self.speichern()

    def _pruefen(self, g, ws):
        """A milestone only counts once it holds in three tests in a row (otherwise it may be luck)."""
        buch = lesen.Buch(self.buch_pfad) if self.buch_pfad else None      # own file handle for this thread
        e = pruefen.alles(g, ws, versuche=30, buch=buch)
        m = pruefen.meilensteine(e)
        self.m_verlauf = (self.m_verlauf + [m])[-3:]
        status = {}
        for k, ok in m.items():
            stabil = len(self.m_verlauf) == 3 and all(x.get(k) for x in self.m_verlauf)
            status[k] = "stabil" if stabil else ("zuletzt" if ok else "nein")
            if stabil and k not in self.erreicht:
                self.erreicht[k] = {"schritte": self.u.schritte, "laufzeit_h": round(self.laufzeit / 3600, 2)}
                self.notiz(f"Milestone reached and stable: {k}", "meilenstein")
        e["schritte"] = self.u.schritte
        self.pruefung, self.meilensteine = e, status

    def speichern(self):
        with self.lock:
            self.u.speichern(self.speicher, {"kurven": self.kurven, "pruefung": self.pruefung,
                                             "meilensteine": self.meilensteine, "erreicht": self.erreicht,
                                             "m_verlauf": self.m_verlauf, "version": 11,
                                             "englisch_seit": self.englisch_seit, "lesen_seit": self.lesen_seit,
                                             "laufzeit": self.laufzeit})

    def zustand(self):
        """Everything the dashboard shows – in English, with the German originals of anything spoken."""
        w = self.lw
        p = self.pruefung
        if p:
            p = dict(p)
            p["phase2"] = [dict(r, en=r["auftrag"], urteil_en=r.get("urteil")) for r in p["phase2"]]
            p["phase3"] = dict(p["phase3"], beispiele=[dict(b, gesagt_en=b["gesagt"], soll_en=b["soll"])
                                                       for b in p["phase3"]["beispiele"]])
        meilensteine = []
        for k, st in self.meilensteine.items():
            st = "zuletzt" if st is True else "nein" if st is False else st
            meilensteine.append({"de": k, "en": k, "status": st,
                                 "erreicht": self.erreicht.get(k)})
        auftrag = L.AUFTRAEGE.get(self.ll.auftrag) if self.ll.auftrag else None
        return {"name": self.name, "schritte": self.u.schritte, "laufzeit_h": round(self.laufzeit / 3600, 2),
                "woerter": len(self.u.ws.liste) - 4 if self.u.mit_leser else len(self.u.ws.liste) - 3, "welten": self.u.n, "oeffentlich": self.oeffentlich,
                "kurven": list(self.kurven[-600:]), "pruefung": p, "meilensteine": meilensteine,
                "log": list(self.log[-80:]),
                "live": {"ort": ORT_NAME[w.ort], "hunger": round(w.hunger, 2),
                         "rucksack": {DING_NAME[d]: n for d, n in w.rucksack.items() if n},
                         "auftrag": auftrag,
                         "apfel_liegt": ORT_NAME[w.fallobst] if w.fallobst else None,
                         "baum_ruht": w.apfel_bereit > 10 ** 8,
                         "kogli": {"ort": ORT_NAME[self.kw.ort], "hunger": round(self.kw.hunger, 2),
                                   "rucksack": {DING_NAME[d]: n for d, n in self.kw.rucksack.items() if n}}},
                "auftraege": list(L.FRAGEN.values()) + list(L.AUFTRAEGE.values()), "englisch_seit": self.englisch_seit,
                "liest": self.u.mit_leser, "gelesen": self.u.gelesen, "lesen_seit": self.lesen_seit}


def starten(port=8091, host="127.0.0.1", speicher="gehirn.pt", welten=32, threads=6, oeffentlich=False, leser=None,
            buch=None):
    b = Betrieb(speicher, welten=welten, threads=threads, oeffentlich=oeffentlich, leser=leser, buch=buch)
    threading.Thread(target=b.ueben, daemon=True).start()

    def live():
        while True:
            b.live_schritt()
            time.sleep(0.5)
    threading.Thread(target=live, daemon=True).start()
    seite = open(os.path.join(os.path.dirname(__file__), "seite.html"), encoding="utf-8").read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, d, code=200):
            data = json.dumps(d, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/zustand"):
                return self._json(b.zustand())
            data = seite.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"fehler": "kein JSON"}, 400)
            if self.path.startswith("/sag") and b.oeffentlich:
                return self._json({"fehler": "read-only"}, 403)
            if self.path.startswith("/sag"):
                text = str(d.get("text", "")).strip()[:200]
                return self._json(b.sagen(text) if text else {"auftrag": False})
            return self._json({"fehler": "unbekannt"}, 404)

    srv = ThreadingHTTPServer((host, port), H)
    print(f"NikolAI{' (public, read-only)' if oeffentlich else ''}: http://{host}:{port}  (brain: {speicher}, {welten} practice worlds, {threads} threads)", flush=True)
    try:
        srv.serve_forever()
    finally:
        b.speichern()
