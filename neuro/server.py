"""Continuous operation: the brain keeps practising, NikolAI lives in its own world in real time, the page shows it all.

* Practice worlds: the brain practises in several copies of the world at once (on a CPU it would take months otherwise).
* Live world: NikolAI lives here in real time (2 steps per second) with its current brain. You can talk to it: an
  instruction like "Take a stone!" makes you the teacher; anything else it simply hears.
* Every 20 minutes the tests in evaluation.py run on a copy of the brain. Every 10 minutes the brain is saved.
* Version 8: talk to it. Whatever you write that isn't an instruction, NikolAI hears in its world and answers with
  its speaking part (from reading, trained on the teacher's questions).
* Version 7: with --reader the brain gets hearing and speaking from reading (reading.py); with --book it keeps reading
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

from . import compat
from . import names as E        # display names of places and things
from . import teacher as L
from . import reading
from . import evaluation
from .practice import Practice, sentence_tensor, words_from
from .world import OBJECTS, VERBS, World

PLACE_NAME = E.PLACE_NAME_EN
THING_NAME = {d: n for d, n in E.THING_EN.items()}
ACTION_TEXT = {"go": "walks", "take": "takes", "eat": "eats", "put": "puts down", "build": "builds", "wait": "waits",
            "say": "says"}
DIRECTION_NAME = E.DIRECTION_EN


class Runtime:
    def __init__(self, save_path, worlds=32, threads=6, name="NikolAI", public_view=False, reader=None, book=None):
        self.public_view = public_view
        torch.set_num_threads(threads)
        self.save_path = save_path
        self.name = name
        self.book_path = book if book and os.path.exists(book) else None
        self.lock = threading.Lock()
        old = compat.migrate(torch.load(save_path, weights_only=False)) if os.path.exists(save_path) else {}
        new_reader = None
        if reader and os.path.exists(reader):
            reader_id = compat.migrate(torch.load(reader, weights_only=False)).get("reader_id")
            if not old.get("with_reader") or old.get("reader_id") != reader_id:    # new reading brain → moves in
                new_reader = reader
        del old
        self.u = Practice(worlds=worlds, seed=int(time.time()) % 100000, reader=new_reader, book=book)
        self.reading_since = None
        self.log = []
        self.curves, self.latest_tests, self.milestones, self.runtime_s = [], None, {}, 0.0
        self.reached = {}
        self.m_history = []
        self.english_since = 0
        if os.path.exists(save_path):
            extra = self.u.load_from(save_path)
            self.runtime_s = extra.get("runtime_s", 0.0)
            if extra.get("version", 1) >= 5:
                self.curves = extra.get("curves", [])
                self.latest_tests = extra.get("latest_tests")
                self.milestones = extra.get("milestones", {})
                self.reached = extra.get("reached", {})
                self.m_history = extra.get("m_history", [])
                self.english_since = extra.get("english_since", 0)
            else:                                   # a German brain: English starts now, curves and milestones anew
                self.english_since = self.u.steps
            self.reading_since = extra.get("reading_since")
        if new_reader:                               # the reading brain moves in now
            self.reading_since = self.u.steps
            self.note("NikolAI now hears and speaks with what it learned from reading.", "milestone")
        self.live_reset()
        self.user_sentences = []

    # ------------------------------------------------------------ live world
    def live_reset(self):
        self.lw = World(random.Random(), rest_days=True)
        self.ll = L.Teacher(random.Random(), instructions=True)
        self.lh = torch.zeros(1, self.u.brain.h)
        self.l_last = self.u.brain.no_action(1)
        self.l_heard = []
        self.l_step = 0
        self.l_question = False
        self.l_ev = None
        # Kogli: NikolAI's friend in the same world (same brain, its own body and memory)
        self.kw = World(random.Random())
        self.kw.share_with(self.lw)
        self.kl = L.Teacher(random.Random(), instructions=True, name="kogli")
        self.ll.friend, self.kl.friend = (self.kw, "kogli"), (self.lw, "nikolai")
        self.kh = torch.zeros(1, self.u.brain.h)
        self.k_last = self.u.brain.no_action(1)
        self.k_heard = []
        self.k_said, self.l_said = [], []

    def note(self, text, kind="act", spoken=False):
        """Log entry. gesprochen: something said out loud (shown as a quote)."""
        self.log.append({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text, "spoken": spoken})
        self.log = self.log[-120:]

    def live_step(self):
        g, w = self.u.brain, self.lw
        with torch.no_grad():                   # no lock: only reads the brain while it keeps practising
            if self.user_sentences:
                self.l_heard = self.l_heard + self.u.ws.encode_words(L.words(" ".join(self.user_sentences)), 16)
                self.user_sentences = []
            h = g.tick(torch.tensor([w.senses()]), sentence_tensor([self.l_heard]), self.l_last, self.lh)
            if self.l_question:                                  # the teacher just asked something: answer
                self.l_question = False
                a = self._answer(h, w)
                if a:
                    self.note(a, "speaks", True)
            verb, od, _ = g.decide(h)
            obj = od.sample()
            act = g.act_id(verb, obj)
            _, ev = w.step(int(verb), int(obj))
            self.l_ev = ev
            if bool(g.talk_switch(h).sample().item()):          # redet nebenbei
                ev["words"] = words_from(self.u.ws, g.talk(h, len(self.u.ws.words_list))[0][0])
                said = " ".join(ev["words"])
                self.note(said, "speaks", True)
            self.l_said = ev.get("words", [])
            self._kogli_step()
            self.ll.friend_said = self.k_said
            sentence, reward, descr, done = self.ll.respond(w, ev)
            self.l_question = bool(self.ll.question)
            self.l_step += 1
            v, o = VERBS[int(verb)], OBJECTS[int(obj)]
            if v not in ("wait", "say"):
                what = DIRECTION_NAME.get(o) or THING_NAME.get(o, o)
                suffix = f" → {PLACE_NAME[ev['goal']]}" if v == "go" and ev["ok"] else ""
                self.note(f"{ACTION_TEXT[v]} {what}{suffix} {'✓' if ev['ok'] else '✗'}", "act")
            if sentence:
                self.note(sentence, "teacher", True)
            if v not in ("wait", "say") and self.l_step % 3 == 0:
                zst = g.speak_state(h, torch.tensor([w.senses()]), act)
                said = " ".join(self.u.ws.word(i) for i in g.speak(zst))
                if said:
                    self.note(said[:1].upper() + said[1:], "describes", True)
            self.lh, self.l_last = h, act
            self.l_heard = (self.u.ws.encode_words(self.k_said) if self.k_said else []) + \
                (self.u.ws.encode_words(L.words(sentence), 16) if sentence else [])
            if w.steps >= 2000:
                w.reset()
                self.kw.reset()
                self.kw.share_with(w)

    def _kogli_step(self):
        """Kogli lives one step too: hears NikolAI and its own teacher, acts, maybe talks."""
        g, w = self.u.brain, self.kw
        h = g.tick(torch.tensor([w.senses()]), sentence_tensor([self.k_heard]), self.k_last, self.kh)
        verb, od, _ = g.decide(h)
        obj = od.sample()
        act = g.act_id(verb, obj)
        _, ev = w.step(int(verb), int(obj))
        self.k_said = []
        if bool(g.talk_switch(h).sample().item()):
            self.k_said = words_from(self.u.ws, g.talk(h, len(self.u.ws.words_list))[0][0])
            ev["words"] = self.k_said
            if self.k_said:
                self.note(" ".join(self.k_said), "kogli", True)
        if ev["verb"] == "take" and ev["obj"] == "apple" and ev["ok"] and ev.get("windfall"):
            self.note(f"Kogli finds the apple in the {PLACE_NAME[w.place]}", "kogli_act")
        self.kl.friend_said = self.l_said
        sentence, _, _, _ = self.kl.respond(w, ev)
        if sentence and "Say:" in sentence:
            self.note(sentence, "teacher_kogli", True)
        self.kh, self.k_last = h, act
        self.k_heard = (self.u.ws.encode_words(self.l_said) if self.l_said else []) + \
            (self.u.ws.encode_words(L.words(sentence), 16) if sentence else [])

    def _answer(self, h, w, temperature=0.0):
        g = self.u.brain
        senses = torch.tensor([w.senses()])
        ids = g.speak(g.speak_state(h, senses, g.no_action(1)), maxlen=20, temperature=temperature)
        text = " ".join(self.u.ws.word(i) for i in ids).replace(" .", ".").strip()
        return text[:1].upper() + text[1:] if text else ""

    def speak(self, text):
        if self.public_view:
            return {"instruction": False, "error": "read-only"}
        a = L.instruction_from_text(text)
        self.note(text, "you", True)
        if a:                                  # an instruction: the human is the teacher now
            self.ll.new_instruction(self.lw, a)
            self.user_sentences.append(text)
            return {"instruction": True}
        with torch.no_grad():                  # anything else: it hears it and answers
            g, w = self.u.brain, self.lw
            question = L.question_from_text(text)
            true_answer = L.answer(question, w, self.l_ev) if question else None      # same moment as what it senses
            senses = torch.tensor([w.senses()])
            heard = self.u.ws.encode_words(L.words(text), 16)
            h = g.tick(senses, sentence_tensor([heard]), self.l_last, self.lh)
            ids = g.speak(g.speak_state(h, senses, g.no_action(1)), maxlen=20, temperature=0.0 if question else 0.7)
            answer = " ".join(self.u.ws.word(i) for i in ids).replace(" .", ".").strip()
            answer = answer[:1].upper() + answer[1:]
            self.lh = h
        if answer:
            self.note(answer, "speaks", True)
        return {"instruction": False, "answer": answer, "true_answer": true_answer}

    # ------------------------------------------------------------ practising, testing, saving
    def practice(self):
        t_curve = t_test = t_save = time.time()
        last = time.time()
        while True:
            with self.lock:
                self.u.practice_round()
            time.sleep(0.003)                  # let go briefly: the live world and the web page get their turn
            now = time.time()
            self.runtime_s += now - last
            last = now
            if now - t_curve >= 60:
                t_curve = now
                b = self.u.report()
                b["runtime_h"] = round(self.runtime_s / 3600, 3)
                self.curves = (self.curves + [b])[-2000:]
            if now - t_test >= 20 * 60 or self.latest_tests is None:
                t_test = now
                with self.lock:
                    g, ws = copy.deepcopy(self.u.brain), copy.deepcopy(self.u.ws)
                threading.Thread(target=self._run_tests, args=(g, ws), daemon=True).start()
            if now - t_save >= 600:
                t_save = now
                self.save_brain()

    def _run_tests(self, g, ws):
        """A milestone only counts once it holds in three tests in a row (otherwise it may be luck)."""
        book = reading.Book(self.book_path) if self.book_path else None      # own file handle for this thread
        e = evaluation.run_all(g, ws, trials=30, book=book)
        m = evaluation.milestones(e)
        self.m_history = (self.m_history + [m])[-3:]
        status = {}
        for k, ok in m.items():
            stable = len(self.m_history) == 3 and all(x.get(k) for x in self.m_history)
            status[k] = "stable" if stable else ("latest" if ok else "no")
            if stable and k not in self.reached:
                self.reached[k] = {"steps": self.u.steps, "runtime_h": round(self.runtime_s / 3600, 2)}
                self.note(f"Milestone reached and stable: {k}", "milestone")
        e["steps"] = self.u.steps
        self.latest_tests, self.milestones = e, status

    def save_brain(self):
        with self.lock:
            self.u.save_brain(self.save_path, {"curves": self.curves, "latest_tests": self.latest_tests,
                                             "milestones": self.milestones, "reached": self.reached,
                                             "m_history": self.m_history, "version": 12,
                                             "english_since": self.english_since, "reading_since": self.reading_since,
                                             "runtime_s": self.runtime_s})

    def state(self):
        """Everything the dashboard shows."""
        w = self.lw
        p = self.latest_tests
        milestones = []
        for k, st in self.milestones.items():
            st = "latest" if st is True else "no" if st is False else st
            milestones.append({"name": k, "status": st, "reached": self.reached.get(k)})
        instruction = L.INSTRUCTIONS.get(self.ll.instruction) if self.ll.instruction else None
        return {"name": self.name, "steps": self.u.steps, "runtime_h": round(self.runtime_s / 3600, 2),
                "words_known": len(self.u.ws.words_list) - 4 if self.u.with_reader else len(self.u.ws.words_list) - 3, "worlds": self.u.n, "public_view": self.public_view,
                "curves": list(self.curves[-600:]), "latest_tests": p, "milestones": milestones,
                "log": list(self.log[-80:]),
                "live": {"place": PLACE_NAME[w.place], "hunger": round(w.hunger, 2),
                         "backpack": {THING_NAME[d]: n for d, n in w.backpack.items() if n},
                         "instruction": instruction,
                         "apple_lies": PLACE_NAME[w.windfall] if w.windfall else None,
                         "tree_rests": w.apple_ready > 10 ** 8,
                         "kogli": {"place": PLACE_NAME[self.kw.place], "hunger": round(self.kw.hunger, 2),
                                   "backpack": {THING_NAME[d]: n for d, n in self.kw.backpack.items() if n}}},
                "instructions": list(L.QUESTIONS.values()) + list(L.INSTRUCTIONS.values()), "english_since": self.english_since,
                "is_reading": self.u.with_reader, "words_read": self.u.words_read, "reading_since": self.reading_since}


def start_server(port=8091, host="127.0.0.1", save_path="brain.pt", worlds=32, threads=6, public_view=False, reader=None,
            book=None):
    b = Runtime(save_path, worlds=worlds, threads=threads, public_view=public_view, reader=reader, book=book)
    threading.Thread(target=b.practice, daemon=True).start()

    def live():
        while True:
            b.live_step()
            time.sleep(0.5)
    threading.Thread(target=live, daemon=True).start()
    page = open(os.path.join(os.path.dirname(__file__), "page.html"), encoding="utf-8").read()

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
            if self.path.startswith("/state"):
                return self._json(b.state())
            data = page.encode()
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
                return self._json({"error": "no JSON"}, 400)
            if self.path.startswith("/say") and b.public_view:
                return self._json({"error": "read-only"}, 403)
            if self.path.startswith("/say"):
                text = str(d.get("text", "")).strip()[:200]
                return self._json(b.speak(text) if text else {"instruction": False})
            return self._json({"error": "unknown"}, 404)

    srv = ThreadingHTTPServer((host, port), H)
    print(f"NikolAI{' (public, read-only)' if public_view else ''}: http://{host}:{port}  (brain: {save_path}, {worlds} practice worlds, {threads} threads)", flush=True)
    try:
        srv.serve_forever()
    finally:
        b.save_brain()
