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

from . import legacy
from . import names as E      # names of places and things
from . import memory as G
from . import imagination as V
from . import teacher as L
from . import reading
from . import evaluation
from .practice import SAY, Practice, sentence_tensor, words_from
from .world import OBJECTS, VERBS, World

PLACE_NAME = E.PLACE_NAME_EN
THING_NAME = {d: n for d, n in E.THING_EN.items()}
ACT_TEXT = {"go": "walks", "take": "takes", "eat": "eats", "put": "puts down", "build": "builds", "wait": "waits",
            "say": "says"}
DIRECTION_NAME = E.DIRECTION_EN


class Runtime:
    def __init__(self, storage, worlds=32, threads=6, name="NikolAI", is_public=False, reader=None, book=None,
                 obsidian=None):
        self.is_public = is_public
        self.obsidian = obsidian
        self.word_diary = G.WordDiary()
        self.imag_history = []
        torch.set_num_threads(threads)
        self.storage = storage
        self.name = name
        self.book_path = book if book and os.path.exists(book) else None
        self.lock = threading.Lock()
        old = legacy.load(storage) if os.path.exists(storage) else {}
        new_reader = None
        if reader and os.path.exists(reader):
            reader_id = legacy.load(reader).get("reader_id")
            if not old.get("with_reader") or old.get("reader_id") != reader_id:    # new reading brain → moves in
                new_reader = reader
        del old
        self.u = Practice(worlds=worlds, seed=int(time.time()) % 100000, reader=new_reader, book=book)
        self.reading_since = None
        self.log = []
        self.curves, self.last_eval, self.milestones, self.runtime = [], None, {}, 0.0
        self.reached = {}
        self.m_history = []
        self.english_since = 0
        if os.path.exists(storage):
            extra = self.u.restore(storage)
            self.runtime = extra.get("runtime", 0.0)
            if extra.get("version", 1) >= 5:
                self.curves = extra.get("curves", [])
                self.last_eval = extra.get("last_eval")
                self.milestones = extra.get("milestones", {})
                self.reached = extra.get("reached", {})
                self.m_history = extra.get("m_history", [])
                self.english_since = extra.get("english_since", 0)
            else:                                   # a German brain: English starts now, curves and milestones anew
                self.english_since = self.u.steps
            self.reading_since = extra.get("reading_since")
            self.word_diary = G.WordDiary(extra.get("word_diary"))
            self.imag_history = extra.get("imag_history", [])
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
        self.l_last = self.u.brain.no_act(1)
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
        self.k_last = self.u.brain.no_act(1)
        self.k_heard = []
        self.k_said, self.l_said = [], []

    def note(self, text, kind="act", spoken=False):
        """Log entry. gesprochen: something said out loud (shown as a quote)."""
        self.log.append({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text, "spoken": spoken})
        if spoken and kind in ("says", "describes"):          # what NikolAI says out loud → word memory
            self.word_diary.remember(text, self.lw)
        self.log = self.log[-120:]

    def live_step(self):
        g, w = self.u.brain, self.lw
        with torch.no_grad():                   # no lock: only reads the brain while it keeps practising
            if self.user_sentences:
                self.l_heard = self.l_heard + self.u.ws.encode_ids(L.words(" ".join(self.user_sentences)), 16)
                self.user_sentences = []
            evaluation.expect(g, w, self.l_heard)
            h = g.think(torch.tensor([w.senses()]), sentence_tensor([self.l_heard]), self.l_last, self.lh)
            if self.l_question:                                  # the teacher just asked something: answer
                self.l_question = False
                a = self._answer(h, w)
                if a:
                    self.note(a, "says", True)
            verb, od, _ = g.decide(h)
            obj = od.sample()
            act = g.act_index(verb, obj)
            _, ev = w.step(int(verb), int(obj), evaluation.choose_mod(g, h, verb, obj, w))
            self.l_ev = ev
            if bool(g.switch(h).sample().item()):          # talks alongside
                ev["words"] = words_from(self.u.ws, g.speak(h, len(self.u.ws.vocab_list))[0][0])
                said = " ".join(ev["words"])
                self.note(said, "says", True)
            self.l_said = ev.get("words", [])
            self._kogli_step()
            self.ll.friend_said = self.k_said
            sentence, reward, descr, done = self.ll.respond(w, ev)
            self.l_question = bool(self.ll.question)
            self.l_step += 1
            v, o = VERBS[int(verb)], OBJECTS[int(obj)]
            if v not in ("wait", "say"):
                what = DIRECTION_NAME.get(o) or THING_NAME.get(o, o)
                if ev.get("size") and ev["ok"]:
                    what = f"{ev['size']} {what}"
                extra_text = f" → {PLACE_NAME[ev['target']]}" if v == "go" and ev["ok"] else ""
                self.note(f"{ACT_TEXT[v]} {what}{extra_text} {'✓' if ev['ok'] else '✗'}", "act")
            if sentence:
                self.note(sentence, "teacher", True)
            if v not in ("wait", "say") and self.l_step % 3 == 0:
                z_state = g.speech_state(h, torch.tensor([w.senses()]), act)
                said = " ".join(self.u.ws.word(i) for i in g.utter(z_state))
                if said:
                    self.note(said[:1].upper() + said[1:], "describes", True)
            self.lh, self.l_last = h, act
            self.l_heard = (self.u.ws.encode_ids(self.k_said) if self.k_said else []) + \
                (self.u.ws.encode_ids(L.words(sentence), 16) if sentence else [])
            if w.steps >= 2000:
                w.fresh()
                self.kw.fresh()
                self.kw.share_with(w)

    def _kogli_step(self):
        """Kogli lives one step too: hears NikolAI and its own teacher, acts, maybe talks."""
        g, w = self.u.brain, self.kw
        evaluation.expect(g, w, self.k_heard)
        h = g.think(torch.tensor([w.senses()]), sentence_tensor([self.k_heard]), self.k_last, self.kh)
        verb, od, _ = g.decide(h)
        obj = od.sample()
        act = g.act_index(verb, obj)
        _, ev = w.step(int(verb), int(obj), evaluation.choose_mod(g, h, verb, obj, w))
        self.k_said = []
        if bool(g.switch(h).sample().item()):
            self.k_said = words_from(self.u.ws, g.speak(h, len(self.u.ws.vocab_list))[0][0])
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
        self.k_heard = (self.u.ws.encode_ids(self.l_said) if self.l_said else []) + \
            (self.u.ws.encode_ids(L.words(sentence), 16) if sentence else [])

    def _answer(self, h, w, temperature=0.0):
        g = self.u.brain
        senses = torch.tensor([w.senses()])
        ids = g.utter(g.speech_state(h, senses, g.no_act(1)), maxlen=20, temperature=temperature)
        text = " ".join(self.u.ws.word(i) for i in ids).replace(" .", ".").strip()
        return text[:1].upper() + text[1:] if text else ""

    def utter(self, text):
        if self.is_public:
            return {"task": False, "error": "read-only"}
        a = L.task_from_text(text)
        self.note(text, "you", True)
        if a:                                  # an instruction: the human is the teacher now
            self.ll.new_task(self.lw, a)
            self.user_sentences.append(text)
            return {"task": True}
        with torch.no_grad():                  # anything else: it hears it and answers
            g, w = self.u.brain, self.lw
            question = L.question_from_text(text)
            truth = L.answer(question, w, self.l_ev) if question else None      # same moment as what it senses
            senses = torch.tensor([w.senses()])
            heard = self.u.ws.encode_ids(L.words(text), 16)
            h = g.think(senses, sentence_tensor([heard]), self.l_last, self.lh)
            ids = g.utter(g.speech_state(h, senses, g.no_act(1)), maxlen=20, temperature=0.0 if question else 0.7)
            answer = " ".join(self.u.ws.word(i) for i in ids).replace(" .", ".").strip()
            answer = answer[:1].upper() + answer[1:]
            self.lh = h
        if answer:
            self.note(answer, "says", True)
        return {"task": False, "answer": answer, "truth": truth}

    # ------------------------------------------------------------ practise, test, save
    def practice(self):
        t_curve = t_eval = t_save = time.time()
        t_imag = 0
        last = time.time()
        while True:
            with self.lock:
                self.u.section()
            time.sleep(0.003)                  # let go briefly: live world and web page get their turn
            now = time.time()
            self.runtime += now - last
            last = now
            if now - t_curve >= 60:
                t_curve = now
                b = self.u.report()
                b["runtime_h"] = round(self.runtime / 3600, 3)
                self.curves = (self.curves + [b])[-2000:]
            if now - t_eval >= 20 * 60 or self.last_eval is None:
                t_eval = now
                with self.lock:
                    g, ws = copy.deepcopy(self.u.brain), copy.deepcopy(self.u.ws)
                threading.Thread(target=self._run_tests, args=(g, ws), daemon=True).start()
            if now - t_imag >= 1800:                  # every half hour: how it imagines things now (history)
                t_imag = now
                self.imag_history = (self.imag_history + [{"t": time.strftime("%d.%m. %H:%M"),
                                                         "pictures": self.imagination(self.GALLERY[:5])}])[-48:]
            if now - t_save >= 600:
                t_save = now
                self.save()
                if self.obsidian:                  # word memory into the Obsidian folder, every 10 minutes
                    try:
                        G.write_vault(self.u.brain, self.u.ws, self.word_diary, self.obsidian)
                    except OSError as e:
                        print("obsidian:", e, flush=True)

    def _run_tests(self, g, ws):
        """A milestone only counts once it holds in three tests in a row (otherwise it may be luck)."""
        book = reading.Book(self.book_path) if self.book_path else None      # own file handle for this thread
        e = evaluation.run_all(g, ws, trials=30, book=book)
        m = evaluation.milestones(e)
        self.m_history = (self.m_history + [m])[-3:]
        status = {}
        for k, ok in m.items():
            stable = len(self.m_history) == 3 and all(x.get(k) for x in self.m_history)
            status[k] = "stable" if stable else ("last" if ok else "no")
            if stable and k not in self.reached:
                self.reached[k] = {"steps": self.u.steps, "runtime_h": round(self.runtime / 3600, 2)}
                self.note(f"Milestone reached and stable: {k}", "milestone")
        e["steps"] = self.u.steps
        self.last_eval, self.milestones = e, status

    def save(self):
        with self.lock:
            self.u.save(self.storage, {"curves": self.curves, "last_eval": self.last_eval,
                                             "milestones": self.milestones, "reached": self.reached,
                                             "m_history": self.m_history, "version": 15,
                                             "english_since": self.english_since, "reading_since": self.reading_since,
                                             "word_diary": self.word_diary.to_dict(), "imag_history": self.imag_history,
                                             "runtime": self.runtime})

    GALLERY = ["apple", "branch", "stone", "bast", "stone axe", "dragon", "big apple", "small apple", "big stone"]

    @torch.no_grad()
    def imagination(self, words):
        """How NikolAI imagines these words: expected properties (mean, spread) and how much it has experienced."""
        g, ws = self.u.brain, self.u.ws
        out = {}
        for w in words:
            ids = ws.encode_ids(L.words(w)) if all(x in ws.index for x in L.words(w)) else []
            if not ids:
                continue
            mu, lv = g.imagine(sentence_tensor([ids]))
            out[w] = {"mu": [round(float(x), 3) for x in mu[0]], "sd": [round(float(x), 3) for x in lv[0].exp().sqrt()],
                      "experience": self.u.experience.get(w, {})}
        return out

    def state(self):
        """Everything the dashboard shows – in English, with the German originals of anything spoken."""
        w = self.lw
        p = self.last_eval
        if p:
            p = dict(p)
            p["phase2"] = [dict(r, en=r["task"], verdict_en=r.get("verdict")) for r in p["phase2"]]
            p["phase3"] = dict(p["phase3"], examples=[dict(b, said_en=b["said"], expected_en=b["expected"])
                                                       for b in p["phase3"]["examples"]])
        milestones = []
        for k, st in self.milestones.items():
            st = "last" if st is True else "no" if st is False else st
            milestones.append({"de": k, "en": k, "status": st,
                                 "reached": self.reached.get(k)})
        task = L.instruction(self.ll.task) if self.ll.task else None
        return {"name": self.name, "steps": self.u.steps, "runtime_h": round(self.runtime / 3600, 2),
                "words": len(self.u.ws.vocab_list) - 4 if self.u.with_reader else len(self.u.ws.vocab_list) - 3, "worlds": self.u.n, "public": self.is_public,
                "curves": list(self.curves[-600:]), "last_eval": p, "milestones": milestones,
                "log": list(self.log[-80:]),
                "live": {"place": PLACE_NAME[w.place], "hunger": round(w.hunger, 2),
                         "backpack": _backpack(w),
                         "task": task,
                         "apple_lies": PLACE_NAME[w.windfall] if w.windfall else None,
                         "tree_rests": w.apple_ready > 10 ** 8,
                         "kogli": {"place": PLACE_NAME[self.kw.place], "hunger": round(self.kw.hunger, 2),
                                   "backpack": _backpack(self.kw)}},
                "instructions": list(L.QUESTIONS.values()) + list(L.INSTRUCTIONS.values()), "english_since": self.english_since,
                "reads": self.u.with_reader, "words_read": self.u.words_read, "reading_since": self.reading_since,
                "cloud": self.word_diary.cloud(), "properties": V.PROPERTIES,
                "gallery": self.imagination(self.GALLERY), "imag_history": self.imag_history[-12:]}


def _backpack(w):
    """What it carries, with sizes (version 17): {"big apple": 1, "small stone": 2, "bast": 1}."""
    out = {}
    for d, n in w.backpack.items():
        big = w.backpack_big.get(d, 0)
        for name, k in (((f"big {THING_NAME[d]}", big), (f"small {THING_NAME[d]}", n - big)) if d in w.backpack_big
                        else ((THING_NAME[d], n),)):
            if k:
                out[name] = k
    return out


def start_server(port=8091, host="127.0.0.1", storage="brain.pt", worlds=32, threads=6, is_public=False, reader=None,
            book=None, public_port=None, obsidian=None):
    b = Runtime(storage, worlds=worlds, threads=threads, is_public=is_public, reader=reader, book=book,
                obsidian=obsidian)
    threading.Thread(target=b.practice, daemon=True).start()

    def live():
        while True:
            b.live_step()
            time.sleep(0.5)
    threading.Thread(target=live, daemon=True).start()
    page = open(os.path.join(os.path.dirname(__file__), "dashboard.html"), encoding="utf-8").read()

    def handler(read_only):
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
                    z = b.state()
                    if read_only:              # public view: read-only, and without what visitors typed
                        z["public"] = True
                        z["log"] = [e for e in z["log"] if e.get("kind") != "you"]
                    return self._json(z)
                if self.path.startswith("/imagine"):
                    from urllib.parse import parse_qs, urlparse
                    w = (parse_qs(urlparse(self.path).query).get("w") or [""])[0].strip().lower()[:40]
                    return self._json({"word": w, "picture": b.imagination([w]).get(w)})
                if self.path.startswith("/neighbours"):
                    from urllib.parse import parse_qs, urlparse
                    w = (parse_qs(urlparse(self.path).query).get("w") or [""])[0].strip().lower()[:30]
                    return self._json({"word": w, "neighbours": G.neighbours(b.u.brain, b.u.ws, [w]).get(w, []),
                                       "world": G.neighbours(b.u.brain, b.u.ws, [w], k=6, source="world").get(w, [])})
                if self.path.startswith("/obsidian.zip"):
                    data = G.vault_zip(b.u.brain, b.u.ws, b.word_diary)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", 'attachment; filename="NikolAI-memory.zip"')
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
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
                    return self._json({"error": "kein JSON"}, 400)
                if self.path.startswith("/say") and (b.is_public or read_only):
                    return self._json({"error": "read-only"}, 403)
                if self.path.startswith("/say"):
                    text = str(d.get("text", "")).strip()[:200]
                    return self._json(b.utter(text) if text else {"task": False})
                return self._json({"error": "unknown"}, 404)
        return H

    srv = ThreadingHTTPServer((host, port), handler(is_public))
    print(f"NikolAI{' (public, read-only)' if is_public else ''}: http://{host}:{port}  (brain: {storage}, {worlds} practice worlds, {threads} threads)", flush=True)
    if public_port:                             # a second door: the same NikolAI, read-only – for a public page
        pub = ThreadingHTTPServer((host, public_port), handler(True))
        threading.Thread(target=pub.serve_forever, daemon=True).start()
        print(f"NikolAI public, read-only: http://{host}:{public_port}", flush=True)
    try:
        srv.serve_forever()
    finally:
        b.save()
