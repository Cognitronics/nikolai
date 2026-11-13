"""NikolAI's word memory made visible: a word cloud and an Obsidian vault.

Two real things, nothing invented:
* What it SAYS: every word NikolAI says out loud is counted – how often, where it was, how hungry it was, and a few
  recent sentences. (Wortbuch)
* What lies CLOSE TOGETHER in its head: every word has a position in its brain – a vector of 256 numbers. Words it
  uses alike lie close together; we measure that with the cosine similarity. Two views:
  - from books: the output side of its reading part (which word comes next) – there "red" lies near "black",
    "white", "gold" and "mother" near "sister", "father";
  - from its world: the output side of its speaking part, shaped by talking about its world – there "apple" lies
    near "stone", "orchard", "branch".
  (Output vectors give clearer neighbours than input vectors: Press & Wolf 2017.)
  These are the [[links]] in Obsidian – the graph view then shows its word clusters.

Honestly: the brain doesn't store notes. The vault is a map of its word vectors plus a diary of what it said – the
closest one can get to "looking into its memory" with this kind of network.
"""

import collections
import io
import re
import time
import zipfile

import torch

STOPWORDS = {"i", "am", "a", "an", "the", "to", "is", "in", "at", "you", "some", "and", "of", "it", ".", "there"}
SPECIAL = {"<pad>", "<start>", "<ende>", "<unk>"}
PLACE_EN = {"hut": "Hut", "orchard": "Orchard", "forest": "Forest", "river": "River", "workshop": "Workshop"}


class WordDiary:
    """Counts what NikolAI says: how often, where, how hungry, recent sentences."""

    def __init__(self, to_dict=None):
        self.w = {}
        for word, d in (to_dict or {}).items():
            self.w[word] = {"n": d["n"], "hunger": d["hunger"], "places": collections.Counter(d["places"]),
                            "first": d["first"], "sentences": collections.deque(d.get("sentences", []), maxlen=5)}

    def remember(self, text, world):
        words = re.findall(r"[a-z]+", text.lower())
        if not words:
            return
        for word in set(words):
            d = self.w.setdefault(word, {"n": 0, "hunger": 0.0, "places": collections.Counter(),
                                         "first": time.strftime("%Y-%m-%d %H:%M"),
                                         "sentences": collections.deque(maxlen=5)})
            d["n"] += 1
            d["hunger"] += world.hunger
            d["places"][world.place] += 1
            d["sentences"].append(text.lower())

    def to_dict(self):
        return {w: {"n": d["n"], "hunger": d["hunger"], "places": dict(d["places"]), "first": d["first"],
                    "sentences": list(d["sentences"])} for w, d in self.w.items()}

    def cloud(self, n=60):
        """The most-said words (without little words like 'i', 'the'), with average hunger and favourite place."""
        top = sorted(((w, d) for w, d in self.w.items() if w not in STOPWORDS), key=lambda x: -x[1]["n"])[:n]
        return [{"word": w, "n": d["n"], "hunger": round(d["hunger"] / d["n"], 2),
                 "place": PLACE_EN.get(d["places"].most_common(1)[0][0], "") if d["places"] else ""} for w, d in top]


@torch.no_grad()
def neighbours(g, ws, words, k=8, source="book"):
    """For each word: the k words closest to it in NikolAI's head (cosine similarity of the word vectors).
    quelle="buch": reading part, quelle="welt": speaking part."""
    e = (g.read_out.weight if source == "book" else g.speech_out.weight).detach()
    n = len(ws.vocab_list)
    valid = torch.tensor([w not in SPECIAL and w not in STOPWORDS for w in ws.vocab_list])
    e = torch.nn.functional.normalize(e[:n], dim=1)
    out = {}
    for word in words:
        i = ws.index.get(word)
        if i is None or i >= n:
            continue
        s = e @ e[i]
        s[~valid] = -2
        s[i] = -2
        vals, idx = s.topk(k)
        out[word] = [(ws.word(int(j)), round(float(v), 2)) for v, j in zip(vals, idx)]
    return out


def _world_words():
    from . import teacher as L
    return sorted({w for s in L.all_sentences() for w in L.words(s)} - STOPWORDS)


def vault(g, ws, word_diary, name="NikolAI"):
    """The Obsidian vault as {filename: text}."""
    said = {w for w, d in word_diary.w.items() if w not in STOPWORDS}
    core = sorted(said | set(_world_words()) | {"red", "green", "blue", "hungry", "happy", "sad", "mother", "eat"})
    nb = neighbours(g, ws, core)
    nw = neighbours(g, ws, core, k=5, source="world")
    everything = set(nb) | {v for d in (nb, nw) for lst in d.values() for v, _ in lst}
    nb.update(neighbours(g, ws, sorted(everything - set(nb)), k=5))        # one ring further, so links don't dangle
    files = {}
    for word, lst in sorted(nb.items()):
        d = word_diary.w.get(word)
        header = ["---", f"word: {word}", f"said: {d['n'] if d else 0}"]
        if d:
            header += [f"first_said: {d['first']}", f"hunger_when_said: {d['hunger'] / d['n']:.2f}",
                     f"places: [{', '.join(PLACE_EN.get(o, o) for o, _ in d['places'].most_common())}]"]
        header += ["tags: [" + ", ".join(t for t, ok in (("said", bool(d)), ("world", word in _world_words()),
                                                       ("books-only", not d and word not in _world_words())) if ok)
                 + "]", "---", ""]
        text = [f"# {word}", "", "**Close in its head – from books:**", ""]
        text += [f"- [[{v}]] · {s:.2f}" for v, s in lst]
        if word in nw:
            text += ["", "**Close in its head – from talking in its world:**", ""]
            text += [f"- [[{v}]] · {s:.2f}" for v, s in nw[word]]
        if d:
            text += ["", "**Where it said it:** " + " · ".join(f"[[{PLACE_EN.get(o, o)}]] ×{n}"
                                                                 for o, n in d["places"].most_common()),
                     "", "**Recently said:**", ""] + [f"> {s}" for s in d["sentences"]]
        else:
            text += ["", "_Never said out loud – known from " + ("its world." if word in _world_words() else "books.") + "_"]
        files[f"words/{word}.md"] = "\n".join(header + text) + "\n"
    for place, en in PLACE_EN.items():
        there_now = sorted(((w, d["places"][place]) for w, d in word_diary.w.items() if d["places"].get(place) and w not in STOPWORDS),
                      key=lambda x: -x[1])[:30]
        files[f"places/{en}.md"] = f"# {en}\n\nWords {name} says here:\n\n" + \
            "\n".join(f"- [[{w}]] ×{n}" for w, n in there_now) + "\n"
    files[f"{name}.md"] = (f"# {name}'s word memory\n\nExported {time.strftime('%Y-%m-%d %H:%M')}.\n\n"
                             f"- {len(word_diary.w)} different words said out loud\n- {len(ws.vocab_list) - 4} words known\n\n"
                             "Open the **graph view**: links are words that lie close together in its head "
                             "(cosine similarity of its word vectors, from reading and living). Notes tagged "
                             "`books-only` are words it never met in its world.\n\nStart: [[apple]] · [[hungry]] · "
                             "[[red]] · " + " · ".join(f"[[{e}]]" for e in PLACE_EN.values()) + "\n")
    return files


def vault_zip(g, ws, word_diary):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for fpath, text in vault(g, ws, word_diary).items():
            z.writestr("NikolAI/" + fpath, text)
    return buffer.getvalue()


def write_vault(g, ws, word_diary, folder):
    """Writes the vault into a folder (e.g. one synced with Obsidian). Only its own notes are replaced."""
    import os
    for fpath, text in vault(g, ws, word_diary).items():
        target = os.path.join(folder, fpath)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(text)
