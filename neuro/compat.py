"""Compatibility with brains saved by older versions (up to version 11, when the code still used German names).

A saved brain is a dictionary: the network weights, the vocabulary, the training step count and the dashboard
history. Before version 12 its keys and the names of the network parts were German ("gehirn", "woerter",
"wahrnehmung.0.weight", …). migrate() renames them, so NikolAI keeps everything it has learned when it moves to the
English code. New brains are saved with English names only.
"""

# network parts: old name → new name (the first part of every weight's name)
MODULES = {"wahrnehmung": "perception", "wort": "word", "hoeren": "hearing", "lese_aus": "read_out",
           "hoer_proj": "hear_proj", "tat": "act", "gedaechtnis": "memory", "verb_kopf": "verb_head",
           "obj_kopf": "obj_head", "wert": "value", "weltmodell": "world_model", "sprech_start": "speak_start",
           "sprechen": "speaking", "sprech_aus": "speak_out", "rede_zelle": "talk_cell", "rede_aus": "talk_out",
           "sprich_schalter": "speak_switch"}

# keys of the saved dictionary and of the dashboard history (training curves, milestones)
KEYS = {"gehirn": "brain", "woerter": "vocab_list", "groesse": "size", "gelesen": "words_read",
        "mit_leser": "with_reader", "leser_id": "reader_id", "schritte": "steps", "verlauf": "history",
        "fehler": "error", "perplexitaet": "ppl", "unigramm": "unigram",
        "kurven": "curves", "pruefung": "latest_tests", "meilensteine": "milestones", "erreicht": "reached",
        "m_verlauf": "m_history", "englisch_seit": "english_since", "lesen_seit": "reading_since",
        "laufzeit": "runtime_s", "laufzeit_h": "runtime_h", "belohnung": "rewards", "neugier": "curiosity",
        "gegessen_pro_1000": "eaten_per_1000", "auftraege_erfuellt_pro_1000": "instructions_done_per_1000",
        "sprech_fehler": "describe_error", "welt_fehler": "world_error", "nachsprech_fehler": "repeat_error",
        "hindsight_fehler": "hindsight_error", "lese_fehler": "read_error", "antwort_fehler": "answer_error",
        "gesagt_pro_1000": "said_per_1000", "wo_gefragt_pro_1000": "asked_where_per_1000",
        "gegeben_pro_1000": "given_per_1000"}
EXTRA_KEYS = dict(KEYS, woerter="words_known")          # in the history "woerter" was the number of words known
STATUS = {"stabil": "stable", "zuletzt": "latest", "nein": "no"}


def migrate_state_dict(sd):
    """Renames the weights of an old brain ("wahrnehmung.0.weight" → "perception.0.weight")."""
    out = {}
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        out[MODULES.get(head, head) + ("." + rest if rest else "")] = v
    return out


def _keys(x):
    if isinstance(x, dict):
        return {EXTRA_KEYS.get(k, k) if isinstance(k, str) else k: _keys(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_keys(v) for v in x]
    return x


def migrate(d):
    """Takes a loaded brain or reading-brain file (old or new) and returns it with English names.
    The optimizer state is kept as it is (it only counts parameters – names don't matter)."""
    if "gehirn" not in d:
        return d                                              # already new
    new = {KEYS.get(k, k): (v if k == "opt" else _keys(v)) for k, v in d.items() if k not in ("gehirn", "extra")}
    new["brain"] = migrate_state_dict(d["gehirn"])
    if "extra" in d:
        extra = _keys(d["extra"])
        extra["milestones"] = {k: STATUS.get(v, v) for k, v in (extra.get("milestones") or {}).items()}
        extra["latest_tests"] = None                          # old test results: the next test runs right away
        new["extra"] = extra
    return new
