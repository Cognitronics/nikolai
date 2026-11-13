"""Loading brains saved by older versions (up to version 15, when the code was still in German).

The network itself is unchanged – only names changed. This translates an old checkpoint (or reading brain) into the
new names, so a brain trained for days keeps everything it learned.
"""

import torch

# parts of the brain (state_dict prefixes)
TEILE = {"wahrnehmung": "perception", "hoeren": "hearing", "hoer_proj": "hear_proj", "sprechen": "speech",
         "sprech_aus": "speech_out", "sprech_start": "speech_start", "rede_zelle": "speak_cell", "rede_aus": "speak_out",
         "lese_aus": "read_out", "wort": "word", "verb_kopf": "verb_head", "obj_kopf": "obj_head", "tat": "act",
         "wert": "value", "weltmodell": "world_model", "gedaechtnis": "memory", "sprich_schalter": "talk_switch",
         "vorstellung": "imagination"}
OBEN = {"gehirn": "brain", "woerter": "words", "groesse": "capacity", "gelesen": "words_read",
        "mit_leser": "with_reader", "leser_id": "reader_id", "erfahrung": "experience",
        "dinge_puffer": "things_buffer", "schritte": "steps", "verlauf": "history"}
EXTRA = {"kurven": "curves", "meilensteine": "milestones", "erreicht": "reached", "m_verlauf": "m_history",
         "englisch_seit": "english_since", "lesen_seit": "reading_since", "wortbuch": "word_diary",
         "vor_verlauf": "imag_history", "laufzeit": "runtime"}
KURVEN = {"belohnung": "reward", "neugier": "curiosity", "gesagt_pro_1000": "said_per_1000",
          "gegeben_pro_1000": "given_per_1000", "gegessen_pro_1000": "eaten_per_1000",
          "auftraege_erfuellt_pro_1000": "instructions_done_per_1000", "sprech_fehler": "speech_error",
          "welt_fehler": "world_error", "nachsprech_fehler": "repeat_error", "hindsight_fehler": "hindsight_error",
          "lese_fehler": "reading_error", "gelesen": "words_read", "antwort_fehler": "answer_error",
          "vorstellung_fehler": "imagination_error", "wo_gefragt_pro_1000": "where_asked_per_1000",
          "laufzeit_h": "runtime_h", "schritte": "steps", "aepfel_pro_100": "apples_per_100",
          "auftraege": "instructions", "woerter": "words"}
ORTE = {"huette": "hut", "obstgarten": "orchard", "wald": "forest", "fluss": "river", "werkstatt": "workshop"}
SINNE = {"sehen": "seeing", "nehmen": "taking", "tragen": "carrying", "essen": "eating"}
STATUS = {"stabil": "stable", "zuletzt": "last", "nein": "no"}


def _teile(sd):
    out = {}
    for k, v in sd.items():
        kopf, _, rest = k.partition(".")
        out[TEILE.get(kopf, kopf) + ("." + rest if rest else "")] = v
    return out


def _erfahrung(e):
    return {wort: {SINNE.get(k, k): n for k, n in d.items()} for wort, d in (e or {}).items()}


def migrate(d):
    """Old (German) checkpoint or reading brain → new names. New ones pass through unchanged."""
    if not isinstance(d, dict) or "gehirn" not in d:
        return d
    d = {OBEN.get(k, k): v for k, v in d.items()}
    d["brain"] = _teile(d["brain"])
    d.pop("opt", None)                                 # optimizer state: starts fresh (harmless)
    if "experience" in d:
        d["experience"] = _erfahrung(d["experience"])
    x = d.get("extra") or {}
    x = {EXTRA.get(k, k): v for k, v in x.items()}
    x.pop("pruefung", None)                            # old test results: the next test run makes new ones
    x["curves"] = [{KURVEN.get(k, k): v for k, v in c.items()} for c in x.get("curves", [])]
    x["milestones"] = {k: STATUS.get(v, v) if isinstance(v, str) else v for k, v in x.get("milestones", {}).items()}
    if "word_diary" in x:
        x["word_diary"] = {w: {"n": e["n"], "hunger": e["hunger"],
                               "places": {ORTE.get(o, o): n for o, n in e.get("orte", {}).items()},
                               "first": e.get("erst", ""), "sentences": e.get("saetze", [])}
                           for w, e in x["word_diary"].items()}
    x["imag_history"] = [{"t": h.get("t", ""), "pictures": {w: {"mu": b["mu"], "sd": b["sd"],
                                                               "experience": _erfahrung({"_": b.get("erfahrung", {})})["_"]}
                                                           for w, b in h.get("bilder", {}).items()}}
                         for h in x.get("imag_history", [])]
    d["extra"] = x
    return d


def load(pfad):
    return migrate(torch.load(pfad, weights_only=False))
