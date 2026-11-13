"""Start: python3 -m neuro --web 8091 --host 0.0.0.0 --speicher ~/nikolai/neuro/gehirn.pt"""

import argparse
import os
import signal
import sys

from . import server


def main():
    a = argparse.ArgumentParser(description="NikolAI – a neural agent that learns language from scratch (Cognitronics)")
    a.add_argument("--web", type=int, default=8091, help="port of the dashboard")
    a.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable in the whole network")
    a.add_argument("--speicher", default="gehirn.pt", help="file the brain is saved to")
    a.add_argument("--welten", type=int, default=32, help="practice worlds at once")
    a.add_argument("--threads", type=int, default=6, help="CPU threads for learning")
    a.add_argument("--leser", default=None, help="reading brain from `python -m neuro.lesen` (version 7)")
    a.add_argument("--buch", default=None, help="book it keeps reading alongside, e.g. the BabyLM folder")
    a.add_argument("--public", action="store_true", help="read-only: no talking to NikolAI (for a public page)")
    args = a.parse_args()
    speicher = os.path.expanduser(args.speicher)
    os.makedirs(os.path.dirname(os.path.abspath(speicher)), exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # systemctl stop: save first
    ex = lambda p: os.path.expanduser(p) if p else None      # noqa: E731
    server.starten(args.web, args.host, speicher, args.welten, args.threads, args.public, ex(args.leser), ex(args.buch))


if __name__ == "__main__":
    main()
