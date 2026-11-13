"""Start: python3 -m neuro --web 8091 --host 0.0.0.0 --save ~/nikolai/neuro/brain.pt"""

import argparse
import os
import signal
import sys

from . import server


def main():
    a = argparse.ArgumentParser(description="NikolAI – a neural agent that learns language from scratch (Cognitronics)")
    a.add_argument("--web", type=int, default=8091, help="port of the dashboard")
    a.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable in the whole network")
    # the German names of versions up to 15 still work, so old service files keep running
    a.add_argument("--save", "--speicher", dest="save", default="brain.pt", help="file the brain is saved to")
    a.add_argument("--worlds", "--welten", dest="worlds", type=int, default=32, help="practice worlds at once")
    a.add_argument("--threads", type=int, default=6, help="CPU threads for learning")
    a.add_argument("--reader", "--leser", dest="reader", default=None,
                   help="reading brain from `python -m neuro.reading` (moves in once)")
    a.add_argument("--book", "--buch", dest="book", default=None,
                   help="book it keeps reading alongside, e.g. the BabyLM folder")
    a.add_argument("--public-port", type=int, default=None, help="second port: the same NikolAI, read-only (public page)")
    a.add_argument("--obsidian", default=None, help="folder for the Obsidian vault of its word memory (updated every 10 min)")
    a.add_argument("--public", action="store_true", help="read-only: no talking to NikolAI (for a public page)")
    args = a.parse_args()
    save = os.path.expanduser(args.save)
    os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # systemctl stop: save first
    ex = lambda p: os.path.expanduser(p) if p else None      # noqa: E731
    server.start_server(args.web, args.host, save, args.worlds, args.threads, args.public, ex(args.reader),
                        ex(args.book), args.public_port, ex(args.obsidian))


if __name__ == "__main__":
    main()
