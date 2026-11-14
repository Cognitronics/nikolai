# AGENT.md — Source of Truth for AI Agents

> **This file is the single source of truth for how AI agents should interact with this project.**
> CLAUDE.md imports actions from here — never duplicate instructions.

## Project Overview

**NikolAI** is a neural agent (~9M parameters) that learns English from scratch in a simulated world using reinforcement learning. It lives in a small world of 5 places, interacts with objects, learns language from a teacher, reads child-scale English (BabyLM corpus ~100M words), and has a friend (Kogli).

**Repository:** github.com/Cognitronics/nikolai  
**License:** MIT  
**Language:** Python (all code in English since v16)

## Architecture

```
neuro/
├── __main__.py       # CLI entry point (argparse + server.start_server)
├── brain.py          # Neural brain: GRU perception, hearing, memory, world model, speech, imagination
├── server.py         # Runtime: 32 parallel worlds, PPO loop, web server, live world, tests
├── world.py          # Simulated world: 5 places, things, hunger, recipe, windfall apples, sizes
├── teacher.py        # Teacher agent: describes, names, instructs, asks, listens, praises
├── practice.py       # PPO learning: curiosity reward, hindsight relabelling, size lessons
├── evaluation.py     # 10 test phases with baselines, verdicts, milestones
├── reading.py        # BabyLM corpus reading: GRU language model, pre-reading
├── imagination.py    # Property-based imagination: 14 properties, Gaussian predictions w/ uncertainty
├── memory.py         # Word cloud, nearest words (cosine), Obsidian vault generation
├── names.py          # English name mappings for places, things, directions
├── legacy.py         # German→English key migration for old brain checkpoints
└── dashboard.html    # Interactive web dashboard (~635 lines HTML/CSS/JS)
```

## Key Concepts

| Concept | Details |
|---------|---------|
| **World** | 5 places (hut, orchard, forest, river, workshop), objects with properties, hunger, recipe (branch+stone+bast=axe) |
| **Brain** | ~9M params, GRU-based, 8K vocabulary, perception→memory→action loop |
| **Learning** | PPO (Schulman et al. 2017), curiosity (Pathak et al. 2017), hindsight relabelling (Andrychowicz et al. 2017) |
| **Reading** | BabyLM corpus (~100M words), GRU language model, pre-reading phase before live training |
| **Imagination** | Predictive coding (Rao & Ballard 1999): 14 properties (colour, size, shape, surface, weight, taste), uncertainty-aware |
| **Tests** | 10+ phases, each with controls (with sentence, without, random, made-up words). "Understood" = ≥70% and ≥30pts better than baseline |

## Build & Run (Current State)

The project currently requires manual setup — no Makefile or dependency file exists yet.

```bash
# 1. Create virtual environment
python3 -m venv ~/nikolai/neuro-venv
~/nikolai/neuro-venv/bin/pip install torch -i https://download.pytorch.org/whl/cpu

# 2. Clone the repo (if not already done)
git clone https://github.com/Cognitronics/nikolai ~/nikolai/code

# 3. Download BabyLM corpus (~550 MB)
mkdir -p ~/nikolai/books/babylm && cd ~/nikolai/books/babylm
for f in aochildes bnc_spoken cbt children_stories gutenberg \
         open_subtitles qed simple_wikipedia switchboard wikipedia; do
  wget -c https://huggingface.co/datasets/cambridge-climb/BabyLM/resolve/main/clean/100M/$f.txt
done

# 4. Pre-read (~8 hours)
cd ~/nikolai/code
python -m neuro.reading \
  --book ~/nikolai/books/babylm \
  --reader ~/nikolai/data/reader.pt \
  --hours 8

# 5. Live training
python -m neuro --web 8091 --host 0.0.0.0 \
  --save ~/nikolai/data/brain.pt \
  --reader ~/nikolai/data/reader.pt \
  --book ~/nikolai/books/babylm \
  --public-port 8092 \
  --obsidian ~/nikolai/obsidian
```

Dashboard: `http://<host>:8091` (private, interactive), `:8092` read-only public page.  
Brains saved every 10 minutes and on stop. Older German-named brains load via `legacy.py`.

## Development Conventions

- **Language:** All code in English (since v16)
- **Single file, single responsibility:** Each `neuro/*.py` has one clear role
- **No external config files:** Paths passed via CLI args
- **Versioned workdirs:** `neuro17/`, `neuro18/`, `neuro19/` — each version gets its own directory
- **Systemd services:** `nikolai-neuro.service` (live) and `nikolai-reading.service` (pre-read). Adjust user/paths for your system.

## What to Avoid

- ❌ Don't add new dependencies without strong justification (currently only `torch`)
- ❌ Don't hardcode paths in code — use CLI args (systemd files are the exception)
- ❌ Don't modify `dashboard.html` without understanding the Cognitronics design
- ❌ Don't change test phases without updating `evaluation.py` milestones

## File Categories

| Category | Files |
|----------|-------|
| **Core model** | `brain.py`, `imagination.py`, `names.py` |
| **World & interaction** | `world.py`, `teacher.py` |
| **Learning** | `practice.py`, `evaluation.py` |
| **Pre-reading** | `reading.py` |
| **Memory & output** | `memory.py`, `dashboard.html` |
| **Infrastructure** | `server.py`, `__main__.py`, `legacy.py` |
| **Deployment** | `nikolai-neuro.service`, `nikolai-reading.service`, `nikolai-public.sh` |
