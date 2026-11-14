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

## Build & Run

```bash
# Quickstart
make setup    # venv, torch (CPU), BabyLM corpus (~550 MB)
make read     # pre-reading phase (~8 hours)
make train    # live training, dashboard at http://<host>:8091

# Install as systemd service
make install-service
```

**Dependencies:** Only `torch` (CPU build). See `requirements.txt`.

## Development Conventions

- **Language:** All code in English (since v16)
- **Single file, single responsibility:** Each `neuro/*.py` has one clear role
- **No external config files:** Paths passed via CLI args or `config.sh`
- **Versioned workdirs:** `neuro17/`, `neuro18/`, `neuro19/` — each version gets its own directory
- **Brains saved as `.pt`:** Older German-named brains still load via `legacy.py`

## What to Avoid

- ❌ Don't add new dependencies without strong justification (currently only `torch`)
- ❌ Don't change the systemd service template variables without updating `config.example.sh`
- ❌ Don't hardcode paths — use `@VARIABLE@` templates or CLI args
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
| **Deployment** | `Makefile`, `config.example.sh`, `*.service`, `nikolai-public.sh` |
