# NikolAI

*A Cognitronics project.* A neural agent that learns English from scratch. No sentence rules, no vocabulary list,
no planner: NikolAI only gets its senses (where am I, what do I see, what am I carrying, how hungry am I), hears
what its teacher says, reads children’s stories, and learns.

## Version 12: English code

The whole code base now uses English names – files, classes, functions, variables, command-line options and the
keys inside saved brains – so anyone can read and contribute. Nothing about NikolAI's behaviour changed: with the
same random seed, version 12 trains and tests exactly like version 11.

Brains saved by older versions still work: `neuro/compat.py` renames their contents when they are loaded, so
NikolAI keeps everything it has learned (the next save is in the new format, which older versions can't read –
keep a copy if you want to go back). The old German command-line options (`--speicher`, `--leser`, `--buch`,
`--stunden`, `--welten`) are still accepted.

## Version 11: practice where it’s hard

- **Rest days:** on about 40 % of practice days the apple tree bears nothing. Then there are only windfall apples –
  listening to hints, asking “where is an apple?” and telling the friend really decide who gets full. (The tests
  stay as they were, with a normal tree.)
- **Fair word tests (Phase 6b):** the start no longer gives the answer away – several things lie around (take the
  right one), it carries several things (put down the right one), or it stands in the hut with four ways and isn’t
  hungry. Before, many rows were “not testable” because it did the right thing anyway.

## Version 10: a friend

NikolAI is no longer alone: **Kogli** lives in the same world – its own body (place, hunger, backpack), the same
ground, apple tree and windfall apples. They hear what the other says, and each feels it a little when the other gets
full (a shared reward, as in research on cooperating agents, e.g. Foerster et al. 2016, Lowe et al. 2017). That gives
a real reason to talk: whoever stands where an apple lies can tell the hungry one. The teacher helps at first
(“Kogli is hungry. Say: there is an apple in the forest.” – a sentence Kogli already understands as a hint); whether
passing it on is worth it, they find out by themselves.

Honestly: NikolAI and Kogli share one brain – like twins who learn together, each with its own body and memory. That
halves nothing and is common in research (parameter sharing), but it also means Kogli can’t tell it isn’t NikolAI, so
only NikolAI is asked its name.

**Phase 8** tests it without a teacher: NikolAI stands where an apple lies, Kogli is hungry in the hut. Does Kogli get
the apple within 10 steps – once with them hearing each other, once with the line cut?

## Version 9: reading like a child

Until version 8 NikolAI read TinyStories – stories written by GPT. That is the path of large language models:
learning from what another model wrote. Now it reads the **BabyLM corpus** instead (Warstadt et al. 2023, the
BabyLM Challenge): about 100 million words – roughly what a child has heard by the age of twelve – all written or
spoken by people. Real parents talking to toddlers (CHILDES), everyday conversations (British National Corpus,
Switchboard), children’s books (Project Gutenberg, Children’s Book Test), film subtitles, educational videos and
Simple Wikipedia. The real conversations show it what talking looks like.

Its meaning still comes only from its world. A new reading brain replaces only hearing and speaking; body, senses
and actions stay (each reading brain has an id, so this happens exactly once per new reading brain). Instructions
and answers have to be relearned for a while after that.

```bash
mkdir -p ~/nikolai/books/babylm && cd ~/nikolai/books/babylm
for f in aochildes bnc_spoken cbt children_stories gutenberg open_subtitles qed simple_wikipedia switchboard wikipedia; do
  wget -c https://huggingface.co/datasets/cambridge-climb/BabyLM/resolve/main/clean/100M/$f.txt
done
sudo cp ~/nikolai/nikolai/*.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl start nikolai-reading
```

## Version 8: talking

Now and then the teacher asks NikolAI about itself: “Where are you?”, “How are you?”, “What do you have?”,
“What do you want?”, “What did you do?”, “Who are you?”. NikolAI learns to give the true answer (“i am in the
forest”, “i took a stone”) – from what it just heard and how it is doing in its world. It answers with its speaking
part, which started out as its reading brain.

On the dashboard you can talk to it (“Talk to it”). For these questions you get grounded answers – the page also
shows the true answer, so you can check. Anything else it answers in its storybook voice from reading: often off,
sometimes surprisingly fitting. Phase 7 tests the answers against guessing.

Honestly: this is a toddler with a picture-book voice, not a chatbot. It does not remember the conversation beyond
a few steps and only knows what is in its little world and its stories.

## Version 7: reading

A child hears millions of words before it speaks well. In its little world NikolAI only hears a few dozen. So now
it **reads**: TinyStories (Eldan & Li 2023), about two million short stories with the vocabulary of a 3-to-4-year-old.
It reads them word by word and keeps guessing the next word – a small language model (GRU, ~3 million parameters), the same
principle as large language models. Nobody tells it what words mean; it only learns in which contexts they appear.

That reading brain becomes its **hearing and speaking**. In the world it then learns what the words refer to.
While living it keeps reading a few lines, so it doesn’t forget its books.

The scientific question (Hill et al. 2020, *Human instruction-following with deep reinforcement learning via
transfer-learning from text*): does an agent that has read understand instructions with **words its teacher never
used** – “Grab a stone!” instead of “Take a stone!”? Phase 6b tests exactly that.

Honestly: reading gives it a feeling for which words belong together, not knowledge of the world. Its language
model is tiny – it will write simple, often odd little stories, not hold conversations.

## The world

Five places with real rules: hut, orchard (apple tree), forest (branches, bast), river (stones), workshop
(workbench). Hunger grows over time; an apple fills it. From a branch, a stone and bast you build a stone axe at the
workbench. Apples are scarce: after picking one, the next takes a while to grow – the tree alone is not enough.
Now and then a windfall apple lies somewhere else (forest, river, workshop or hut), each time in a different place.
NikolAI only sees it when it is there. The teacher knows where it is.

## The brain (PyTorch, about 9 million parameters – most of them for its 8,000 words)

- **Perception** (senses → numbers), **hearing** (words are just numbers at first), **memory** (GRU).
- **Acting:** chooses a verb (go, take, eat, put down, build, wait) and a thing or direction.
- **World model:** predicts what it will sense next. The prediction error is its **curiosity** – surprising
  things are interesting, familiar things get boring (Pathak et al. 2017, “Curiosity-driven Exploration”).
- **Describing:** learns to predict what the teacher will say about its action – and then says it itself.
  That is why it first speaks in the second person (“you go to the forest”): toddlers echo what they hear too.
- **Talking while acting:** a switch decides each step whether NikolAI also says something (up to five words).
  It learns this in two ways: **repeating** what the teacher says (“Say: I want an apple.”) and **reward** when
  what it says achieves something.
- **Learning:** PPO (Schulman et al. 2017) with rewards from the body (hunger), curiosity and the teacher’s praise.
- **Hindsight (version 6):** after every successful action, practising asks: which instruction would this have
  fulfilled? (“You took a stone” → “Take a stone!”). It replays the last few steps as if NikolAI had heard that
  sentence at the start, and learns: whoever hears this sentence does this. So every action becomes a lesson, not only
  the rare real instructions (Andrychowicz et al. 2017, *Hindsight Experience Replay*; Ghosh et al. 2019, *GCSL*;
  Chan et al. 2019, *ACTRCE*). Test-only instructions are never relabelled.

## The teacher

A simple program that talks about what NikolAI is really doing (“You take an apple.”), names things (“This is the
apple tree.”), gives instructions (“Go to the forest!”) and praises. It listens, too: if NikolAI asks for something it
really needs, it gets it – but receiving is never rewarded in itself, only what NikolAI achieves with it. (In version
3 NikolAI found a loophole: it kept asking for bast, received it with praise, put it down and asked again. In AI
research this is called *reward hacking*.)

Since version 6 the teacher also:
- tells NikolAI where the windfall apple is when it is hungry (“There is an apple at the river.”), or models the
  question (“Hungry? Say: where is an apple?”) – and answers when NikolAI asks “where … apple”. Listening and asking
  now really pay off: this is the idea behind signalling games (Lewis 1969) and grounded communication research
  (e.g. Lazaridou et al. 2017).
- gives instructions **against habit**: what NikolAI does on its own anyway is asked for less often, and nothing
  about apples while it is hungry. Otherwise it can earn praise without listening.

## Honest notes

- It practises in **32 copies of the world at once** – on a CPU it would otherwise take months. On the dashboard you
  see NikolAI live in its own world, in real time, with its current brain.
- The senses are given (like eyes and a stomach); everything else is learned.
- **Two instructions are never given in training** (“Put down the apple!”, “Take some bast!”), only in tests. If they
  work, NikolAI has really understood the words and combined them in a new way.
- Every test compares: with the sentence, without it (same situation) and acting at random. “Understood” means:
  with the sentence it works in at least 70 % and at least 30 points more often than without. A milestone only
  counts as stable once it holds in **three tests in a row**.
- Describing is still easy: the teacher’s sentence depends only on what just happened. Learning it fast does not
  mean NikolAI can tell stories.
- **Phase 5** tests whether it uses what it is told: hungry in the hut, the tree empty, one apple elsewhere, 4 steps.
  Two steps suffice if it understood; guessing finds it about one time in three. **Phase 5b**: does it ask “where …?”
  on its own when hungry (and less when full)?
- **Version 6** keeps the English brain and vocabulary of version 5. The world is harder now (fewer apples on the
  tree), so hunger may first rise a little.
- **Version 5: NikolAI now learns English.** A brain from an earlier (German) version keeps what it knows about
  the world – finding food, paths, collecting things – but starts with an empty vocabulary. Curves and milestones
  start anew; the dashboard shows how many steps it has spent learning English.

## Reading (version 7)

```bash
mkdir -p ~/nikolai/books
wget -c -O ~/nikolai/books/tinystories.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
sudo cp ~/nikolai/nikolai/nikolai-reading.service ~/nikolai/nikolai/nikolai-neuro.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl start nikolai-reading
journalctl -u nikolai-reading -f          # watch it read (a sample story every 2 minutes)
```

The reading service stops NikolAI, reads for 8 hours (all 7 free cores), saves `~/nikolai/neuro/reader_babylm.pt` and then
starts NikolAI again: it keeps its body, senses and actions, but hears and speaks with its reading brain – so
instructions have to be relearned at first. Don’t start NikolAI by hand while it is reading. Read more later:
start `nikolai-reading` again – it continues where it stopped (but only a brain that hasn’t moved in yet takes it over).

## Install (mini PC)

```bash
sudo apt install -y python3-venv
python3 -m venv ~/nikolai/neuro-venv
~/nikolai/neuro-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/Cognitronics/nikolai.git ~/nikolai/nikolai
sudo cp ~/nikolai/nikolai/nikolai-neuro.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now nikolai-neuro
```

Dashboard: `http://<mini-pc>:8091`. The brain is saved every 10 minutes and on stop (`~/nikolai/neuro/brain.pt`).
`--threads 6` leaves two cores for everything else; `Nice=10` gives other services priority.
`--public` makes the dashboard read-only (no talking to NikolAI) – for a public page.

## Upgrading a mini PC from version 11

Version 12 lives in a git clone of this repository, and the data files get English names. Once:

```bash
sudo systemctl stop nikolai-neuro
cp ~/nikolai/neuro/gehirn.pt ~/nikolai/neuro/gehirn_v11_backup.pt      # in case you want to go back
mv ~/nikolai/neuro/gehirn.pt ~/nikolai/neuro/brain.pt
mv ~/nikolai/neuro/leser_babylm.pt ~/nikolai/neuro/reader_babylm.pt
mv ~/nikolai/buecher ~/nikolai/books
git clone https://github.com/Cognitronics/nikolai.git ~/nikolai/nikolai
sudo rm -f /etc/systemd/system/nikolai-lesen.service
sudo cp ~/nikolai/nikolai/*.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl start nikolai-neuro
```

Later updates: `cd ~/nikolai/nikolai && git pull && sudo systemctl restart nikolai-neuro`.

## Files

| File | What it does |
|---|---|
| `neuro/world.py` | The world: places, things, hunger, recipe |
| `neuro/teacher.py` | The teacher: describes, names, instructs, listens |
| `neuro/brain.py` | The neural brain |
| `neuro/practice.py` | Practising: PPO, curiosity, repeating, describing |
| `neuro/evaluation.py` | Tests and milestones |
| `neuro/reading.py` | Reading: books, vocabulary, language model, pre-reading |
| `neuro/server.py` | Continuous training, live world, web server |
| `neuro/page.html` | The dashboard (Cognitronics design) |
| `neuro/names.py` | Display names of places and things |
| `neuro/compat.py` | Loads brains saved by older versions (German names) |

## Next steps

More properties so that words must be combined (“Take the red apple!”), a bigger world with real needs (winter, fire,
wood that needs the stone axe), a second agent to talk to, and later a language model as a chatty educator.

## Contributing

NikolAI is an open project and help is welcome – from machine learning and neuroscience to simulation, infrastructure,
visualisation and plain curiosity. Open an [issue](https://github.com/Cognitronics/nikolai/issues) with a question,
an idea or something you'd like to work on, send a pull request, or write to office@cognitronics.ai.

## Support

NikolAI is built and trained in spare time on a small home server. If you'd like to help cover hardware and
electricity, you can [support the project via PayPal](https://paypal.me/cognitronics) – thank you! ♥

## License

MIT – see [LICENSE](LICENSE).

