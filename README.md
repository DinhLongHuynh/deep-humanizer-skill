# deep-humanizer-skill

This is an agent skill that edits LLM-generated text until it reads close to human writing. It works across several layers, from rewriting the prose to checking where the text falls in an embedding distribution (see details below).

Most humanizer tools and skills stop at surface edits, such as swapping words and stripping em dashes. The differences between instruction-tuned LLM prose and human prose sit deeper, beneath the grammar: in how information is distributed across a text, how paragraphs are built, and where a document lands in embedding space.

This skill gives an agent an editing workflow for each of these layers, plus a local toolkit that measures every layer against a calibrated human reference.

The skill works in Claude Code, Claude.ai, Codex, Cursor, Gemini CLI, GitHub Copilot and any other agent that reads SKILL.md.

## How this skill operate?

The SKILL instructs agents to work in a loop: **diagnose → edit in passes → meaning gate → check round → held-out confirmation → report**.

| Layer | What differs between LLM and human prose | Measured by |
|---|---|---|
| L1: Surface | Over-used vocabulary (669 sourced items), stock constructions, dash habits, chat residue | `surface_scan.py` |
| L2: Syntax and register | Participial clauses, nominalisations, coordinated pairs, repeated POS templates, narrow sentence-length spread | `stylometry.py` |
| L3: Probability | Fast-DetectGPT curvature, Binoculars, surprisal spread and shape | `lm_signals.py` |
| L4: Embedding geometry | Intrinsic dimension, sentence trajectory, position in a reference bank, distance from the author's style | `embed_geometry.py` |
| Paragraph structure | One-finding sentences at the same pace, no writer stance, summary closers, the same arc in every answer | `structure_scan.py` |

Around these layers:

- **`check.py`**: runs every layer, places each feature on the human distribution, lists the sentences and paragraphs to fix with a tactic for each, runs held-out open detectors, and says whether to polish, restructure or stop. If a held-out detector disagrees with a calibrated CLEAN, the verdict becomes DETECTORS DISAGREE.
- **`meaning_guard.py`**: checks that no claim, number, citation, quote or name was lost or invented, with a claim-level mode for paragraph rewrites and a list of planned cuts.
- **`detectors.py`**: four open detectors run locally (tabularisai ModernBERT, desklib, fakespot, tropa-mini), plus adapters for eight commercial APIs (GPTZero, Pangram, Originality, Sapling, Winston, Hive, Copyleaks, ZeroGPT) that are used only when you set a key.
- **`batch_compare.py`**: tests a set of documents as a distribution (MMD, classifier two-sample test, per-feature KS).
- **`calibrate.py`**: builds a reference bank for your own register or your own writing.
- **`build_profile.py`** and **`voice_profile.py`**: measure your own writing and compare every draft with it, so the agent edits toward your voice rather than toward an average human (see [Tailor it to your writing style](#tailor-it-to-your-writing-style)).

The editing instructions themselves are in `SKILL.md` (Passes A0 to D) and `references/tactics.md` (20 tactics, each with evidence, risks and a way to confirm the effect).

## Repository layout

```
deep-humanizer-skill/
├── README.md
├── .gitignore                      # keeps personal profiles, banks and caches out of git
├── install.sh                      # copies or links the skill into agent folders
├── examples/ai_abstract_sample.md  # an LLM-written abstract for testing the install
└── skills/deep-humanizer/          # the skill itself (this folder is what agents load)
    ├── SKILL.md                    # workflow the agent follows
    ├── kernel.py                   # optional helper for Claude Science; other agents ignore it
    ├── scripts/                    # toolkit + requirements.txt
    ├── assets/                     # vocabulary list, detector API specs, default reference bank, voice template
    └── references/                 # research digest, signals, tactics, detectors, validation, bibliographies
```

## Installation

Installation has two parts: put the skill folder where your agent looks for skills, then install the Python toolkit that the skill calls.

### 1. Install the skill into your agent

The folder must keep the name `deep-humanizer`, with `SKILL.md` directly inside it (`.../skills/deep-humanizer/SKILL.md`, not `.../skills/skills/deep-humanizer/SKILL.md`).

**Any agent, with the skills CLI.** It detects the agents on your machine and links the skill into each one:

```bash
npx skills add DinhLongHuynh/deep-humanizer-skill --skill deep-humanizer -g
```

**With the GitHub CLI** (`gh skill`, recent versions):

```bash
gh skill install DinhLongHuynh/deep-humanizer-skill deep-humanizer --scope user
```

**With the install script:**

```bash
git clone https://github.com/DinhLongHuynh/deep-humanizer-skill.git
cd deep-humanizer-skill
./install.sh                # Claude Code (~/.claude/skills) and the shared ~/.agents/skills folder
./install.sh claude         # Claude Code only
./install.sh agents         # ~/.agents/skills (Codex, Cursor, Gemini CLI, GitHub Copilot and others)
./install.sh --link agents  # symlink instead of copy, so a git pull updates the skill
```

**By hand.** Copy or symlink `skills/deep-humanizer` into one of these folders:

| Agent | Personal (all projects) | Project only |
|---|---|---|
| Claude Code | `~/.claude/skills/` | `.claude/skills/` |
| OpenAI Codex | `~/.agents/skills/` (older versions: `~/.codex/skills/`) | `.agents/skills/` |
| Cursor | `~/.cursor/skills/` or `~/.agents/skills/` | `.cursor/skills/` |
| Gemini CLI | `~/.gemini/skills/` or `~/.agents/skills/` | `.gemini/skills/` |
| GitHub Copilot (VS Code, CLI) | `~/.copilot/skills/` or `~/.agents/skills/` | `.github/skills/` |
| Other Agent Skills clients | `~/.agents/skills/` in most cases; check the client's documentation | |

**Claude.ai and Claude Desktop.** Zip the `skills/deep-humanizer` folder (the zip should contain the `deep-humanizer` folder at its root) and upload it in the Skills section of Settings. The editing workflow then works in chat. The toolkit needs a Python environment with the packages below, so the check round only runs where the agent can execute code with those packages.

**Agents without skill support.** Point the agent at the file, for example with a line in `AGENTS.md`: "For humanizing or AI-text checks, follow `skills/deep-humanizer/SKILL.md`."

Restart the agent or open a new session after installing, then ask something like "does this read as AI?" to confirm the skill triggers.

### 2. Install the Python toolkit

Python 3.10 or newer (tested with 3.12). A GPU is used if present, otherwise, CPU works.

```bash
python -m venv .venv && source .venv/bin/activate      # or a conda environment
pip install -r skills/deep-humanizer/scripts/requirements.txt
python -m spacy download en_core_web_sm
export HF_HOME=~/hf_cache        # keep the model cache in one place so models download once
```

The first run downloads about 10-12 GB of open models from Hugging Face: Qwen2.5-1.5B base and instruct (probability layer), roberta-base (intrinsic dimension), all-mpnet-base-v2 (sentence embeddings, meaning gate), Style-Embedding (author voice), an NLI cross-encoder, and the four open detectors. If downloads stall behind a proxy or in a sandbox, set `HF_HUB_DISABLE_XET=1`.

On a 10-core laptop CPU, a full check of a 300-word text takes about 30-60 s after the models are loaded; `--fast` (surface, syntax and probability without Binoculars) takes about 10 s.

### 3. Check the install

```bash
python skills/deep-humanizer/scripts/check.py examples/ai_abstract_sample.md --fast
```

The sample is an abstract written by an LLM, so the report should end in FLAGGED (combined probability above 0.9 on the default bank) and list targets, hotspot sentences and surface tells.

## Using it

Ask your agent in plain language: "make this sound less like ChatGPT", "polish this so it sounds like me", "does this read as AI?", or "check these ten answers for a shared template". The agent follows `SKILL.md`. It will ask for:

- **The register and reader** (abstract, essay, email, fiction), because the register decides what is normal.
- **Your voice profile, or your own writing samples** (3 or more texts, ideally 1,500+ words, written without AI help). This is the strongest lever: the check round then measures voice match, and the agent copies your sentence habits. See [Tailor it to your writing style](#tailor-it-to-your-writing-style).
- **Human controls**: 2-3 texts of the same register written before 2023, used to find out which detectors are informative for this kind of text.
- **Facts that must survive.**
- **Consent before any text is sent to a commercial detector API.**

The same steps by hand:

```bash
S=skills/deep-humanizer/scripts
python $S/check.py draft.md --md r0.md --json r0.json --controls controls/ --profile blog     # diagnose
# edit draft.md -> v1.md following SKILL.md, Passes A0-D
python $S/meaning_guard.py draft.md v1.md --nli                                                # meaning gate
python $S/check.py v1.md --baseline draft.md --md r1.md --controls controls/ --profile blog    # check round
python $S/check.py v1.md --final --external gptzero                                            # held-out confirmation (needs GPTZERO_API_KEY)
python $S/batch_compare.py rewrites/ --against ~/.deep-humanizer/profiles/blog/samples/      # several documents
```

Commercial adapters read their keys from environment variables (`GPTZERO_API_KEY`, `PANGRAM_API_KEY`, `ORIGINALITY_API_KEY`, `SAPLING_API_KEY`, `WINSTON_API_KEY`, `HIVE_API_KEY`, `COPYLEAKS_EMAIL` + `COPYLEAKS_API_KEY`, `ZEROGPT_API_KEY`). They follow each vendor's documentation as of November 2026 but have not been tested with live keys; inspect the raw output of the first call.

## Tailor it to your writing style

Out of the box, the skill edits toward a *generic* human band: 120 texts written before ChatGPT, in three registers. Your own writing may sit somewhere else. Dense academic prose, a blog voice with dashes and short paragraphs, or plain non-native English can all look "AI-like" against a generic reference and to detectors. Tailoring tells the agent which band is yours, so it edits toward your voice and stops "fixing" habits that are genuinely yours.

There are three levels, and each builds on the one before:

| Level | What you provide | What it changes | Requirements |
|---|---|---|---|
| 1. Voice card | A `voice.md` you write by hand | The agent follows your rules; the check round flags your banned phrases and stops flagging AI-list words you really use | Nothing for the agent; the toolkit for the check round |
| 2. Measured profile | 3 or more samples of your writing (1,500+ words) | Your own range for 26 habits, "voice gaps" in every report, spelling convention, sentence openers, style-embedding match | The toolkit (spaCy only for the build; about 10 s) |
| 3. Personal reference bank | 30-100+ of your texts and LLM texts on the same topics | The calibrated verdict itself is measured against your writing instead of the generic bank | The full toolkit; a few hours on a laptop CPU |

What a profile never changes: the calibrated verdict (unless you build a Level 3 bank), the held-out detector gate and the meaning gate. Rules in `voice.md` override the skill's generic style advice (dashes, contractions, vocabulary, sentence length, three-item lists). They never override the rules against invented facts, lost claims or character tricks.

### Choosing your samples

- **Written by you without AI help.** Texts from before 2023 are safest, or drafts you know are yours. A sample polished by an LLM teaches the profile the LLM's habits.
- **One register per profile.** Your papers, blog posts and emails have different bands; build `papers`, `blog` and `email` separately. In our test (below), the author's one reflective post, checked against a profile of their technical posts, showed three voice gaps (more "we", hedges and exclamations): real differences of register, not of authorship.
- **Prose only.** Plain text or Markdown, one text per file. Remove quotations from other people, code, tables, reference lists, signatures and boilerplate.
- **Enough text.** At least 3 files and 1,500 words; 5,000+ words gives tighter ranges. Long texts are cut into chunks of about 300 words, so a few long texts are fine.

### Level 1: write a voice card

Copy the template into a profile folder and fill it in:

```bash
mkdir -p ~/.deep-humanizer/profiles/blog
cp skills/deep-humanizer/assets/voice_template.md ~/.deep-humanizer/profiles/blog/voice.md
```

| Section | What to write |
|---|---|
| Registers I write in | One line per register: reader, "I" or "we", formality, typical length |
| Habits to keep | Things you do on purpose ("I open with a question", "colons rather than dashes") |
| Rules that override the generic skill | Where you differ from the skill's defaults ("I use em dashes", "British spelling", "no contractions in papers") |
| Phrases I avoid | One phrase per bullet. The check round flags each one in a draft |
| Words that are mine | AI-list words you genuinely use. The check round stops flagging them |
| Example sentences in my voice | 3-5 sentences you wrote and like. The agent copies their rhythm and stance, never their content |

Keep the two headings "Phrases I avoid" and "Words that are mine" exactly as they are; the scripts read their bullet lists. Bullets that start with "(" are placeholders and are ignored.

### Level 2: measure your writing

```bash
S=skills/deep-humanizer/scripts
python $S/build_profile.py ~/my_writing/blog/ --name blog --register "Blog posts: first person, short paragraphs"
```

This writes `~/.deep-humanizer/profiles/blog/` with three things:

- **`profile.json`**: your range for each surface and syntax feature, measured on about 300-word chunks of your samples, plus countable habits (spelling, dash glyphs, "I" and "we", sentence openers, paragraph size, function words).
- **`voice.md`**: created from the template on the first run, with the AI-list words you use pre-filled under "Words that are mine". The bottom block summarises your measured habits, including the LLM-favourite openers ("Moreover", "Furthermore", ...) that never appear in your samples. Edit everything above that block.
- **`samples/`**: a copy of your samples, used for the style-embedding match. Pass `--no-copy` to keep them where they are.

Re-running `build_profile.py` replaces only the measured block in `voice.md`, so your edits are kept. It refuses to mix a different set of samples into an existing profile unless you pass `--force`. `--lm` also measures the probability layer on your samples (slower). If your files separate paragraphs with single line breaks (common when pasting from web pages), the builder treats each sentence-final line as a paragraph and says so; hard-wrapped Markdown is left alone.

Then check drafts against it:

```bash
python $S/voice_profile.py draft.md --profile blog   # voice-only check: surface and syntax, no large models
python $S/check.py draft.md --profile blog           # full check round plus the voice section
```

With a profile, the report adds:

- **Voice gaps**: habits where the draft is outside your own range, with your median and usual range, and what to do ("Vary sentence length the way you do").
- **Generic targets that match your own style**: features that look AI-like against the generic band, but where your own samples sit too. These are no longer edit targets. If a detector still flags them, the agent tells you instead of quietly erasing your style.
- **Your phrase lists**: phrases you avoid that appear in the draft, and AI-list words split into "yours, keep" and "not yours".
- **Spelling, openers, dashes**: American forms in a British writer's draft (or the reverse), sentence openers that never appear in your samples, and em dashes where you use spaced en dashes.
- **Style embedding and function words**: how close the draft's overall voice is to your samples, compared with how close your samples are to each other.

We tested the comparison on six of the author's own blog posts, building a profile from five and checking the sixth each time. The held-out posts averaged 0.67 voice gaps (four of six had none; at most three). Ten texts from four LLMs, checked against the full profile, averaged 2.8 (range 1-7; none had zero), mostly "X and Y" pairs, trailing "-ing" clauses, AI-list words and long sentences. It is a small test, and the two groups overlap: the profile guides edits, it is not a detector.

### Telling your agent to use your profile

- **In the request**: "Humanize this with my voice profile `blog`." A bare name is looked up in `~/.deep-humanizer/profiles/`; a path to any folder or to a `voice.md` also works.
- **As a default**: `export DEEP_HUMANIZER_PROFILE=blog` in your shell profile. `check.py` and `voice_profile.py` then use it without `--profile`, and the report names the profile it used. A wrong value only prints a warning.
- **Different location**: `DEEP_HUMANIZER_HOME=/path` moves the root from `~/.deep-humanizer`.
- **Claude.ai and other sandboxed agents** that cannot read your home folder: put the profile inside the skill folder before zipping it (for example `skills/deep-humanizer/profiles/blog/voice.md`) and name that path in your request. `profiles/` is git-ignored, so it will not be committed.

### Level 3: calibrate on your own writing

A personal bank replaces the generic human class with your texts, so the verdict itself measures "how far from me" instead of "how far from an average human".

1. Put your texts in one folder, one per file (`my_writing/`).
2. For each text, ask an LLM for a text on the same topic and of the same length, for example "Write a blog post of about 800 words on: <title or one-line summary>". Use the model you usually draft with, do not ask it to imitate your style (that blurs the AI class), and save the result under the same file name in `ai_versions/` (`post1.md` or `post1_ai.md`).
3. Build the corpus and the bank:

```bash
python $S/make_corpus.py --human my_writing/ --ai ai_versions/ --register blog --out my_corpus.jsonl
python $S/calibrate.py my_corpus.jsonl --out ~/.deep-humanizer/banks/blog --cache ~/.deep-humanizer/cache/blog
python $S/check.py draft.md --reference ~/.deep-humanizer/banks/blog --profile blog
```

`make_corpus.py` pairs files by name, so a topic never lands in both training and test folds, and cuts long texts into 150-450-word chunks (the range of the default bank). It warns when a class has fewer than 30 documents; about 100 per class gives stable results. Calibration runs every layer on every document (about 30-60 s each on a laptop CPU); `--cache` lets an interrupted run resume. `calibrate.py` refuses to write into the shipped default bank unless you pass `--overwrite-default`.

The same steps build a bank for a register the default bank does not cover: use texts of that register written before 2023 as the human class.

### Privacy

Profiles, banks and caches contain your writing. By default they live in `~/.deep-humanizer/`, outside this repository; the repository's `.gitignore` also excludes `profiles/`, `banks/` and calibration caches, and `build_profile.py` warns if you put a profile inside a git repository. Everything above runs locally. `--external` sends only the draft being checked to the detector you name, never your samples or profile.

## Responsible use

The skill improves prose; it does not obfuscate. It refuses character-level tricks (homoglyphs, zero-width characters, planted misspellings) and flags them when it finds them. It never adds facts, and it lists every statement about the author's own experience for the author to confirm. Many universities, journals and employers require disclosure of AI assistance. Editing a text so it reads well does not change what you have to disclose.

## References

The design rests on a literature review of embedding- and distribution-level detection, current detector tools, and humanization and evasion studies (Tulchinskii et al. 2023; Bao et al. 2024; Hans et al. 2024; Reinhart et al. 2025; Shaib et al. 2024; Kobak et al. 2025; Dugan et al. 2024, RAID; and others). The digest with verified arXiv identifiers is in `references/research_digest.md`; the full annotated bibliographies are in `references/literature/`. The skill was designed and validated with an AI agent (Claude, in Claude Science); `references/validation_code/` contains the experiment code.
