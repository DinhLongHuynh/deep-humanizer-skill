# deep-humanizer

An agent skill that edits LLM-written prose until it is closer to human writing at the levels readers and detectors actually measure, and then checks whether it got there.

Most "humanizers" swap words and strip em dashes. The differences between instruction-tuned LLM prose and human prose sit much deeper: in grammatical register, in how information is spread across tokens, in how paragraphs are built, and in where a document lands in embedding space. This skill gives an agent a research-based editing workflow for all of these layers, plus a local toolkit that measures each one against a calibrated human reference and refuses to call a text finished on a single score.

The skill follows the open [Agent Skills](https://agentskills.io) format (a folder with a `SKILL.md`), so it works in Claude Code, Claude.ai, Codex, Cursor, Gemini CLI, GitHub Copilot and other agents that read `SKILL.md`.

## What it does

The agent works in a loop: **diagnose → edit in passes → meaning gate → check round → held-out confirmation → report**.

| Layer | What differs between LLM and human prose | Measured by |
|---|---|---|
| L1 Surface | Over-used vocabulary (669 sourced items), stock constructions, dash habits, chat residue | `surface_scan.py` |
| L2 Syntax and register | Participial clauses, nominalisations, coordinated pairs, repeated POS templates, narrow sentence-length spread | `stylometry.py` |
| L3 Probability | Fast-DetectGPT curvature, Binoculars, surprisal spread and shape | `lm_signals.py` |
| L4 Embedding geometry | Intrinsic dimension, sentence trajectory, position in a reference bank, distance from the author's style | `embed_geometry.py` |
| Paragraph structure | One-finding sentences at the same pace, no writer stance, summary closers, the same arc in every answer | `structure_scan.py` |

Around these layers:

- **`check.py`**: runs every layer, places each feature on the human distribution, lists the sentences and paragraphs to fix with a tactic for each, runs held-out open detectors, and says whether to polish, restructure or stop. If a held-out detector disagrees with a calibrated CLEAN, the verdict becomes DETECTORS DISAGREE.
- **`meaning_guard.py`**: checks that no claim, number, citation, quote or name was lost or invented, with a claim-level mode for paragraph rewrites and a list of planned cuts.
- **`detectors.py`**: four open detectors run locally (tabularisai ModernBERT, desklib, fakespot, tropa-mini), plus adapters for eight commercial APIs (GPTZero, Pangram, Originality, Sapling, Winston, Hive, Copyleaks, ZeroGPT) that are used only when you set a key.
- **`batch_compare.py`**: tests a set of documents as a distribution (MMD, classifier two-sample test, per-feature KS).
- **`calibrate.py`**: builds a reference bank for your own register or your own writing.

The editing instructions themselves are in `SKILL.md` (Passes A0 to D) and `references/tactics.md` (20 tactics, each with evidence, risks and a way to confirm the effect).

## Repository layout

```
deep-humanizer-skill/
├── README.md
├── install.sh                      # copies or links the skill into agent folders
├── examples/ai_abstract_sample.md  # an LLM-written abstract for testing the install
└── skills/deep-humanizer/          # the skill itself (this folder is what agents load)
    ├── SKILL.md                    # workflow the agent follows
    ├── kernel.py                   # optional helper for Claude Science; other agents ignore it
    ├── scripts/                    # toolkit + requirements.txt
    ├── assets/                     # vocabulary list, detector API specs, default reference bank
    └── references/                 # research digest, signals, tactics, detectors, validation, bibliographies
```

## Installation

Installation has two parts: put the skill folder where your agent looks for skills, then install the Python toolkit that the skill calls.

### 1. Install the skill into your agent

The folder must keep the name `deep-humanizer`, with `SKILL.md` directly inside it (`.../skills/deep-humanizer/SKILL.md`, not `.../skills/skills/deep-humanizer/SKILL.md`).

**Any agent, with the skills CLI.** It detects the agents on your machine and links the skill into each one:

```bash
npx skills add YOUR_GITHUB_USER/deep-humanizer-skill --skill deep-humanizer -g
```

**With the GitHub CLI** (`gh skill`, recent versions):

```bash
gh skill install YOUR_GITHUB_USER/deep-humanizer-skill deep-humanizer --scope user
```

**With the install script** (after `git clone`):

```bash
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

Python 3.10 or newer (tested with 3.12). A GPU is used if present; CPU works.

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
- **Your own writing samples** (3 or more texts, ideally 1,500+ words, written without AI help). This is the strongest lever: the check round then measures voice match, and the agent copies your sentence habits.
- **Human controls**: 2-3 texts of the same register written before 2023, used to find out which detectors are informative for this kind of text.
- **Facts that must survive.**
- **Consent before any text is sent to a commercial detector API.**

The same steps by hand:

```bash
S=skills/deep-humanizer/scripts
python $S/check.py draft.md --md r0.md --json r0.json --controls controls/ --author samples/   # diagnose
# edit draft.md -> v1.md following SKILL.md, Passes A0-D
python $S/meaning_guard.py draft.md v1.md --nli                                                # meaning gate
python $S/check.py v1.md --baseline draft.md --md r1.md --controls controls/                    # check round
python $S/check.py v1.md --final --external gptzero                                            # held-out confirmation (needs GPTZERO_API_KEY)
python $S/batch_compare.py rewrites/ --against samples/                                        # several documents
```

Commercial adapters read their keys from environment variables (`GPTZERO_API_KEY`, `PANGRAM_API_KEY`, `ORIGINALITY_API_KEY`, `SAPLING_API_KEY`, `WINSTON_API_KEY`, `HIVE_API_KEY`, `COPYLEAKS_EMAIL` + `COPYLEAKS_API_KEY`, `ZEROGPT_API_KEY`). They follow each vendor's documentation as of November 2026 but have not been tested with live keys; inspect the raw output of the first call.

## How well it works

All numbers come from `skills/deep-humanizer/references/validation.md`.

- **Detection.** On a matched bank of 120 human texts written before ChatGPT (arXiv abstracts, CNN news, Reddit stories) and 120 texts from four 2025-26 models, the combined local model separates human from AI text with a grouped cross-validated AUROC of 0.978. Stylometry is the strongest local layer (0.947) and surface vocabulary the weakest (0.78).
- **Results that contradict 2023 findings.** Current models produce *higher* intrinsic dimension than human text (PHD 10.10 vs 9.70) and *more* varied vocabulary (MTLD 137 vs 98). The 2019 GPT-2 detector is at chance (AUROC 0.58).
- **Editing.** On 30 held-out AI texts, an LLM editor following the skill for three passes lowered the median combined probability from 0.966 to 0.706. Surface and probability layers moved most; syntax barely moved. All 30 kept their meaning, and 2 of 30 reached CLEAN.
- **A clean score is not the end.** In a later test, four short reflective texts scored CLEAN after sentence-level edits, while tropa-mini and ZeroGPT still called them AI. The remaining signal was in how the paragraphs were built. That case produced the paragraph-restructure pass, the structure scan and the rule that a disagreeing held-out detector overrides CLEAN.

In short: the check round is a reliable instrument, and the editing workflow moves text a real distance. The last part of the gap needs the author's own sentences, examples and opinions, and the skill says so rather than forcing it.

## Calibrating for your own writing

The default bank covers three registers and texts of 150-400 words. For anything else, build a bank from your own material:

```bash
# corpus.jsonl lines: {"id": "...", "text": "...", "label": "human" | "ai", "group": "topic-id", "register": "..."}
python skills/deep-humanizer/scripts/calibrate.py corpus.jsonl --out banks/my_register --cache cache/
python skills/deep-humanizer/scripts/check.py draft.md --reference banks/my_register
```

Use human texts written before 2023 (or your own early drafts) and LLM texts written to the same prompts, matched by `group`. About 100 documents per class gives stable quantiles.

## Responsible use

The skill improves prose; it does not obfuscate. It refuses character-level tricks (homoglyphs, zero-width characters, planted misspellings) and flags them when it finds them. It never adds facts, and it lists every statement about the author's own experience for the author to confirm. Many universities, journals and employers require disclosure of AI assistance. Editing a text so it reads well does not change what you have to disclose.

## Models and data

Models are downloaded from Hugging Face at first use, each under its own licence: Qwen/Qwen2.5-1.5B and -Instruct, FacebookAI/roberta-base, sentence-transformers/all-mpnet-base-v2, AnnaWegmann/Style-Embedding, cross-encoder/nli-deberta-v3-xsmall, tabularisai/ai-text-detection, desklib/ai-text-detector-v1.01, fakespot-ai/roberta-base-ai-text-detection-v1 and wasitaigeneratedcom/tropa-mini-ai-text-detector. Check the model cards before commercial use. The reference bank ships features and sentence embeddings only, not the source texts. `assets/ai_vocabulary.json` cites a source for every item.

## Background

The design rests on a literature review of embedding- and distribution-level detection, current detector tools, and humanization and evasion studies (Tulchinskii et al. 2023; Bao et al. 2024; Hans et al. 2024; Reinhart et al. 2025; Shaib et al. 2024; Kobak et al. 2025; Dugan et al. 2024, RAID; and others). The digest with verified arXiv identifiers is in `references/research_digest.md`; the full annotated bibliographies are in `references/literature/`. The skill was designed and validated with an AI agent (Claude, in Claude Science); `references/validation_code/` contains the experiment code.
