# deep-humanizer-skill

An agent skill that edits text written by an AI model (an LLM such as ChatGPT or Claude) until it reads like human writing.

Most humanizer tools only make surface edits: they swap words and remove em dashes. But AI text also differs from human text in less visible ways: how information is spread across sentences, how paragraphs are built, and how predictable each word is. This skill gives your agent an editing method for each of these levels, plus a toolkit that runs on your computer and measures each level against real human writing, so you can see what is still AI-like and what has been fixed.

It works in Claude Code, Claude.ai, Codex, Cursor, Gemini CLI, GitHub Copilot, and any other agent that reads `SKILL.md` files.

## How it works

When you ask your agent to humanize a text, it follows the instructions in `SKILL.md`, step by step:

1. **Diagnose.** Run the toolkit on your text. The report says how AI-like the text is and which sentences and paragraphs to fix first.
2. **Edit in passes.** Rewrite the text in several rounds, from deep changes (paragraph structure, sentence construction) to surface ones (word choice, punctuation). The agent edits the text; it does not regenerate it from scratch.
3. **Meaning check.** Compare the edit with the original to make sure no claim, number, citation, quote or name was lost or made up. An edit that fails this check is thrown out, however good it looks.
4. **Check round.** Run the toolkit again on the edited text to see what changed. Repeat steps 2-4 at most three times.
5. **Final confirmation.** Run every available AI detector as an independent check (see *held-out detectors* below), including commercial ones if you have keys and agree to send the text.
6. **Report.** Give you the final text and a before/after summary, including any remaining risk.

### What the toolkit measures

The toolkit looks at the text on five levels. Each level is measured by its own script in `skills/deep-humanizer/scripts/`.

| Level | What it looks at | Typical AI pattern | Script |
|---|---|---|---|
| L1: Surface | Words, phrases and punctuation | Over-used words ("delve", "tapestry"; 669 items from published studies), stock phrases, dashes everywhere, chatbot leftovers ("I hope this helps") | `surface_scan.py` |
| L2: Syntax and register | How sentences are built | Trailing "-ing" clauses (", highlighting..."), many "-tion" nouns, "X and Y" pairs, the same sentence pattern repeated, sentences all about the same length | `stylometry.py` |
| L3: Probability | How predictable each word is to a language model | The text uses the words a model finds most likely, so it is unusually predictable and evenly so (measured with Fast-DetectGPT, Binoculars and surprisal) | `lm_signals.py` |
| L4: Embedding geometry | Where the text sits when converted into numbers by a language model (an "embedding") | Sentences that all repeat one point; a text that sits closer to AI texts than to human texts | `embed_geometry.py` |
| Paragraph structure | How paragraphs are built | Paragraphs that list findings at the same pace with no opinion from the writer, paragraphs that end on a summary, every section following the same arc | `structure_scan.py` |

### The other scripts

- **`check.py`**: the main script. It runs all five levels, compares each result with the human reference texts, lists what to fix (with a suggested technique for each), runs the held-out detectors, and gives a verdict (see below).
- **`meaning_guard.py`**: the meaning check from step 3.
- **`detectors.py`**: AI detectors. Four free ones run on your computer (tabularisai ModernBERT, desklib, fakespot, tropa-mini). Eight commercial ones (GPTZero, Pangram, Originality, Sapling, Winston, Hive, Copyleaks, ZeroGPT) are used only if you add your own API key, and only with your consent.
- **`batch_compare.py`**: checks a set of texts together, for example ten blog posts. A set can share an AI pattern even when each text passes alone.
- **`build_profile.py`** and **`voice_profile.py`**: measure your own writing so the agent edits toward *your* voice. See [Tailor it to your writing style](#tailor-it-to-your-writing-style).
- **`make_corpus.py`** and **`calibrate.py`**: build your own reference bank (see below).

The editing instructions are in `SKILL.md` (Passes A0 to D) and `references/tactics.md` (20 techniques, each with the evidence for it, its risks, and how to confirm it worked).

### Words used in this README

| Term | Meaning |
|---|---|
| Register | The type of text and its tone: an abstract, a blog post, an email, a story. It decides what is normal. Contractions are wrong in an abstract but fine in a blog post |
| Reference bank | A set of human texts and AI texts that the toolkit compares your text with. The default bank, included in the skill, has 120 human texts written before ChatGPT (research abstracts, news articles, short stories) and 120 AI texts on the same topics |
| Human band | The range of values the human texts in the reference bank show for a feature. Inside the band means "normal for human writing" |
| Verdict | `check.py`'s overall answer. **CLEAN**: reads like the human texts. **BORDERLINE**: in between. **FLAGGED**: reads like the AI texts. **DETECTORS DISAGREE**: the toolkit says CLEAN or BORDERLINE, but an independent detector still thinks the text is AI |
| Held-out detectors | AI detectors the agent never tunes the wording against. Rewording until one detector is satisfied only teaches the text to fool that detector. Instead, they decide when to stop, and they point to paragraphs that still read as AI so the agent can rebuild them |
| Human controls | 2-3 texts of the same register that you know are human (written before 2023). If a detector calls them AI too, it is unreliable for this kind of text and is ignored |
| Voice profile | A folder that describes your own writing, so the agent edits toward your voice instead of toward an average human. See [Tailor it to your writing style](#tailor-it-to-your-writing-style) |

## Repository layout

```
deep-humanizer-skill/
├── README.md
├── .gitignore                      # keeps personal profiles, banks and caches out of git
├── install.sh                      # copies or links the skill into agent folders
├── examples/ai_abstract_sample.md  # an AI-written abstract for testing the install
└── skills/deep-humanizer/          # the skill itself (this folder is what agents load)
    ├── SKILL.md                    # the step-by-step instructions the agent follows
    ├── kernel.py                   # optional helper for Claude Science; other agents ignore it
    ├── scripts/                    # the toolkit, plus requirements.txt
    ├── assets/                     # word list, detector API details, default reference bank, voice template
    └── references/                 # research summary, measurements, techniques, detectors, validation, bibliographies
```

## Installation

Installation has two parts: put the skill folder where your agent looks for skills, then install the Python toolkit that the skill runs.

### 1. Install the skill into your agent

The folder must keep the name `deep-humanizer`, with `SKILL.md` directly inside it (`.../skills/deep-humanizer/SKILL.md`, not `.../skills/skills/deep-humanizer/SKILL.md`). Pick one of the methods below.

**Any agent, with the skills CLI.** It finds the agents on your machine and adds the skill to each one:

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
./install.sh --link agents  # link instead of copy, so a git pull updates the skill
```

**By hand.** Copy or link `skills/deep-humanizer` into one of these folders:

| Agent | For all your projects | For one project only |
|---|---|---|
| Claude Code | `~/.claude/skills/` | `.claude/skills/` |
| OpenAI Codex | `~/.agents/skills/` (older versions: `~/.codex/skills/`) | `.agents/skills/` |
| Cursor | `~/.cursor/skills/` or `~/.agents/skills/` | `.cursor/skills/` |
| Gemini CLI | `~/.gemini/skills/` or `~/.agents/skills/` | `.gemini/skills/` |
| GitHub Copilot (VS Code, CLI) | `~/.copilot/skills/` or `~/.agents/skills/` | `.github/skills/` |
| Other Agent Skills clients | `~/.agents/skills/` in most cases; check the client's documentation | |

**Claude.ai and Claude Desktop.** Zip the `skills/deep-humanizer` folder (the zip should contain the `deep-humanizer` folder at its root) and upload it under Settings > Skills. The editing instructions then work in chat. The toolkit only runs where the agent can run Python with the packages from step 2; without it, the agent edits by reading and tells you that no measurement was made.

**Agents without skill support.** Point the agent at the file, for example with this line in `AGENTS.md`: "For humanizing or AI-text checks, follow `skills/deep-humanizer/SKILL.md`."

Restart the agent or open a new session after installing. Then ask something like "does this read as AI?" to confirm the skill is picked up.

### 2. Install the Python toolkit

You need Python 3.10 or newer (tested with 3.12). A GPU is used if you have one; otherwise the CPU works.

```bash
python -m venv .venv && source .venv/bin/activate      # create and enter a Python environment (or use conda)
pip install -r skills/deep-humanizer/scripts/requirements.txt
python -m spacy download en_core_web_sm                # English grammar model used by the syntax level
export HF_HOME=~/hf_cache                              # where downloaded models are stored
```

Run `source .venv/bin/activate` again in each new terminal before using the scripts. Add the `export HF_HOME=...` line to your shell profile (`~/.zshrc` or `~/.bashrc`) so the models are downloaded only once.

The first run downloads about 10-12 GB of free models from Hugging Face: two small language models (Qwen2.5-1.5B base and instruct) for the probability level, roberta-base, all-mpnet-base-v2 and Style-Embedding for the embedding level and the meaning check, an NLI model that checks whether two sentences say the same thing, and the four free detectors. If the download stalls behind a proxy or in a sandbox, set `HF_HUB_DISABLE_XET=1`.

On a 10-core laptop CPU, a full check of a 300-word text takes about 30-60 seconds once the models are loaded. `--fast` runs only the surface, syntax and probability levels and takes about 10 seconds.

### 3. Check the install

```bash
python skills/deep-humanizer/scripts/check.py examples/ai_abstract_sample.md --fast
```

The sample is an abstract written by an AI model, so the report should end with the verdict FLAGGED (an AI probability above 0.9) and list what to fix: the features that look most AI-like, the sentences that need work first ("hotspot sentences"), and surface patterns such as over-used words.

## Using it

Ask your agent in plain language, for example "make this sound less like ChatGPT", "polish this so it sounds like me", "does this read as AI?", or "check these ten answers for a shared template". Before editing, the agent asks for what it cannot work out itself:

- **The register and the reader** (abstract, essay, email, story), because the register decides what is normal.
- **Your voice profile, or a few texts you wrote yourself** (3 or more, ideally 1,500+ words in total, written without AI help). This helps more than anything else: the agent copies your habits, and the check round measures how close the draft is to your voice. See [Tailor it to your writing style](#tailor-it-to-your-writing-style).
- **Human controls**: 2-3 texts of the same register that you know are human, written before 2023. They show which detectors can be trusted for this kind of text.
- **Facts that must stay**: numbers, names, citations, quotes.
- **Your consent before any text is sent to a commercial detector.**

### Running the steps yourself

You can run the same steps from the command line. `draft.md` is your original text, and `v1.md` is your edited version.

```bash
S=skills/deep-humanizer/scripts

# 1. Diagnose: writes the report to r0.md (readable) and r0.json (full data)
python $S/check.py draft.md --md r0.md --json r0.json --controls controls/ --profile blog

# 2. Edit draft.md into v1.md, following the passes in SKILL.md

# 3. Meaning check: lists anything lost or added between the two versions
python $S/meaning_guard.py draft.md v1.md --nli

# 4. Check round: --baseline adds a before/after table
python $S/check.py v1.md --baseline draft.md --md r1.md --controls controls/ --profile blog

# 5. Final confirmation: --final runs all free detectors; --external adds a commercial one (needs its API key)
python $S/check.py v1.md --final --external gptzero

# For a set of texts: compare a folder of edited texts with your own writing
python $S/batch_compare.py rewrites/ --against ~/.deep-humanizer/profiles/blog/samples/
```

`controls/` is a folder with your human control texts, and `--profile blog` uses your voice profile. Both are optional: leave them out if you don't have them.

**Commercial detectors.** Each one reads its API key from an environment variable. Set it before running, for example `export GPTZERO_API_KEY=...`. The variables are `GPTZERO_API_KEY`, `PANGRAM_API_KEY`, `ORIGINALITY_API_KEY`, `SAPLING_API_KEY`, `WINSTON_API_KEY`, `HIVE_API_KEY`, `COPYLEAKS_EMAIL` + `COPYLEAKS_API_KEY`, and `ZEROGPT_API_KEY`. `--external` sends the text to that company. The connections follow each company's documentation as of November 2026 but have not been tested with live keys, so check the raw output of your first call.

## Tailor it to your writing style

Out of the box, the skill edits toward the human texts in the default reference bank: research abstracts, news articles and short stories written before ChatGPT. Your writing may differ from them. Dense academic prose, a blog voice with dashes, or plain non-native English can all look "AI-like" by comparison. A **voice profile** tells the agent what your own writing looks like, so it edits toward your voice and leaves your real habits alone.

### In short

You need 3 or more texts you wrote yourself (1,500+ words in total) and the Python toolkit.

```bash
S=skills/deep-humanizer/scripts

# 1. Measure your writing (about 10 s; creates ~/.deep-humanizer/profiles/blog/)
python $S/build_profile.py ~/my_writing/blog/ --name blog --register "Blog posts: first person, short paragraphs"

# 2. Open ~/.deep-humanizer/profiles/blog/voice.md and fill in the sections at the top

# 3. Ask your agent: "Humanize this with my voice profile blog"
#    or check a draft yourself:
python $S/check.py draft.md --profile blog
```

The rest of this section explains each step.

### What a profile is

A profile is a folder, by default `~/.deep-humanizer/profiles/<name>/`, outside this repository. It holds three things:

| File | What it is | Who writes it |
|---|---|---|
| `voice.md` | Your style card: your rules, habits, phrases you avoid, words that are yours. The agent reads it in full before editing | You write the top sections; `build_profile.py` fills a "measured" block at the bottom |
| `profile.json` | Your usual range for each measured feature, such as sentence length, paragraph size, how often you write "I" or "we", contractions, dashes and "-ing" clauses. Your samples are cut into pieces of about 300 words, and the range is how much these pieces vary. 26 of the features are compared with every draft | `build_profile.py` |
| `samples/` | A copy of your texts, used to compare a draft's overall style with yours | `build_profile.py` |

Make one profile per register. Your papers, blog posts and emails have different habits, so make `papers`, `blog` and `email` profiles separately.

### Step 1: choose your samples

- **Written by you without AI help.** Texts from before 2023 are safest. A text polished by an AI model teaches the profile the model's habits.
- **One register.** Everything in the folder should be the kind of text the profile is for.
- **Prose only.** Plain text or Markdown, one text per file. Remove other people's quotations, code, tables, reference lists, signatures and boilerplate.
- **Enough text.** At least 3 files and 1,500 words in total. 5,000+ words gives more reliable ranges. A few long texts are fine.

**Where to put them.** Any folder you like; there is no fixed location. Make one folder per profile and put the files directly inside it, because subfolders are not read. Only `.txt` and `.md` files count, so save Word documents or PDFs as plain text first. For example:

```
~/my_writing/blog/
├── 2019-first-post.md
├── 2020-why-i-switched.txt
└── 2021-lessons.md
```

You pass this folder to `build_profile.py` in Step 2, and it copies the files into the profile's `samples/` folder. After that, the original folder is no longer needed (unless you use `--no-copy`).

### Step 2: build the profile

```bash
python $S/build_profile.py ~/my_writing/blog/ --name blog --register "Blog posts: first person, short paragraphs"
```

This creates the profile folder, measures your texts, copies them into `samples/`, and creates `voice.md` from the template in `assets/voice_template.md`. The new `voice.md` already lists, under "Words that are mine", the words from the skill's AI word list that you use yourself. Its measured block at the bottom summarises your habits, including the sentence openers AI models like ("Moreover", "Furthermore", ...) that never appear in your texts.

The two arguments after the folder:

- `--name blog`: the profile's name. You use it to tell the agent and the scripts which profile to use. Without it, the profile is called `default`.
- `--register "..."`: optional. One line, in your own words, describing this kind of writing: who reads it, whether you write "I" or "we", how formal it is. Examples: `"Blog posts: first person, short paragraphs"`, `"Journal papers: we, formal, no contractions"`, `"Work emails: short, friendly, to colleagues"`. It is not used in any measurement. It becomes the first line under "Registers I write in" in a new `voice.md`, where the agent reads it. If you leave it out, write that line yourself in Step 3.

Other options:

- `--out DIR`: put the profile somewhere else.
- `--no-copy`: leave your texts where they are instead of copying them.
- `--lm`: also measure the probability level (slower).
- `--force`: replace the texts of an existing profile with a different set. Without it, the script refuses to mix two sets.

You can run it again whenever you add texts. It rewrites `profile.json` and only the measured block of `voice.md`, so your own edits stay. If your files separate paragraphs with single line breaks instead of blank lines (common when pasting from web pages), it treats each line as a paragraph and tells you.

### Step 3: fill in `voice.md`

Edit the sections above the measured block. Leave out anything you have nothing to say about.

| Section | What to write |
|---|---|
| Registers I write in | One line per register: reader, "I" or "we", formality, typical length |
| Habits to keep | Things you do on purpose ("I open with a question", "colons rather than dashes") |
| Rules that override the generic skill | Where you differ from the skill's default advice ("I use em dashes", "British spelling", "no contractions in papers") |
| Phrases I avoid | One phrase per bullet. The check round reports each one it finds in a draft |
| Words that are mine | Words from the AI word list that you really use. The check round marks them "yours, keep" and the agent keeps them. Delete any pre-filled word you don't want |
| Example sentences in my voice | 3-5 sentences you wrote and like. The agent copies their rhythm and tone, never their content |

Two rules for this file:

- Keep the headings "Phrases I avoid" and "Words that are mine" exactly as written, because the scripts read the bullets under them. Bullets that start with "(" are the template's placeholders and are ignored.
- Your rules override the skill's general style advice (dashes, contractions, vocabulary, sentence length, lists of three). They never override the rules against made-up facts, lost claims or character tricks.

### Step 4: use the profile

**With an agent.** Name the profile in your request: "Humanize this with my voice profile `blog`." A name on its own is looked up in `~/.deep-humanizer/profiles/`. A path to a profile folder or to a `voice.md` also works.

**As a default.** Add `export DEEP_HUMANIZER_PROFILE=blog` to your shell profile. `check.py` and `voice_profile.py` then use it without `--profile`, and the report names the profile it used. If the name is wrong, the scripts print a warning and run without a profile.

**From the command line.**

```bash
python $S/voice_profile.py draft.md --profile blog   # quick voice-only check (surface and syntax; a few seconds)
python $S/check.py draft.md --profile blog           # full check round plus a voice section
```

**Other locations.** `DEEP_HUMANIZER_HOME=/path` moves the main folder from `~/.deep-humanizer`. Claude.ai and other sandboxed agents cannot read your home folder, so put the profile inside the skill folder before zipping it (for example `skills/deep-humanizer/profiles/blog/`) and name that path in your request. `profiles/` is ignored by git, so it will not be committed.

### What the profile adds to the report

With a profile, the check report gets a "Voice match" section, and the list of things to fix is split in two:

- **Voice gaps**: habits where the draft is outside your usual range, with your typical value (the median), your usual range, and what to do ("Vary sentence length the way you do"). The agent edits toward your typical value.
- **Generic targets that match your own style**: features that look AI-like compared with the reference bank, but that your own texts show too. The agent no longer treats them as things to fix. If a detector still flags the text, the agent tells you instead of quietly removing your style.
- **Your phrase lists**: phrases you avoid that appear in the draft, and words from the AI word list split into "yours, keep" and "not yours".
- **Spelling, openers and dashes**: American spelling in a British writer's draft (or the reverse), AI-favourite sentence openers that never appear in your texts (needs 40+ sentences of samples), and em dashes where you use spaced en dashes or hyphens.
- **Overall style** (full `check.py` only, not `--fast` or `voice_profile.py`): two measures of how close the draft is to your texts as a whole, compared with how close your texts are to each other. One uses a style model; the other compares how often you use small words such as "the", "of" and "but" (needs about 1,500 words of samples).

What a profile does **not** change: the verdict, the held-out detectors and the meaning check. It changes what the agent edits, not how the draft is scored. To change the scoring, build a personal reference bank (below).

A profile with only a hand-written `voice.md` (no samples, no `build_profile.py`) also works: the agent follows your rules and the report checks your two phrase lists, but there are no voice gaps or style comparisons. To start one, copy `assets/voice_template.md` to `~/.deep-humanizer/profiles/<name>/voice.md`.

**How well it works.** We tested it on six of the author's blog posts: each time we built a profile from five posts and checked the sixth. The checked posts had 0.67 voice gaps on average (four of six had none, the most was three). Ten texts from four AI models, checked against the same author's profile, had 2.8 gaps on average (between 1 and 7, none had zero), mostly "X and Y" pairs, trailing "-ing" clauses, AI-list words and long sentences. One reflective post, checked against a profile of technical posts, showed three gaps (more "we", hedges and exclamations): a difference in the kind of writing, not in who wrote it. It is a small test and the two groups overlap, so use the profile to guide edits, not to decide whether a text was written by AI.

### Optional: a personal reference bank

A voice profile guides the edits, but the verdict is still based on the default reference bank. A personal reference bank replaces the human texts in that bank with yours, so the verdict itself answers "how far is this from my writing?" instead of "how far is this from an average human's?". It needs more material (30-100+ of your texts) and the full toolkit, and building it takes a few hours on a laptop CPU.

1. Put your texts in one folder, one per file (`my_writing/`).
2. For each text, ask an AI model for a text on the same topic and of the same length, for example "Write a blog post of about 800 words on: <title or one-line summary>". Use the model you usually draft with. Do not ask it to imitate your style, because the bank needs typical AI writing to compare against. Save each result in a second folder, `ai_versions/`, under the same file name as your matching text (`post1.md`, or `post1_ai.md`).
3. Build the bank, then check drafts against it:

```bash
# Combine both folders into one file of labelled texts
python $S/make_corpus.py --human my_writing/ --ai ai_versions/ --register blog --out my_corpus.jsonl
# Measure every text and build the bank (slow; --cache lets you resume if it stops)
python $S/calibrate.py my_corpus.jsonl --out ~/.deep-humanizer/banks/blog --cache ~/.deep-humanizer/cache/blog
# Check a draft against your bank instead of the default one
python $S/check.py draft.md --reference ~/.deep-humanizer/banks/blog --profile blog
```

`make_corpus.py` pairs your texts with the AI texts by file name, so that a text and its AI twin are always kept together when the bank is tested. It cuts long texts into pieces of roughly 150-450 words, the length of the texts in the default bank. It warns if either folder gives fewer than 30 pieces; about 100 each gives reliable results. `calibrate.py` measures every piece on every level (about 30-60 seconds each on a laptop CPU). It refuses to overwrite the default bank that ships with the skill unless you pass `--overwrite-default`.

The same steps build a bank for a kind of writing the default bank does not cover, such as legal texts or product reviews: use texts of that kind written before 2023 instead of your own.

### Privacy

Profiles, banks and caches contain your writing. By default they are stored in `~/.deep-humanizer/`, outside this repository. The repository's `.gitignore` also excludes `profiles/`, `banks/` and calibration caches, and `build_profile.py` warns you if you put a profile inside a git repository. Everything in this section runs on your computer. `--external` sends only the draft being checked to the detector you name, never your samples or profile.

## Responsible use

The skill improves writing; it does not disguise it. It refuses character-level tricks (look-alike letters from other alphabets, invisible characters, deliberate misspellings) and flags them when it finds them. It never adds facts, and it lists every statement about the author's own experience for the author to confirm. Many universities, journals and employers require you to disclose AI assistance. Editing a text so it reads well does not change what you have to disclose.

## References

The design is based on a review of research on AI-text detection and humanization (Tulchinskii et al. 2023; Bao et al. 2024; Hans et al. 2024; Reinhart et al. 2025; Shaib et al. 2024; Kobak et al. 2025; Dugan et al. 2024, RAID; and others). A summary with verified arXiv identifiers is in `references/research_digest.md`, and the full annotated bibliographies are in `references/literature/`. The skill was designed and tested with an AI agent (Claude, in Claude Science); `references/validation_code/` contains the experiment code.
