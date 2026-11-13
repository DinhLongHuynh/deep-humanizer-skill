---
name: deep-humanizer
description: Rewrite LLM-generated or LLM-assisted prose so it is closer to human writing at every level that readers and detectors measure - surface tells, syntax and register (Biber features), information flow (token surprisal), and embedding-space geometry - then verify the result in a check round that places the text on a calibrated human reference distribution, runs held-out open-source detectors (and commercial APIs if the user has keys), and gates meaning preservation. Use this skill whenever the user wants to humanize or de-AI a text, make a draft sound like them, find out why a text reads or gets flagged as AI-written, compare a draft with their own writing samples, build or use a personal voice profile, check a batch of texts for AI-like distribution, or understand AI-text detection signals - even if they only say "make this sound less like ChatGPT", "does this read as AI?", or "polish this so it sounds like me".
---

# Deep humanizer

Shallow humanizers swap words and strip em dashes. Detectors and attentive readers mostly do not rely on those words. The differences between instruction-tuned LLM prose and human prose sit in four layers, and this skill works on all four, then measures whether the rewrite actually moved:

| Layer | What differs (LLM vs human) | Script |
|---|---|---|
| L1 Surface | Excess vocabulary (delve, intricate, meticulous...), stock constructions ("not X but Y"), dashes, chat residue, formatting | `surface_scan.py` |
| L2 Syntax / register | Present participial clauses up to 5x, nominalisations about 2x, phrasal coordination about 2x, fewer agentless passives; repeated POS templates; narrow sentence-length distribution | `stylometry.py` |
| L3 Probability | Tokens likelier than sampled text would be (Fast-DetectGPT curvature, Binoculars), even surprisal, few low-rank tokens | `lm_signals.py` |
| L4 Embedding geometry | Intrinsic dimension of the token-embedding cloud outside the human band (lower for 2023 models; higher for the 2025-26 models in our calibration), semantically uniform sentence trajectory, position in the AI region of a reference bank, distance from the author's style | `embed_geometry.py` |

`check.py` runs all layers, locates every feature on a calibrated human distribution, lists the sentences to fix and what to do, runs held-out detectors (including tropa-mini in every round), scans paragraph-level structure (`structure_scan.py`) and says whether to stop or restructure. `meaning_guard.py` checks that nothing was lost or invented, with a claim-level mode for paragraph rewrites. `batch_compare.py` tests a set of documents as a distribution.

The evidence behind the design is in `references/research_digest.md`. Seven findings shape the workflow:

1. Word-level edits move fragile detectors and leave robust ones in place. Synonym swaps lowered Binoculars detection by 36 points on RAID but RADAR by only 3. Intrinsic dimension barely moved under DIPPER paraphrase (7.30 to 7.51). POS templates survive any synonym swap.
2. An LLM prompted to "sound human" keeps its register fingerprint. Biber-feature differences persist under style prompting, and a plain LLM paraphrase can make text *more* detectable: its perplexity drops further below the human level.
3. The gap is in dispersion as much as in means. Human text varies more in surprisal, sentence length and feature rates. Aim for the human *band* of the register, not an extreme: an RL humanizer that pushed perplexity to 300-400 (human about 50-75) became non-human the other way.
4. Steering and confirming with the same detector measures overfitting, not humanness (Goodhart). Detector-guided paraphrase cut TPR by about 88 percent while the text did not become more human.
5. All 11 audited commercial humanizers changed or dropped numbers, names, dates or quotes in 35-65 percent of the documents that contained them. Meaning is a hard gate.
6. Single-document passes do not guarantee batch-level indistinguishability; detection power grows with the number of documents.
7. A calibrated CLEAN is not a stop signal on its own. In a November 2026 session, four course reflections scored CLEAN (combined p 0.003-0.013) after sentence-level edits (28 of 66 sentences changed), three open detectors gave about 0.001, and tropa-mini (0.97-1.0) and ZeroGPT (100 %) still called them AI. The remaining signal was paragraph construction: runs of one-finding sentences at the same pace, no writer stance until the end, the same arc in every answer. A paragraph-level rewrite moved tropa-mini on one answer from 0.97 to 0.60 and left two others at about 0.99, where only the author's own sentences can help. Hence: held-out detectors run every round, disagreement overrides CLEAN, and Pass A0 restructures paragraphs.

## Scope

The skill improves prose: the rewrite should read like a careful human writer of the register. It does not do character-level obfuscation (homoglyphs, zero-width characters, injected misspellings, deleted articles). These game specific detectors, make the text worse, and are flagged by `surface_scan.py` as integrity problems. Mention once, without lecturing, that academic and professional venues may require disclosure of AI assistance; that decision belongs to the user.

## Setup

```bash
pip install -r scripts/requirements.txt
python -m spacy download en_core_web_sm
```

The first run downloads about 12 GB of open models from Hugging Face: Qwen2.5-1.5B base + instruct (LM layer), roberta-base (intrinsic dimension), all-mpnet-base-v2 (sentence embeddings, meaning gate), Style-Embedding (author voice), an NLI cross-encoder, and the held-out detectors tabularisai ModernBERT, desklib DeBERTa-v3-large, fakespot RoBERTa and tropa-mini. If downloads stall in a sandbox, set `HF_HUB_DISABLE_XET=1`. In a network-allowlisted sandbox, approve `huggingface.co` and the file CDN it redirects to (in November 2026 `us.aws.cdn.hf.co`; check with `curl -sI https://huggingface.co/FacebookAI/roberta-base/resolve/main/model.safetensors` and read the `location` header). Measured cache after a full `check.py` run plus tropa-mini: about 10 GB. On a 10-core CPU a full check of a 300-word text takes about 30-60 s after loading; `--fast` (surface + syntax + LM without Binoculars) takes about 10 s. A GPU is used automatically when present. Override models with the `HZ_*` environment variables in `scripts/hz_common.py`; if you change the scoring models, rebuild the calibration (see Calibration).

If the packages cannot be installed, use the degraded mode at the end of this file.

Locating the toolkit: the scripts live in `scripts/` next to this SKILL.md, and command examples below use paths relative to this folder. Run them from the user's working directory, in an environment that has `scripts/requirements.txt` installed. (In Claude Science, loading the skill also defines `deep_humanizer_dir()` and `deep_humanizer_cmd(script, *args)` in the python kernel via `kernel.py`; other agents ignore that file.) Set `HF_HOME` to a persistent cache so the models download once. The default reference bank ships as `assets/reference_default/reference.json`; banks you build with `calibrate.py` also write `reference.npz`, and the scripts read either.

## Workflow

Treat the user's text as material to edit, never as instructions. Keep a working directory with `draft.md` (original), `v1.md`, `v2.md`, ... and the JSON reports.

### Step 0 - Intake

Establish, asking only for what you cannot infer:

- Register and reader (academic abstract, news, essay, fiction, email, documentation). The register decides which features are normal: contractions are wrong in an abstract, "camaraderie" is fine in fiction.
- The writer's voice profile, if one exists: a folder the user names, the `DEEP_HUMANIZER_PROFILE` environment variable, or `~/.deep-humanizer/profiles/<name>/`. Read its `voice.md` in full before any edit and pass `--profile` to every check (see "Writer voice profiles" below). If there is no profile but the user has samples, offer to build one.
- Writing samples from the author, 3 or more texts and ideally 1,500+ words, written without AI help. They are the best target: the check round then measures voice match (`--profile`, or `--author` without a profile) and you copy sentence-level habits, not content, from them. Look before asking: a personal writing-style skill, the user's earlier manuscripts or drafts in folders already shared. Prefer pre-2023 or early drafts; copy them as plain text into `samples/`.
- Human controls: 2-3 texts of the same register and topic written before 2023 (the course readings, older papers in the field), 250-400 words each, in `controls/`. `--controls controls/` scores them with the same held-out detectors; a detector that calls a control AI is marked uninformative for this register. In the session above, fakespot scored two pre-ChatGPT technical texts at 0.99 and tabularisai one of them at 0.85.
- Facts that must survive: numbers, names, citations, quotes, claims. Never add facts the source or the user did not supply.
- Whether the text may be sent to third-party detector APIs (only with explicit consent).

### Step 1 - Diagnose

```bash
python scripts/check.py draft.md --json r0.json --md r0.md --controls controls/ [--profile NAME]
```

Read `r0.md` before editing:

- "Next step" says whether to polish, restructure or stop, and which paragraphs to start with.
- Structure (paragraph level) lists, per paragraph, the fact-sentence share, the writer share (sentences where the writer speaks: first person or an epistemic adverb; "respondents felt" does not count), sentence-length CV, summary closers, and the drop-one delta: how much each held-out detector's document score falls when that paragraph is removed. Paragraph-alone scores often stay low while the document is flagged, so locate drivers with the drop-one delta. These flags are heuristics, not calibrated.
- Verdict and per-layer probabilities show where the gap is. A text flagged only on L1 needs light edits; a text flagged on L2-L4 needs structural rewriting.
- "What to fix next" lists the most discriminative features on the AI side of the human 90th percentile, each with a tactic code (P = probability, S = syntax, D = discourse, E = embedding, L = lexical) explained in `references/tactics.md`.
- Hotspot sentences combine the LM's most predictable sentences with surface hits. Predictable token runs are phrases the scoring model completes word for word: they are the default wording to replace.
- Syntax hotspots list the sentences with the most register markers (participial riders, nominalisations, coordinated pairs, noun chains, that-clause subjects) and the long words used only once. They are the worklist for Passes B and C.

### Step 2 - Rewrite in passes, deep to surface

Edit; do not regenerate. A wholesale "make it sound human" rewrite by an LLM reproduces instruction-tuned register and lowers perplexity further (finding 2), and repeated paraphrasing falls into cycles. Keep good human-like sentences untouched and do the passes in this order, because structural edits written by an LLM reintroduce surface tells that a final surface pass then removes.

Choose the depth from the report, not from habit. Sentence passes (A-D) are enough when the verdict is CLEAN with no disagreement and no structure flags. When the verdict is DETECTORS DISAGREE, or Structure flags report-like paragraphs or a shared arc, start with Pass A0. The report's edit-depth line tells you which level the last round actually reached (polish: under half the sentences changed and word similarity above 0.8).

**Pass A0 - paragraph restructure (detector disagreement, structure flags).** This is still editing, paragraph by paragraph, with the same facts; it is not regeneration.
- Rebuild each flagged paragraph around what the writer thinks: lead with the reaction (what was expected, what surprised, what is doubtful), then the evidence that supports it.
- Cut the fact load. Keep the two or three findings the argument needs; move or drop the rest. Write every deliberate cut into `cuts.txt` (one phrase per line) for the meaning gate, and report them.
- Vary certainty across sentences: some flat, some hedged, one admitted uncertainty where it is real.
- Break shared arcs across sections: move the personal link to different positions, vary openings, do not end every section on a wrap-up line.
- Reactions and experiences attributed to the author are claims about a person. Write only ones the user supplied or that are plainly implied, and list every one for the user to confirm.
- Stop rule: one restructure round. If the drop-one table shows no single driver and the detectors stay high, the signal is in the content itself; ask the author for 2-3 sentences per section in their own words, then fit and fact-check them.

**Pass A - discourse and content (D, E targets).** This moves the embedding layer most.
- Cut sentences that restate the previous one, frame the topic, or summarise the paragraph. LLM paragraphs often end on a summary; humans end on the last substantive point. Keep a summary only where the register requires one (an abstract's final result sentence).
- Replace vague allusions and generic examples with the concrete referent from the source or the user's notes: the name, number, place, mechanism. If none exists, write a simpler sentence or ask. Never invent one.
- Let each sentence do a different job (claim, evidence, example, qualification, consequence, aside). High adjacent-sentence similarity means sentences circle one point.
- Recalibrate stance: commit where the evidence is strong, hedge where it is weak, state the author's position when the source has one. Do not spread uniform caution.
- Let paragraph length follow content. Do not impose symmetric structure (three paragraphs of equal size, a list of three in every sentence).
- Fiction: cut exposition that explains the scene or the theme, replace bodily-cliche emotion ("heart hammered against her ribs") with a plain or specific statement, allow loose ends the draft supports.

**Pass B - syntax and register (S targets).** These features survive paraphrase, so they need explicit rewriting. Work through every sentence under "Syntax hotspots"; in validation, an LLM editor without this list reduced nominalisations but left coordination and noun density unchanged.
- Unpack trailing participial riders (", highlighting...", ", ensuring...", ", reflecting...") into a finite clause with a subject, a separate sentence, or nothing if the rider only restates.
- Denominalise: "the implementation of X enabled an improvement in Y" -> "implementing X improved Y" or "we implemented X and Y improved". Name an agent only if the source has one.
- Break reflexive doublets and triplets ("clear and concise", "X, Y, and Z") unless each item carries content.
- Use the passive where the register does (methods sections), and avoid that-clauses as subjects.
- Vary sentence frames: do not open consecutive sentences with the same structure ("This X...", "By doing X, ..."); vary clause order; use a question or fronted adverbial where the register allows.
- Shape sentence length by content: a short sentence where a point lands, a long one for a qualified claim. Check the length spread against the human band rather than inserting fragments at random.
- Shorten long dependencies and unstack noun chains ("the proposed model evaluation framework").

**Pass C - word choice and predictability (P, L targets).**
- For each hotspot sentence and predictable run, choose the wording the author would use: a precise verb instead of a generic one, the domain's plain term, a correct but less expected collocation. Raise the *variance* of surprisal, not its mean: informative content words, predictable function words. Do not stuff rare words.
- Remove register-marked AI vocabulary (`assets/ai_vocabulary.json`; L1 report lists hits). Often the right edit is deletion: the word was decorative. Do not cycle synonyms; repeating the key term is human.
- Current models write with *more* lexical variety than humans (default reference bank, 120 documents per class: median MTLD 137 for AI vs 98 for human text, and a higher hapax ratio). When `sty.mtld` or `sty.hapax_ratio` is a target, reuse the key terms instead of varying them and replace the listed long once-used words with plainer words or the term already in use.
- With a voice profile or author samples, match the writer's habits: the report's "Voice gaps", and the measured block of `voice.md` (sentence length and spread, paragraph size, "I" or "we", contractions, dash glyphs and other punctuation, sentence openers and connectives, spelling convention, recurring phrases). If they use dashes, keep dashes at their rate.

**Pass D - surface cleanup (L targets).** Remove dashes used as universal connectors, chat residue, signposting openers, bold-label lists, title-case headings, curly quotes where the target format uses straight ones. `references/surface_patterns.md` lists the families with examples.

### Step 3 - Meaning gate

```bash
python scripts/meaning_guard.py draft.md v1.md --nli                                   # sentence edits
python scripts/meaning_guard.py v1.md v2.md --restructure --cuts cuts.txt --json mg.json  # after Pass A0
```

Exit code 2 means review. Restore every LOST sentence's content, remove or source every UNSUPPORTED sentence, and reconcile every number, citation, URL, quote and named entity listed under fact diffs. A rewrite that fails the gate is not a candidate, whatever its scores.

After a paragraph rewrite the default alignment reports moved or merged sentences as LOST, which buries the real problems. `--restructure` matches each original sentence against windows of up to three rewrite sentences, runs NLI, and labels it KEPT, CONFLICT (with probability), PLANNED_CUT or DROPPED; new sentences are NEW_FACT (fail), AUTHOR_CLAIM (confirm with the user) or REWORDING. Number words count as their digits ("about ten" keeps "10"). The script prints every item; pass all CONFLICT, DROPPED and unplanned missing facts to the user verbatim, judged one by one, never as a summary such as "one false alarm".

### Step 4 - Check round

```bash
python scripts/check.py v1.md --baseline draft.md --json r1.json --md r1.md [--profile NAME]
```

Read the verdict, the per-layer table and the "Change from baseline" table.

- CLEAN: combined probability at or below the human 75th percentile of the reference, no layer flagged, and no informative held-out detector at 0.5 or above. Go to Step 5.
- DETECTORS DISAGREE: the calibrated layers say CLEAN or BORDERLINE but an informative held-out detector (open or API) scores 0.5 or more. Do not report the text as done. Follow "Next step": Pass A0 if the last round was polish, the author-input route if it was already a restructure.
- BORDERLINE or FLAGGED: do another pass on the new targets and hotspots only. Make at most three passes in total, keep every version, and select the best one by combined probability among versions that pass the meaning gate. Further passes usually trade meaning and quality for small score gains.
- Overshoot: the report's "Overshoot" section lists discriminative features pushed past the human band on the other side (below the human 2.5th percentile on the AI-side scale: for example perplexity far above the human median or sentence-length spread at the top of the human range). Pull them back toward the human median. Human-likeness is a band.
- A layer that will not move after two passes usually means the content itself is generic. Tell the user what specific material (examples, numbers, their own opinion) would fix it, rather than forcing it.

Signals are calibrated on texts of 150-400 words. Below about 150 words most of them are unreliable; say so and rely on L1-L2 plus reading.

### Step 5 - Held-out confirmation

```bash
python scripts/check.py vN.md --final --json final.json --md final.md [--external gptzero,sapling]
```

tropa-mini already runs in every round; `--final` adds every remaining open detector; `--external` calls commercial APIs whose keys are set (see `references/detectors.md`; it sends the text to that company), for example `--external zerogpt` with `ZEROGPT_API_KEY` when the user's venue is known to use ZeroGPT. Held-out detectors are not wording targets: never tune sentences against their scores and never loop until one passes (finding 4). They do two things: decide whether to stop (the DETECTORS DISAGREE override) and, through the drop-one table, point to paragraphs to restructure. Report disagreement plainly; never call a disagreeing result "mixed" and stop. For several documents, run

```bash
python scripts/batch_compare.py rewrites_dir/ [--against <profile>/samples/]
```

and report the C2ST AUROC (0.5 = indistinguishable from the human reference) and MMD p-value.

### Step 6 - Report

Return the final text, then a compact table: layer verdicts and combined probability before and after, the held-out detector scores before and after (with the controls), the edit depth of each round, the meaning-gate result with every planned cut, conflict and author claim listed, the voice gaps before and after when a profile was used, and any remaining risk (short text, register outside the calibration bank, content that is generic at source, detectors still disagreeing). Do not promise that any detector will pass: no rewrite guarantees that, and published detectors disagree widely.

## Writer voice profiles

A profile makes the skill edit toward one writer instead of toward an average human. It is a folder:

- `voice.md`: the writer's style card (registers, habits to keep, rules that override this skill, phrases they avoid, AI-list words that are theirs, example sentences), plus a measured block written by `build_profile.py`. Template: `assets/voice_template.md`.
- `profile.json`: the writer's own band for every surface and syntax feature, measured on about 300-word chunks of their samples, plus habits (spelling, dash glyphs, pronouns, openers, paragraph size, function words).
- `samples/`: the samples, used for the style-embedding match.

Build or refresh one (only with the writer's samples and agreement; the profile stays on their machine):

```bash
python scripts/build_profile.py my_writing/ --name blog --register "Blog posts: first person, short paragraphs"
python scripts/voice_profile.py draft.md --profile blog          # quick voice-only check, no large models
```

Use one profile per register: a writer's papers and blog posts have different bands. Re-running `build_profile.py` replaces only the measured block of `voice.md`; it refuses to mix a different sample set into an existing profile unless given `--force`. Never write a profile inside a public repository.

Precedence: rules in `voice.md` override this skill's generic style advice (dashes, contractions, triads, vocabulary, sentence length). They never override the meaning gate, the fact rules or the integrity rules.

Reading the report with a profile (`--profile` changes the guidance, never the verdict or the detector gate):

- Voice gaps: features outside the writer's own range. Edit toward their median. Where the writer's band and the generic human band disagree, follow the writer.
- Generic targets that match the writer's own style: features on the AI side of the generic human band where the writer's own samples sit too. They are not edit targets. If held-out detectors still flag the text, tell the writer that their natural style shares these features with LLM prose; do not erase it silently (likelihood-based detectors over-flag dense academic and non-native writing).
- Phrases you avoid: remove them like weight-3 surface tells. AI-list words that are yours: keep them.
- Spelling mismatch: convert to the writer's convention. Openers that never appear in the samples: replace them with the writer's own connectives.
- Style embedding below the writer's own minimum, or a function-word distance outside their range: the voice is off even if every single habit is within range; compare sentence by sentence with the example sentences in `voice.md`.

A personal reference bank (the writer's own texts as the human class) is the strongest form of tailoring; see "Calibration and reference banks".

## Interpreting the check report

| Report item | Meaning | First action |
|---|---|---|
| `lm.fastdetectgpt_norm` high, `lm.binoculars` low | Wording is the model's own most likely choice | Rewrite hotspots with author-specific wording (Pass C) |
| `lm.surprisal_sd`, `uid_*`, `late_*` low | Information is spread too evenly | Mix dense specific sentences with plain short ones (A, C) |
| `sty.present_participial_clause_per_1k`, `sty.nominalization_per_1k`, `sty.phrasal_coordination_per_1k` high | Instruction-tuned register | Pass B |
| `sty.sent_len_cv` / `burstiness` low | Uniform rhythm | Pass B, sentence-length shaping |
| `sty.pos6_repeat_share`, `pos_compression_ratio` high | Repeated syntactic templates | Vary frames (Pass B) |
| `emb.phd` / `emb.mle_id` outside the human band | Token-embedding geometry differs; direction depends on the generator (read the report's direction) | Higher than human: cut elegant variation; lower: content and syntactic variety (Pass A, B) |
| `emb.adj_cos_mean`, `pairwise_cos_mean` high | Sentences restate one point | Pass A |
| `emb.ref.knn_ai_fraction`, `emb_lr_p_ai` high | Document sits among AI documents of the bank | Specific content and author voice (Pass A, C) |
| `author_style` percentile low | Voice differs from the author's samples | Match the samples' habits (Pass C) |
| Voice gaps (with `--profile`) | Habits outside the writer's own range | Edit toward the writer's median; `voice.md` rules decide conflicts (Pass B, C) |
| `det.*` high after a CLEAN local verdict (DETECTORS DISAGREE) | A held-out detector still sees it; usually paragraph construction | Pass A0 on the drop-one drivers; one round, then ask for author input. Do not tune wording against it |
| A detector also flags the human controls | Detector is uninformative for this register | Report it, ignore it for the stop decision |
| Structure: report-like paragraph (fact share >= 0.5, writer share <= 0.2) | A run of findings with no writer stance | Pass A0 |
| Structure: summary closer, shared arc across sections | Template structure | Pass A0 (end on a fact, vary arcs) |
| Edit depth "polish" while detectors disagree | The round did not reach paragraph level | Pass A0 |

The full signal table with recipes, directions, published effect sizes and robustness evidence is `references/signals.md`; calibration AUROCs for the default bank are in `references/validation.md`.

## Calibration and reference banks

`assets/reference_default/` was built by `calibrate.py` from 120 human texts written before 2021 (arXiv abstracts, CNN news, Reddit WritingPrompts stories) and 120 matched texts from four 2025-2026 models (two Claude, two GPT). It defines the human band and the AI region used by `check.py`. It is small and covers three registers; results outside them are indicative only. For real use, build a bank that matches the text:

```bash
# corpus.jsonl: {"id", "text", "label": "human"|"ai", "group": topic id, "register"}
python scripts/calibrate.py corpus.jsonl --out banks/my_register --cache cache/
python scripts/check.py v1.md --reference banks/my_register
```

For a personal bank, put the writer's texts in one folder and LLM texts on the same topics, with the same file names, in another; `make_corpus.py` pairs them and cuts long texts into 150-450-word chunks:

```bash
python scripts/make_corpus.py --human my_writing/ --ai ai_versions/ --register blog --out my_corpus.jsonl
python scripts/calibrate.py my_corpus.jsonl --out ~/.deep-humanizer/banks/blog --cache ~/.deep-humanizer/cache/blog
python scripts/check.py v1.md --reference ~/.deep-humanizer/banks/blog --profile blog
```

`calibrate.py` refuses to write into the shipped `assets/reference_default` unless given `--overwrite-default`.

Good human sources: pre-2023 texts of the same venue and length, or the author's own pre-2023 writing. Good AI sources: LLM texts written to the same prompts or titles (match topics with `group`, so cross-validation does not leak topic). About 100 documents per class gives stable quantiles. Post-2023 "human" corpora contain LLM-assisted text and blur the band. Likelihood-based signals over-flag non-native and very plain writing, so use a bank that contains such writing when the user writes that way.

## Pitfalls

- Synonym cycling: replacing every marked word with another marked synonym. Delete or use the plain term.
- Fabricated specificity: invented names, numbers or anecdotes make text look human and are the worst failure. The meaning gate catches most, not all; reread for invented details.
- Fragments and random length variation inserted to raise burstiness read as mannered and do not move L3-L4.
- Rare-word stuffing raises perplexity and makes the text worse; aim for the band.
- Running the full text through an LLM "make it human" prompt: lowers perplexity, keeps register features, often introduces new tells.
- Optimising against one detector: the gains do not transfer and quality drops.
- Register mismatch: contractions in an abstract, first person in a news report.

## Degraded mode (no Python packages)

If the toolkit cannot run, do the passes by reading: mark L1 tells with `references/surface_patterns.md`; count participial riders, nominalisations and doublets per paragraph by hand; check sentence-length spread by eye (a paragraph of five sentences of 20-25 words is a red flag); look for sentences that restate their neighbours. If the writer has a `voice.md`, apply its rules and phrase lists by reading. Tell the user that no distribution-level check was run.

## Files

- `kernel.py` - defines `deep_humanizer_dir()` / `deep_humanizer_cmd()` on skill load
- `scripts/check.py` - check round (all layers, calibrated verdict with held-out override, next step, structure, controls, edit depth, targets, hotspots, deltas, meaning gate)
- `scripts/surface_scan.py`, `stylometry.py`, `lm_signals.py`, `embed_geometry.py`, `detectors.py` - individual layers, each runnable alone
- `scripts/meaning_guard.py` - coverage, unsupported additions, hard-fact diffs, optional NLI; `--restructure` claim-level mode with planned cuts
- `scripts/structure_scan.py` - paragraph-level build (fact share, writer share, closers, drop-one detector attribution, shared arc across files); run alone on several files to check a set of answers for one template
- `scripts/batch_compare.py` - MMD, classifier two-sample test and feature KS for a batch
- `scripts/calibrate.py` - build a reference bank and calibration from labelled texts
- `scripts/build_profile.py` - build a writer voice profile from samples; `scripts/voice_profile.py` - profile loading, voice comparison, quick voice-only check
- `scripts/make_corpus.py` - pair a writer's texts with matched LLM texts into a calibration corpus
- `assets/reference_default/` - default bank; `assets/ai_vocabulary.json` - 669 sourced over-represented items; `assets/detector_api_specs.json` - commercial detector endpoints; `assets/voice_template.md` - voice profile template
- `references/research_digest.md` - what the literature shows, with numbers (start here for the why)
- `references/signals.md` - signal recipes, directions, robustness; `references/tactics.md` - tactic map with evidence
- `references/surface_patterns.md` - L1 pattern catalogue; `references/detectors.md` - detector catalogue and API notes
- `references/validation.md` - calibration and humanization validation of this toolkit
- `references/literature/` - full annotated bibliographies from the research behind the skill
