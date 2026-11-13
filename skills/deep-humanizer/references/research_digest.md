# Research digest: how LLM prose differs from human prose, and what that means for humanizing

Synthesis of three literature reviews carried out for this skill in November 2026 (full annotated bibliographies with verified identifiers, summaries and numbers in `literature/`). Identifiers were checked against arXiv and Crossref; numbers were string-matched against the papers' full text by the reviewers. Effect sizes come from the original models, registers and lengths and do not transfer directly to other settings; that is why the toolkit calibrates every signal on a reference bank instead of using published thresholds.

Contents: 1 layers of difference · 2 detector families and their robustness · 3 what evasion research teaches · 4 measurement pitfalls · 5 design decisions · 6 key references

## 1. Where human and LLM prose differ

### 1.1 Lexical layer

- Excess vocabulary is real and large for specific words: *delves* excess ratio r = 28.0 in 2024 PubMed abstracts and at least 13.5 percent of 2024 abstracts LLM-processed (Kobak et al. 2025, arXiv:2406.07016); *meticulous* 34.7x, *commendable* 9.8x, *intricate* 11.2x more likely per sentence in AI-modified peer reviews (Liang et al. ICML 2024, arXiv:2403.07183); *camaraderie* 162x in GPT-4o output (Reinhart et al. 2025, arXiv:2410.16107); some strings over 1,000x the human rate across 67 models (Antislop, arXiv:2510.15061). Juzek and Ward (arXiv:2412.11385) trace the over-use partly to preference learning.
- It is the weakest lever. On RAID (arXiv:2405.07940), synonym substitution reduced Binoculars detection by 36.1 points but RADAR by 3.4, and raised Originality's. Model fingerprints sit in word choice yet survive paraphrase: embedding classifiers separate five chat models at 97.1 percent and stay above 90 percent after LLM rephrasing or translation (Sun et al., "Idiosyncrasies in LLMs", arXiv:2502.12150).
- Lexical diversity is two-sided: some open models fall below human MTLD (human 96.51 vs 57.37-94.56; Muñoz-Ortiz et al., arXiv:2308.09067), ChatGPT-4 essays exceed human diversity (Herbold et al., arXiv:2304.14276), and commercial humanization lowers it. Target the register's band.

### 1.2 Register and syntax

- Reinhart et al. (PNAS 2025, arXiv:2410.16107), with Biber's lexico-grammatical features on matched human/LLM text: GPT-4o uses present participial clauses at 5.3x the human rate (d = 1.38), nominalisations 2.1x (d = 1.23), that-clauses as subject 2.6x, phrasal coordination 1.9x, and agentless passives at about 0.5x. Instruction-tuned models show 2-5x participial clauses; base models are close to human rates. A random forest separates human from LLM text at 93-98 percent pairwise.
- Prompting a model toward human-like style does not remove these differentiators (Rallapalli et al. 2026, arXiv:2604.14111). Base-model text reads as human to commercial detectors where instruct text does not (Xu et al. 2026, "Base Models Look Human To AI Detectors", arXiv:2605.19516), which points at instruction tuning as the source.
- Syntactic templates (Shaib et al., EMNLP 2024, arXiv:2407.00211): 95 percent of model outputs contain length-6 POS templates against 38 percent of human references, the rate stays at 96.6-97.4 percent across temperatures and top-p, and 76 percent of model templates occur in pre-training data against 35 percent for human text. Synonym-swap humanizers leave templates intact.
- Rhythm: human sentence lengths are more scattered (Muñoz-Ortiz et al.), and human texts vary more in sentence-level features in general (Zanotto and Aroyehun, arXiv:2412.03025). LLM syntax uses longer dependencies and stacked nouns that are harder to process (Clark et al. 2026, arXiv:2610.29162).

### 1.3 Probability and information flow

- LLM decoding favours high-probability continuations, so machine text has positive conditional probability curvature: its tokens are likelier than tokens sampled from the scoring model would be (Fast-DetectGPT, arXiv:2310.05130; mean AUROC 0.9887 white-box and 0.9677 black-box in the paper). Binoculars (arXiv:2401.12070) normalises perplexity by cross-perplexity between two related models and detected over 90 percent of ChatGPT text at 0.01 percent FPR with 7B models.
- Humans distribute information more unevenly. Uniform-information-density features (surprisal variance, consecutive differences, extremes over 20-token spans) separate authors and beat GLTR, DetectGPT and GPTZero by over 20 percent (GPT-who, arXiv:2310.06202); human surprisal distributions are wider and heavier-tailed (DivEye, arXiv:2509.18880); AI text shows 24-32 percent lower log-probability volatility than human text in its second half (arXiv:2601.04833). GLTR (arXiv:1906.04043) showed humans sample more tokens outside the model's top ranks.
- Small scoring models work: OPT-125M reached AUC 0.81 on ChatGPT text where GPT-J-6B was at 0.45 (Mireshghallah et al., arXiv:2305.09859).

### 1.4 Embedding geometry and style

- Intrinsic dimension (Tulchinskii et al., NeurIPS 2023, arXiv:2306.04723): treat a text's RoBERTa token embeddings as a point cloud and estimate its persistent-homology dimension. English human text is mostly 9-10, generated text about 1.5 lower regardless of generator. Per-text AUC is moderate (0.78 for ChatGPT), but the signal barely moves under DIPPER paraphrase (mean PHD of generated text 7.30 to 7.51), and its false-positive rate on non-native essays (26 percent) was lower than GPTZero's (52 percent). Below about 200 tokens it becomes unstable (Short-PHD, arXiv:2504.02873).
- Style representations: content-independent style embeddings separate human and model text where semantic embeddings do not (Rivera Soto et al., ICLR 2024, arXiv:2401.06712), and style-based detection stays at AUROC about 95 across evasion attacks (Rivera Soto et al., ICML 2026, arXiv:2505.14608). Contrastive retrieval detectors (DeTeCtive, arXiv:2410.20964) and intermediate-layer probes (Text Fluoroscopy, EMNLP 2024; RepreGuard, arXiv:2508.13152) are strong in distribution.
- Distribution-level tests: MMD between feature distributions (MMD-MP, arXiv:2402.16041) and register-aware Biber-feature MMD (arXiv:2605.23651) found every tested instruction-tuned model off the human baseline in every register.

### 1.5 Discourse and content

- Homogenisation: writing with InstructGPT made essays across users more similar (Padmakumar and He, arXiv:2309.05196); LLM outputs are more similar to each other than human outputs are (arXiv:2501.19361, arXiv:2510.22954). In long-form debate, 65.3 percent of human main arguments were unique against 3.4 percent of LLM arguments (Argument Collapse, arXiv:2606.01736).
- AI stories explain their theme 77 percent of the time vs 52 percent for humans, use vague allusions more (72 vs 50 percent) where humans name specific texts and authors (47 vs 24 percent), and break the fourth wall less (39 vs 67 percent) (StoryScope, arXiv:2604.03136); narrative features still detected AI stories at 93.9 percent macro-F1 after span-level artefact removal.
- Stance and engagement are thinner: humans use more modal and epistemic constructions (Herbold et al.), more questions and personal asides; 76.1 vs 63.4 percent of essays take a strong stance (Argument Collapse).
- Expert editing (LAMP, CHI 2025, arXiv:2409.14509): the main edit categories are awkward word choice (28 percent), poor sentence structure (20), unnecessary exposition (18) and cliches (17); 74 percent of writer edits are replacements; writer-edited paragraphs were preferred 65 percent of the time. LLM stories pass 9-30 percent of expert creativity tests (TTCW, arXiv:2309.14556).

## 2. Detector families and how robust they are

| Family | Examples | Strength | Weakness for this use case |
|---|---|---|---|
| Likelihood / curvature (zero-shot) | Perplexity, log-rank, Fast-DetectGPT, Binoculars, DNA-GPT, Glimpse | No training, strong on raw output | Weak against humanized or rewritten text: Binoculars detected 6.7 percent and Fast-DetectGPT 23.3 percent of O1-PRO humanized articles (Russell et al., ACL 2025, arXiv:2501.15654); 15.1 and 30.8 percent TPR at 1 percent FPR on LLM-rewritten human text (ARB, arXiv:2607.29539); repetition penalty alone cost up to 32 points (RAID). Over-flag non-native writing (Liang et al., arXiv:2304.02819). |
| Supervised classifiers | RoBERTa/DeBERTa detectors, RADAR (arXiv:2307.03838), Pangram (arXiv:2402.14873), commercial tools | High in-distribution accuracy; adversarially trained ones resist paraphrase | Generalise poorly to new generators and registers; open checkpoints trained on 2019-2024 data |
| Geometry / style | PHD, style embeddings, retrieval | Survive lexical paraphrase | Need length (PHD) or reference data; moderate per-text AUC |
| Register / stylometry | Biber features, POS templates, sentence statistics | Interpretable, content-light, survive synonym swaps | Domain shift; individual features are easy to game in isolation |
| Human experts | Frequent LLM users as readers | 99.3 percent majority-vote accuracy (Russell et al.) | Slow; rely on vocabulary but also on structure |

Low scores on likelihood detectors are therefore not evidence of human-likeness, and high agreement across families is what the check round looks for.

## 3. What evasion research teaches a humanizer

1. Edit, do not regenerate. A plain LLM paraphrase lowered GPT-2-family perplexity on MAGE from 14.94 (AI original) to 9.28, against 15.02 for human text, and raised Fast-DetectGPT's TPR at 1 percent FPR by 15 percent (Cheng et al., NeurIPS 2025, arXiv:2506.07001). Repeated paraphrasing converges to 2-period attractor cycles (arXiv:2502.15208).
2. Optimising against a detector is a different objective from human-likeness. Detector-guided paraphrase cut TPR by 87.88 percent on average (Cheng et al.); AuthorMist (arXiv:2503.08716) reached perplexity 300-400 against a human median of about 50-75; StealthRL (arXiv:2602.08934) reached 97.6 percent attack success with E5 similarity 0.901 but NLI entailment 0.316. Gains against one detector transfer poorly.
3. Meaning needs entailment and fact checks. All 11 audited commercial humanizers changed numbers, names, dates or quotes in 35-65 percent of documents, and fact retention was unrelated to evasion (Wang et al. 2026, arXiv:2611.00359).
4. What survives word-level attacks: retrieval (Krishna et al., arXiv:2303.13408), style embeddings, intrinsic dimension, templates, narrative structure. A rewrite that only changes words leaves them all in place.
5. Single documents vs distributions: a style-aware paraphraser (fine-tuned on human exemplars) evaded all detectors on single documents, but detection recovered as the number of documents grew (Rivera Soto et al. 2026, arXiv:2505.14608). Check batches as distributions.
6. Character-level tricks (homoglyphs, zero-width characters, misspellings, article deletion) move some detectors strongly on RAID but make the text worse and are trivially normalised. They are out of scope.

## 4. Measurement pitfalls

- Length: a Binoculars threshold set at 200 tokens drifts to 29 percent FPR at 10 tokens; with binned calibration at 5 percent FPR the best TPR at 200 tokens was 59 percent (Janga and Zhan 2026, arXiv:2611.10901). Calibrate within length ranges and abstain on short texts.
- Register: thresholds and feature bands differ by genre; Biber-based classifiers lost accuracy on arXiv abstracts (genre shift).
- Population: non-native and plain-style human writing is over-flagged by likelihood signals (Liang et al. 2023, arXiv:2304.02819: widely used detectors consistently misclassified non-native English writing as AI-generated).
- Contamination: post-2023 "human" corpora contain LLM-assisted text (up to 17.5 percent of computer-science papers and up to 24 percent of corporate press releases; arXiv:2404.01268, arXiv:2502.09747). Use pre-2023 or vetted human references.
- Held-out evaluation: any detector used to guide or select rewrites must be excluded from the evaluation (Cheng et al.).

## 5. Design decisions in this skill and their basis

| Decision | Basis |
|---|---|
| Four measured layers, one report | Different detector families key on different layers (sections 1-2) |
| Human band from a calibrated reference, not fixed thresholds | Section 4; published thresholds are model-, length- and register-specific |
| Passes ordered discourse -> syntax -> word choice -> surface | Structural signals survive word edits; LLM-written edits reintroduce surface tells |
| Targeted edits with at most three passes and best-of selection | Cheng et al.; attractor cycles; LAMP (writers replace spans) |
| Aim for the band, flag overshoot | AuthorMist overshoot; dispersion findings (GPT-who, DivEye) |
| Meaning gate with coverage, support, fact diffs and NLI | Humanizer fact audit; StealthRL NLI gap |
| Open detectors and commercial APIs used only for confirmation | Goodhart effect in Cheng et al., SHADE, AuthorMist transfer gaps |
| Author samples as the preferred target | Style embeddings are the signal generic rewriting misses (Rivera Soto et al.); voice is what the user wants |
| Batch MMD and classifier two-sample test | Detection recovers at the distribution level |
| Character-level tricks flagged, never used | RAID; normalisation defeats them; quality loss |

## 6. Key references (arXiv IDs verified)

Tulchinskii et al. 2023 (2306.04723) · Bao et al. Fast-DetectGPT 2024 (2310.05130) · Hans et al. Binoculars 2024 (2401.12070) · Mitchell et al. DetectGPT 2023 (2301.11305) · Gehrmann et al. GLTR 2019 (1906.04043) · Basani and Chen DivEye 2026 (2509.18880) · Mireshghallah et al. 2024 (2305.09859) · Reinhart et al. PNAS 2025 (2410.16107) · Shaib et al. 2024 (2407.00211) · Muñoz-Ortiz et al. 2024 (2308.09067) · Rallapalli et al. 2026 (2604.14111) · Xu et al. 2026 (2605.19516) · Kobak et al. 2025 (2406.07016) · Liang et al. 2024 (2403.07183, 2404.01268) · Juzek and Ward 2025 (2412.11385) · Paech et al. Antislop 2025 (2510.15061) · Sun et al. Idiosyncrasies 2025 (2502.12150) · Rivera Soto et al. 2024 (2401.06712) and 2026 (2505.14608) · Venkatraman et al. GPT-who (2310.06202) · Dugan et al. RAID 2024 (2405.07940) · Krishna et al. DIPPER 2023 (2303.13408) · Sadasivan et al. 2023 (2303.11156) · Cheng et al. 2025 (2506.07001) · David and Gervais AuthorMist 2025 (2503.08716) · Russell et al. 2025 (2501.15654) · Liang et al. 2023 non-native bias (2304.02819) · Padmakumar and He 2024 (2309.05196) · Chakrabarty et al. TTCW 2024 (2309.14556) and LAMP 2025 (2409.14509) · Janga and Zhan 2026 (2611.10901) · Wang et al. humanizer fact audit 2026 (2611.00359) · Emi and Spero Pangram 2024 (2402.14873) · Hu et al. RADAR 2023 (2307.03838).
