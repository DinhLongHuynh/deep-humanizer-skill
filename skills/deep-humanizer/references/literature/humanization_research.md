# Humanization, detector evasion, and the linguistics of human vs LLM prose

Annotated bibliography for the `deep-humanizer` skill (reference file). Compiled 2026-11-13.

## 0. How this file was built

- **Identity check.** Every arXiv ID below was resolved through the arXiv API (batch lookup; no ID failed). DOIs were confirmed through the Crossref REST API. Venue is reported only when it was confirmed by (a) the paper's own PDF header, (b) the arXiv `comment`/`journal_ref` field, or (c) a Crossref proceedings/journal record whose title matched the arXiv title. Otherwise the entry says **venue UNVERIFIED** (the preprint itself is verified). OpenAlex was not used (no API key configured).
- **Numbers.** Full texts (PDF) were downloaded for 50 arXiv papers. Findings were extracted with LLM assistance, and every quoted number used here was then string-matched against the extracted PDF text. Numbers marked *(table)* were read from a table row whose column headers were checked by hand. Numbers that could not be matched were dropped.
- **Scope rule.** Character-level manipulations (homoglyphs, zero-width characters, whitespace insertion, misspelling injection, article deletion) are recorded only as **detector gaming, out of scope** (Section 3). The skill targets prose that is closer to human writing, not obfuscated prose.
- Tags: `[arXiv:ID]` verified preprint; `[DOI:…]` verified DOI; `venue: … (source)` shows where the venue was confirmed.

---

## 1. Findings that matter most for the skill

1. **A plain LLM paraphrase makes text more predictable, not more human.** In Cheng et al. (2025), GPT-2-family perplexity on MAGE was 14.94 for original AI text, 15.02 for human text, and **9.28 after simple LLM paraphrasing**. Detector-guided paraphrases came back to 14.26–17.11 *(table)*. Successive LLM paraphrasing converges to 2-period attractor cycles (Wang et al., 2025), and LLM fingerprints survive paraphrasing and translation (Sun et al., ICML 2025). The skill should therefore make targeted edits, not regenerate the text wholesale.
2. **Detector-optimised rewriting overshoots or degrades.** AuthorMist outputs reach perplexity 300–400, against a human median of roughly 50–75. SHADE's best GRPO policy produced shorter texts (374 vs 551 words) with lower TTR and fewer hapaxes. StealthRL reached 97.6% attack success with E5 similarity 0.901, but NLI entailment was only 0.316. Evasion and humanlikeness are different objectives.
3. **What stays robust under paraphrase is structure and style.** Retrieval stays robust (Krishna et al.). So do style-embedding detectors: StyleDetect AUROC averages 95 across evasion strategies (Rivera Soto et al., ICML 2026). Intrinsic dimension rises slightly under DIPPER (Tulchinskii et al.). Discourse/narrative features give 93.9% macro-F1 after span-level cleanup (StoryScope). Structural features beat style features by 8.8 macro-F1 (SlopShape). Expert human readers detect with 99.3% majority-vote accuracy (Russell et al., ACL 2025). A rewrite that only changes words leaves all of these signals in place.
4. **The register-level fingerprint is grammatical.** Instruction-tuned models use present participial clauses at 2–5× the human rate (GPT-4o 5.3×, d = 1.38), nominalisations at 1.5–2×, that-clauses as subject 2.6×, and phrasal coordination 1.9×. They use agentless passives at about half the human rate (Reinhart et al., PNAS 2025). Base models are close to human rates, which points to instruction tuning as the source.
5. **Syntactic templates are pervasive and decoding does not remove them.** 95% of model outputs contain length-6 POS templates, against 38% of human references. On Rotten Tomatoes the template rate stays at 96.6–97.4% across temperatures and top-p values *(table)* (Shaib et al., EMNLP 2024).
6. **Lexical diversity is not a one-directional knob.** Some LLMs are below human MTLD: human 96.51 vs 57.37–94.56 across six open models (Muñoz-Ortiz et al.). GPT-4 is above human (Herbold et al.). In fiction rewrites the LLM uses a wider synonym spread (+0.29 bits entropy; Suvanto et al., ICAART 2026). Commercial humanisation lowers diversity (Karr et al., 2026). Target the register's reference interval, not a maximum.
7. **Surprisal is more uneven in human text.** Human UID scores have a higher mean and larger standard deviation (GPT-who). Human text has higher token-probability variance than beam search (Holtzman et al.) and samples more often from the tail of the model distribution (GLTR). The target is local variance at content words, not a higher mean perplexity.
8. **Excess vocabulary is well documented but is the weakest lever.** Examples: *delves* r = 28.0 (Kobak et al.); *delves* +6697% occurrences per million 2020→2024 (Juzek & Ward); *meticulous* 34.7-fold (Liang et al., ICML 2024); *camaraderie* 162× (Reinhart et al.); *Elara* 85,513× (Antislop). Synonym substitution moved Binoculars by −36.1 points but RADAR by only −3.4 (RAID) *(table)*.
9. **LLM argumentation and narrative are homogeneous.** In Argument Collapse, 65.3% of human main arguments are unique within a debate against 3.4% of vanilla LLM arguments. AI stories explain their theme 77% of the time vs 52% for humans, and use vague allusions rather than named references (StoryScope). Writing with InstructGPT raises corpus homogenisation (Padmakumar & He, ICLR 2024). Rewriting cannot fix this without author-provided content.
10. **Stance and engagement are thinner in LLM prose.** Students use more epistemic stance and authorial presence (Jiang & Hyland 2024) and more engagement markers, especially questions and personal asides (Jiang & Hyland 2025). Humans use more modal and epistemic constructions (Herbold et al.). 76.1% of human essays take a strong stance vs 63.4% of LLM essays (Argument Collapse).
11. **Edits by writers beat LLM edits, and LLM edits beat raw generation.** Writer-edited paragraphs were most preferred 65% of the time. 74% of writer edits are replacements. Top edit categories: awkward word choice (28%), poor sentence structure (20%), unnecessary exposition (18%), clichés (17%) (LAMP, CHI 2025).
12. **Meaning-preservation must be checked with entailment, not only embeddings.** BERTScore drops more than T5 similarity under evasive paraphrasing (8.94% vs 3.90%, DEPO). StealthRL shows high embedding similarity alongside low NLI. DAMAGE reports humanizer quality "win rates" of 26.0% / 14.67% / 2.67% across humanizer tiers.
13. **Detection recovers at the multi-document level.** Rivera Soto et al. and Chakraborty et al. both argue that single-document indistinguishability does not imply distributional indistinguishability. The check round should evaluate a document set, not one sample.

---

## 2. Evasion, paraphrase and humanizer studies

Each entry lists: **operation** · **what moved** · **what did not move** · **semantic / quality cost** · **implication for the skill**.

### 2.1 Krishna, Song, Karpinska, Wieting, Iyyer (2023). *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense.* [arXiv:2303.13408] venue: NeurIPS 2023 (PDF header)
- **Operation.** DIPPER, an 11B T5-XXL discourse-level paraphraser trained on aligned translations. Scalar codes control lexical (L) and order (O) diversity. Single pass, no detector feedback.
- **Moved.** DetectGPT accuracy fell "from 70.3% to just 4.6%" at 1% FPR (GPT2-XL, L60/O60). Watermark detection fell "from 100% to 57.2%". The OpenAI classifier fell from 30.0% to 15.6% (davinci-003). GPTZero went from 13.9 to 1.2 on GPT2-1.5B text *(table)*.
- **Did not move.** A retrieval defence (BM25 over the provider's generation log) detected 97.8% of the highest-diversity paraphrases on GPT2-XL. When the retrieval corpus grew from 1M to 15M generations, detection fell only 1 point on PG19 (98.3 → 97.3) and 9.6 points on Wikipedia (90.0 → 80.4).
- **Implication.** Single-pass discourse paraphrase removes token-level probability signatures. Content-level identity survives.

### 2.2 Sadasivan, Kumar, Balasubramanian, Wang, Feizi (2023/2025). *Can AI-Generated Text be Reliably Detected?* [arXiv:2303.11156] venue: TMLR (arXiv comment)
- **Operation.** Recursive paraphrasing (DIPPER, LLaMA-2-7B-Chat), repeated several rounds.
- **Moved.** Watermark TPR@1%FPR "degrades from 99.8% to 80.7% and 54.6%" with DIPPER and LLaMA-2-7B-Chat respectively, and to 4.0% under recursive paraphrasing "with a trade-off of 1.5 in the perplexity score". Zero-shot detector AUROC fell from 96.5% to 25.2%. The RoBERTa-Large OpenAI detector fell from 100% to 60% TPR@1%FPR.
- **Did not move (relatively).** Trained detectors degrade each round but "seem to be more robust to paraphrase attacks than the other detectors".
- **Quality.** Human ratings: content preserved in 70% (high) plus 23% (somewhat equivalent) of paraphrases; 89% rated high-quality grammar.
- **Implication.** Each rewrite round costs fluency, measured as perplexity. Bound the number of passes.

### 2.3 Nicks et al. (2024; full author list not verified). *Language Model Detectors Are Easily Optimized Against.* venue: ICLR 2024 (ICLR virtual poster page and OpenReview forum id 4eJDMjYZZG, found by web search). No arXiv ID.
- **Operation.** Fine-tunes a generator with DPO/RL, using a detector's "humanness" score as reward under a KL constraint. Described this way by the paper's abstract (web) and by DAMAGE, HUMPA and Rivera Soto et al.
- **Primary numbers: not extracted.** The OpenReview PDF returned HTTP 403 even after network access was granted. **Primary quantitative results UNVERIFIED.**
- **Secondary, verified numbers.** Rivera Soto et al. (ICML 2026, Table 1) re-implemented the attack. Targeting FastDetectGPT lowered FastDetectGPT AUROC from 65 to 35 and Binoculars from 66 to 34. StyleDetect stayed at 96–97. HUMPA notes that quality drops after this kind of fine-tuning, so β must be set carefully.
- **Implication.** Reward-optimised generators beat the targeted zero-shot detector family. Style embeddings are not affected.

### 2.4 Cheng, Sadasivan, Saberi, Saha, Feizi (2025). *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text.* [arXiv:2506.07001] venue: NeurIPS 2025 (PDF header)
- **Operation.** Training-free. An instruction-tuned LLM paraphrases while a guidance detector picks, at each step, the top-p (p = 0.99) candidate token that looks most human to the detector.
- **Moved.** Average T@1%F fell by 87.88% under OpenAI-RoBERTa-Large guidance. T@1%F fell 64.49% on RADAR and 98.96% on Fast-DetectGPT. Simple paraphrasing gave only a 30.27% relative drop. KGW watermark detection fell 1.000 → 0.769 in one setting *(table)*.
- **Did not move.** RADAR, which is adversarially trained, *improved* its detection rate after simple and recursive paraphrasing.
- **Quality.** 87% of adversarial paraphrases were rated 4 or 5 out of 5. Perplexity: original AI 14.94, human 15.02, simple paraphrase 9.28, adversarial paraphrase 14.26–17.11 *(table)*.
- **Implication.** Naive LLM rewriting lowers perplexity, moving text away from human. The perplexity check must compare against a human interval of the same register.

### 2.5 David & Gervais (2025). *AuthorMist: Evading AI Text Detectors with Reinforcement Learning.* [arXiv:2503.08716] venue UNVERIFIED
- **Operation.** GRPO-fine-tuned Qwen2.5-3B paraphraser. Reward is a commercial or open detector API score.
- **Moved.** Attack success rates of 78.6%–96.2% against individual detectors. The Originality.ai-trained model reached 95.17% mean ASR across detectors.
- **Did not move.** The OpenAI-detector-trained model reached only 32.33% mean ASR across detectors, so transfer depends on the reward detector.
- **Linguistic cost.** Paraphrased texts reach perplexity 300–400, "well beyond typical AI-generated text ranges". Human texts have medians around 50–75.
- **Implication.** RL against detectors pushes text into improbable regions, which is not humanlike. A perplexity ceiling is needed, not only a floor.

### 2.6 Ranganath & Ramesh (2026). *StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors.* [arXiv:2602.08934] venue: PlurVA-LLM workshop @ AACL-IJCNLP 2026 (arXiv comment)
- **Operation.** RL paraphrase policy trained against a detector ensemble with a semantic-fidelity term. Transfer evaluated on held-out detector families (MAGE test pool).
- **Moved.** Mean TPR@1%FPR fell to 0.024 and mean AUROC from 0.79 to 0.43, at 97.6% ASR. TPR@1%FPR on held-out Binoculars was 0.002.
- **Cost.** E5 similarity 0.901 and BERTScore 0.868, but **NLI 0.316** and judged quality 2.51.
- **Implication.** Embedding similarity alone hides meaning drift. Gate on bidirectional entailment.

### 2.7 Wang, Shen, Bu, Zou (2026). *Detector-Evasive LLM Paraphrasing via Constrained Policy Optimization (DEPO).* [arXiv:2606.00392] venue UNVERIFIED
- **Operation.** Constrained policy optimisation: maximise evasion subject to an explicit semantic-preservation constraint, instead of a scalarised reward.
- **Moved.** MAGE ASR rose from 0.01 to 0.73 at BERTScore 0.857. In another setting, ASR was 0.58 on RoBERTa, 0.36 on RADAR, 0.92 on Binoculars and 0.91 on Fast-DetectGPT.
- **Did not move (relatively).** RADAR has the lowest ASR in that setting (0.36).
- **Measurement note.** BERTScore drops more than T5 similarity (8.94% vs 3.90% average relative drop).
- **Implication.** Treat meaning preservation as a hard constraint (accept/reject), not as a weight.

### 2.8 Tommasel & Rodriguez (2026). *Team DArgk at the 2026 ELOQUENT lab … Residuals of Humanity (SHADE).* [arXiv:2609.22221] venue: CLEF 2026 ELOQUENT lab working notes (title/paper)
- **Operation.** GRPO fine-tuning against a surrogate detector, with KL regularisation β.
- **Moved (surrogate).** 98.5% of outputs were classified human (65/66, full fine-tuning, β = 0.05, retry-based selection). One-shot, the rate was 59.1%.
- **Did not transfer.** The authors report that the surrogate advantage "does not fully transfer to the official evaluation setting".
- **Linguistic drift.** Outputs were shorter (374 ± 136 vs 551 ± 26 words) and had fewer sentences, lower TTR and a lower hapax proportion. With β = 0, generations degenerated into repeated tokens.
- **Implication.** Optimising a single detector produces simplification, not humanlike prose.

### 2.9 Wang et al. (2025). *Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors (HUMPA).* [arXiv:2410.19230] venue: ICLR 2025 (PDF header)
- **Operation.** A small LM is DPO-tuned on detector-scored preference pairs. At decoding time, its logit ratio against its own base model is added to a large source model's logits.
- **Moved.** Average AUROC drop of 70.4%, maximum 95.0%. Fast-DetectGPT TPR@1%FPR fell 95.8% (relative). Likelihood and LogRank fell 91.3% and 90.1%.
- **Did not move.** DetectGPT was the most robust detector, with the smallest AUROC decrease.
- **Quality.** Within ΔSBERT ≤ 0.02 and ΔROUGE-1 ≤ 0.03.
- **Implication.** A ≤1.5B proxy model can steer a larger generator. This is relevant only if the skill controls decoding, which it does not for post-hoc rewriting.

### 2.10 Fang et al. (2025). *Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks (CoPA).* [arXiv:2505.15337] venue: EMNLP 2025 (Crossref DOI 10.18653/v1/2025.emnlp-main.433)
- **Operation.** Training-free contrastive decoding. A human-style prompted distribution minus a machine-style prompted distribution, using an off-the-shelf LLM.
- **Moved.** Detector fooling improved by 30.55% on average across three datasets. Against Fast-DetectGPT on GPT-3.5 texts the improvement was 57.72% at FPR = 5%.
- **Did not move.** Under Fast-DetectGPT on GPT-4 texts, DIPPER "even increases the likelihood to be classified as machine-generated".
- **Quality.** Average semantic similarity above 90%.
- **Implication.** Explicitly modelling the machine-like distribution and subtracting it is the training-free analogue of "avoid the typical continuation". A small local model can approximate it.

### 2.11 Pedrotti et al. (2025). *Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors.* [arXiv:2505.24523] venue: Findings of ACL 2025 (Crossref DOI 10.18653/v1/2025.findings-acl.156)
- **Operation.** DPO alignment of generators toward human style, with preference pairs selected on linguistic features.
- **Moved.** MAGE accuracy fell from 76% to 47% (XSUM) and from 77% to 49% (arXiv abstracts). MGT detectors lose 5–35 points on average, up to 60%. MAGE TPR fell from 0.99 at 5% FPR to below 0.2.
- **Implication.** Shifting feature distributions (TTR, clause length, POS) toward human is enough to break trained classifiers. This supports feature-targeted rewriting.

### 2.12 Dugan et al. (2024). *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors.* [arXiv:2405.07940] venue: ACL 2024 (Crossref DOI 10.18653/v1/2024.acl-long.674)
- **Operation.** 11 generators, 4 decoding settings, 11 adversarial attacks.
- **Accuracy changes by attack** *(table; columns: Paraphrase / Synonym / Misspelling / Homoglyph)*:
  - Binoculars: +0.7 / −36.1 / −1.6 / −41.9
  - GLTR: −15.4 / −31.4 / −2.8 / −38.3
  - RADAR: −3.6 / −3.4 / −1.4 / −11.6
  - Originality: +11.7 / +11.5 / −6.4 / −75.7
- **Decoding.** A repetition penalty lowered accuracy by up to 32 points regardless of decoding strategy. Random sampling and repetition penalty both reduce repetitiveness.
- **Implication.** Metric detectors are sensitive to lexical substitution; RADAR is not. Repetition is a real signal, and reducing it is legitimate humanisation.

### 2.13 Masrour, Emi, Spero (2025). *DAMAGE: Detecting Adversarially Modified AI Generated Text.* [arXiv:2501.03437] venue UNVERIFIED
- **Operation.** Audit of 19 commercial humanizers, plus a humanizer the authors fine-tuned through the GPT-4o fine-tuning API, using the detector's AI prediction as a negative signal.
- **Moved / not moved** *(table, before → after humanization)*: GPTZero 99.73% → 60.04%; Binoculars 94.15% → 28.23%; LLM-based baseline 100% → 95.74%; DAMAGE 100% → 98.26%. Against the adversarially trained humanizer, the detector "is still able to detect 93.2%" of samples.
- **Quality.** Humanizer tiers had average win rates of 26.0% (L1), 14.67% (L2) and 2.67% (L3), with fluency damage in the lower tiers.
- **Implication.** Commercial humanizers mostly defeat zero-shot detectors. Detectors trained with humanized hard negatives stay robust.

### 2.14 Russell, Karpinska, Iyyer (2025). *People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text.* [arXiv:2501.15654] venue: ACL 2025 (Crossref DOI 10.18653/v1/2025.acl-long.267)
- **Accuracy by condition** *(table; columns GPT-4o / Claude / GPT-4o paraphrased / o1-pro / o1-pro humanized / overall)*:
  - Expert majority vote: 100 / 100 / 100 / 96.7 / 100 / 99.3
  - Binoculars: 100 / 93.3 / 60.0 / 73.3 / **6.67** / 66.7
  - Fast-DetectGPT: overall 80.0, humanized o1-pro 23.3
  - GPTZero: overall 85.3, humanized o1-pro 46.7
  - RADAR: overall 15.3
- Only Pangram matched the experts. Experts misclassified 1 of 300 articles.
- **Implication.** Prompt-humanized text defeats zero-shot detectors but not trained human readers. Human-judged quality and naturalness belong in the validation, and the skill should not claim success from detector scores alone.

### 2.15 Zheng et al. (2025). *TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors.* [arXiv:2503.08708] venue: KDD 2025 (Crossref DOI 10.1145/3711896.3737418)
- **Moved.** AUC fell 0.913 → 0.185 (Essay), 0.808 → 0.020 (Reuters) and 0.727 → 0 (WP) under the strongest attacks.
- **Did not move, or moved the wrong way.** The prompt attack raised AUC from 0.907 to 0.928 in one setting. Recursive paraphrasing was "ineffective against the OpenAI-detector, even increasing" detection.
- **Cost.** Recursion produced perplexity "nearly four times that of the original". The prompt attack's perplexity stayed "almost identical to the original".
- **Implication.** No attack wins on effectiveness, quality and cost together.

### 2.16 Wang et al. (2024). *Stumbling Blocks: Stress Testing the Robustness of MGT Detectors Under Attacks.* [arXiv:2402.11638] venue: ACL 2024 (Crossref DOI 10.18653/v1/2024.acl-long.160)
- **Moved.** GLTR, Rank and LogRank drop 51.35% on average under all attacks, "especially not robust to editing, paraphrasing, and co-generating attacks".
- **Did not move.** Watermarking keeps AUC-ROC above 99%. The SimpleAI detector stays robust under all attacks.

### 2.17 Zhou et al. (2024). *Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack (HMGC).* [arXiv:2404.01907] venue: LREC-COLING 2024 (Crossref record)
- **Operation.** Detector-guided word substitution under a Universal Sentence Encoder similarity constraint.
- **Moved.** Detector accuracy fell from 99.63% to 51.06%. Removing the USE constraint raised ASR by about 1% and perplexity by 0.2.

### 2.18 Lu et al. (2024). *Large Language Models can be Guided to Evade AI-Generated Text Detection (SICO).* [arXiv:2305.10847] venue: TMLR 05/2024 (PDF header)
- **Operation.** In-context substitution and generation prompts built from detector feedback.
- **Moved.** AUC decreased "by 0.5 on average".
- **Cost.** DIPPER outputs showed a 0.27 readability decline relative to human text.

### 2.19 Rivera Soto, Chen, Andrews (2026). *Attacks on Machine-Text Detectors Retain Stylistic Fingerprints.* [arXiv:2505.14608] venue: ICML 2026, PMLR 306 (PDF header)
- **Finding** *(Table 1, AUROC)*. Across evasion strategies, the few-shot style-embedding detector (StyleDetect) averaged 95, FastDetectGPT 66 and Binoculars 70. DIPPER *raised* the zero-shot detectors' AUROC (FastDetectGPT 71 → 81, Binoculars 79 → 83) but lowered StyleDetect from 98 to 89. TinyStyler gave StyleDetect 87.
- **Operation that did move style.** A style-aware paraphraser (SFT on human exemplars plus DPO for undetectability) evades all detectors on single documents. Detection recovers as the number of documents grows.
- **Implication.** The hard signal is authorial style. Conditioning on human exemplars, ideally the user's own prose, is what moves it. Validation should include a style-embedding check and a multi-document check.

### 2.20 Karr, Khvatskii, Hua, Chawla (2026). *Why AI Detection Fails for Academic Integrity.* [arXiv:2608.11256] venue: ACM AI Leadership Summit (arXiv comment; proceedings not confirmed)
- **Findings.** After Undetectable AI humanization, fewer than 4% of AI-labelled rewrites remain flagged. Light "refine abstract only" edits are flagged 38–80%. Unmodified 2023–2025 abstracts are flagged 9–15%, rising with long-token and Academic Word List density; Pangram flags 2.1% in chemistry vs 26.6% in theology.
- **Linguistic change.** Across n = 2,568 humanized pairs, long-token ratio, AWL density and lexical diversity all fall (p < 0.001).
- **Implication.** Commercial humanization works by simplifying and flattening vocabulary. Register-driven false positives mean the check round must use register-matched human baselines.

### 2.21 Perkins et al. (2024). *GenAI Detection Tools, Adversarial Techniques and Implications for Inclusivity in Higher Education.* [arXiv:2403.19148] venue: Int. J. Educational Technology in Higher Education 21:53 (DOI 10.1186/s41239-024-00487-w, arXiv journal_ref)
- **Operation.** Prompted adversarial techniques against 7 detectors, including Turnitin, Copyleaks and GPT Kit.
- **Moved.** Outputs with spelling errors (12.9% detected) or "higher burstiness (15.9%) were almost undetectable". Turnitin had the largest drop (42.1%).
- **Did not move.** Increasing complexity was least effective, with only a 2% drop.
- **Implication.** Sentence-length variation is a known lever. Spelling errors are out of scope.

### 2.22 Liang, Yuksekgonul, Mao, Wu, Zou (2023). *GPT detectors are biased against non-native English writers.* [arXiv:2304.02819] venue: Patterns (DOI 10.1016/j.patter.2023.100779)
- **Findings.** TOEFL essays were misclassified as AI at an average FPR of 61.22%. Prompting for richer word choice cut the FPR to 11.77%. Literary-language prompting cut detection of ChatGPT essays "from up to 100% to up to 13%".
- **Implication.** Perplexity detectors key on lexical sophistication, and plain human prose is penalised. The skill must not treat low detector risk as equivalent to human-like text.

### 2.23 Tulchinskii et al. (2023). *Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts.* [arXiv:2306.04723] venue: NeurIPS 2023 (PDF header)
- **Finding.** Human text has an average intrinsic dimension of about 9 for alphabetic languages and about 7 for Chinese. AI text is "≈ 1.5 lower".
- **Robustness.** Under DIPPER, the PHD classifier's accuracy "even increases slightly". Raising DIPPER lexical diversity slightly raises mean PHD and MLE; reordering has almost no effect.
- **Implication.** This is a cheap embedding-geometry check (RoBERTa-base, ≥ ~200 tokens). If PHD does not move, the rewrite was cosmetic.

### 2.24 Wang, Li, Yan, Cheng, Zhang (2025). *Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing.* [arXiv:2502.15208] venue UNVERIFIED
- **Finding.** Successive paraphrasing "converges to stable periodic states, such as 2-period attractor cycles". The pattern persists with higher randomness and with alternating prompts or LLMs.
- **Implication.** Iterative LLM rewrite loops oscillate instead of progressively humanising. The skill should stop after a fixed number of passes and select the best pass by measured signals.

### 2.25 Supporting detection and theory papers (verified identity; used for the check round)
- Mitchell et al. (2023) DetectGPT [arXiv:2301.11305]; Bao et al. Fast-DetectGPT [arXiv:2310.05130]; Hans et al. Binoculars [arXiv:2401.12070]. Venues for these three: UNVERIFIED.
- Verma et al. Ghostbuster [arXiv:2305.15047], NAACL 2024 (Crossref DOI 10.18653/v1/2024.naacl-long.95).
- Hu, Chen, Ho. RADAR [arXiv:2307.03838], NeurIPS 2023 (Crossref record).
- Gehrmann, Strobelt, Rush. GLTR [arXiv:1906.04043]: the classifier "learns that real text samples from the tail of the distribution more frequently". Venue UNVERIFIED.
- Mao et al. Raidar [arXiv:2401.12970]: LLMs "are more likely to modify human-written text than AI-generated text when tasked with rewriting". Edit distance under LLM rewriting is a usable local check. Venue UNVERIFIED.
- Li et al. Learning to Rewrite [arXiv:2408.04237]. Venue UNVERIFIED.
- Emi & Spero. Pangram technical report [arXiv:2402.14873]: hard-negative mining "reduces false positive rates by 100x-1000x on holdout sets".
- Chakraborty et al. *On the Possibilities of AI-Generated Text Detection* [arXiv:2304.04736]: the sample size needed for detection grows as machine text approaches human text. Venue UNVERIFIED.
- Ponde, Das, More, Gupta (2026). *On the Indistinguishability of Human v/s AI Generated Text* [arXiv:2608.26797]: repeated paraphrasing guided by human samples converges toward the human distribution under mixing and stability conditions. Venue UNVERIFIED.
- Weber-Wulff et al. (2023). *Testing of detection tools for AI-generated text*, Int. J. Educational Integrity [DOI:10.1007/s40979-023-00146-z]. Tools are "neither accurate nor reliable", and obfuscation "significantly" worsens them.
- Verified but not analysed (no full-text extraction): Shi et al., *Red Teaming LM Detectors with LMs*, TACL 2024 [arXiv:2305.19713; DOI 10.1162/tacl_a_00639]; Meng et al., GradEscape [arXiv:2506.08188]; Ayoobi et al., ESPERANTO [arXiv:2409.14285]; Thai et al., EditLens [arXiv:2510.03154]; Reddy & Stamp [arXiv:2506.12843]; Dycke et al. [arXiv:2606.04906]; Ahmed & Hammond, DependencyAI [arXiv:2602.15514]; Xu & Zubiaga, RLHF effects on detectability [arXiv:2503.17965].

## 3. Character-level tricks: detector gaming, out of scope

These are recorded so the skill can exclude them and the check round can detect accidental ones:
- **Homoglyph substitution.** SilverSpeak [arXiv:2406.11239] and the RAID homoglyph attack: Originality −75.7 points, Binoculars −41.9 *(table)*. StealthRL notes such attacks "are detectable by normalization". DEPO uses a substitution-rate-0.3 homoglyph baseline.
- **Zero-width / whitespace insertion, article deletion.** RAID whitespace and article-deletion attacks.
- **Misspelling injection.** RAID misspelling attack; Perkins et al. spelling-error prompts (12.9% detected).
- **Skill rule.** The rewrite must not introduce non-ASCII look-alikes, zero-width characters (U+200B–U+200D, U+FEFF), irregular whitespace or deliberate typos. The check script should Unicode-NFKC-normalise and flag any of these.

---

## 4. Properties that separate human from LLM prose

### 4.1 Register and grammatical features (Biber multidimensional features)
**Reinhart, Markey, Laudenbach, Pantusen, Yurko, Weinberg, Brown (2025).** *Do LLMs write like humans? Variation in grammatical and rhetorical styles.* PNAS 122, e2422455122 [DOI:10.1073/pnas.2422455122; arXiv:2410.16107]
- **Data.** Parallel corpora HAP-E (6 genres, 8,290 texts) and CAP (COCA registers). Models: GPT-4o, GPT-4o Mini, Llama 3 8B/70B (base and Instruct).
- **GPT-4o vs human.**
  - Present participial clauses at 5.3× the human rate (d = 1.38)
  - That-clauses as subject 2.6× (d = 0.77)
  - Nominalizations 2.1× (d = 1.23)
  - Phrasal coordination 1.9× (d = 0.81)
  - Agentless passives at roughly half the human rate
  - Downtoners more often (GPT-4o); clausal coordination avoided (GPT-4o) but over-used (Llama 3)
- **Instruction-tuned models in general.** Participial clauses at 2–5× and nominalisations at 1.5–2× the human rate.
- **Separability.** Random forests on the 66 Biber features reach 93–98% accuracy across corpora.
- **Mechanism.** Base models match human feature rates. Instruction tuning "appears to make the model output less human, not more". Vocabulary: camaraderie 162×, tapestry 155×, intricate 119× (GPT-4o); tapestry appears in 23% and amidst in 27% of GPT-4o outputs. The authors read these words as "a preference for grandiose, if hollow, summative sentences" and note they are conspicuous mainly *outside* fiction.

**Herbold et al. (2023).** *A large-scale comparison of human-written versus new ChatGPT-generated essays.* [arXiv:2304.14276] venue UNVERIFIED
- ChatGPT uses more nominalisations and more complex sentences.
- Humans "make more use of modal and epistemic constructions".
- ChatGPT-4 has higher lexical diversity than humans; ChatGPT-3 has lower.
- ChatGPT-4 uses fewer discourse markers.
- All ChatGPT essays share an identical conclusion opening ("In conclusion, [...]").

**Muñoz-Ortiz, Gómez-Rodríguez, Vilares (2024).** *Contrasting Linguistic Patterns in Human and LLM-Generated News Text.* [arXiv:2308.09067] venue: Artificial Intelligence Review 57 (arXiv journal_ref)
- Sentence length: models generate sentences concentrated in the 10–30 token range more often than humans.
- Lexical diversity (MTLD): human 96.51 vs 57.37–94.56 across six models; STTR: human 0.491 vs 0.424–0.466 *(table)*.
- Parts of speech: humans use more nouns and adjectives; LLMs use more pronouns, numbers and symbols.
- Dependencies: human dependency lengths are more optimised (shorter than expected). LLMs use more SBAR subordinate clauses and fewer amod/appos modifiers.

**Clark, Martin, Mahowald et al. (2026).** *Why is AI-generated language so hard to understand?* [arXiv:2610.29162] venue UNVERIFIED
- Most models use "longer dependencies, more stacked nouns, and more reduced object relatives". They prefer the no-relativizer variant.
- In RL experiments without KL regularisation, the short variant was chosen "over 98%" of the time, "unlike human speakers, who choose between the variants by context".

**Suvanto, McGlinchey, Wahde, Barclay (2026).** ICAART 2026 [DOI:10.5220/0014236000004052; arXiv:2601.07368]
- Data: LLM rewrites of novel excerpts.
- Humans perform at chance (0.499). Unigram classifiers reach 0.93–0.98.
- The LLM uses a larger variety of synonyms (entropy +0.29 bits) and more participles. Also reported: temporal drift ("exited": 0 human vs 61 LLM occurrences), Americanisms, foreign-language usage and colloquialisms.

### 4.2 Syntactic templates
**Shaib, Elazar, Li, Wallace (2024).** *Detection and Measurement of Syntactic Templates in Generated Text.* [arXiv:2407.00211] venue: EMNLP 2024 (Crossref DOI 10.18653/v1/2024.emnlp-main.368)
- 95% of model outputs contain length-6 POS templates vs 38% of human references.
- 75% of OLMo templates occur in the pre-training data, vs 34% of non-templated sequences.
- Template rates stay at 96.6–97.4% across temperature and top-p settings on Rotten Tomatoes *(table)*.
- Companion tool: Shaib et al., *Standardizing the Measurement of Text Diversity* [arXiv:2403.00553], AACL 2025 (arXiv comment). Recommended reporting set: compression ratio, self-repetition of long n-grams, Self-BLEU and BERTScore.

### 4.3 Surprisal, UID and probability profile
- **Holtzman et al. (2020).** *The Curious Case of Neural Text Degeneration* [arXiv:1904.09751], ICLR 2020 (PDF header). Human text shows "increased variance" in per-token probability. Text from maximisation or top-k "is too probable".
- **Gehrmann et al. GLTR** [arXiv:1906.04043]. Human text samples from the tail (outside the top-100 ranks) more often.
- **Venkatraman, Uchendu, Lee (2024).** *GPT-who* [arXiv:2310.06202], Findings of NAACL 2024 (PDF header). "UID scores of human-written text have a higher mean and larger standard deviation". Machines spread information "more evenly or smoothly".
- **Meister et al. (2021).** *Revisiting the Uniform Information Density Hypothesis* [arXiv:2109.11635], EMNLP 2021 (journal_ref). Gives the formal UID operationalisations (variance and super-linear surprisal cost) usable as metrics.
- **Goh & Barabási (2008).** *Burstiness and memory in complex systems*, EPL 81, 48002 [DOI:10.1209/0295-5075/81/48002]. Defines the burstiness coefficient B = (σ − μ)/(σ + μ), applicable to sentence lengths or inter-arrival gaps of a word.
- **McCarthy & Jarvis (2010).** *MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment*, Behavior Research Methods 42(2) [DOI:10.3758/brm.42.2.381]. The measurement reference for length-robust lexical diversity (MTLD, HD-D). The skill should use these instead of raw TTR. (Identity verified via Crossref; full text not read.)
- **Tarım & Onan (2025).** *Can You Detect the Difference?* [arXiv:2507.10475] venue UNVERIFIED. Diffusion-LM text (LLaDA) "closely mimics human text in perplexity and burstiness", and no single metric separates it from human text. Single statistics are insufficient on their own.

### 4.4 Lexical over-representation ("excess vocabulary") and why it happens
- **Kobak, González-Márquez, Horvát, Lause (2025).** Science Advances 11(27) [DOI:10.1126/sciadv.adt3813; arXiv:2406.07016]. PubMed 2010–2024, excess usage against a counterfactual extrapolation.
  - delves r = 28.0, underscores r = 13.8, showcasing r = 10.7
  - potential δ = 0.052, findings δ = 0.041, crucial δ = 0.037
  - 66% of the 379 excess style words in 2024 were verbs and 14% adjectives
  - Lower bound of LLM-processed abstracts in 2024: 13.5%, ranging from below 5% to over 40% across fields, countries and journals
  - 291-word rare set and 10-word common set → `ai_vocabulary.json`
- **Juzek & Ward (2025).** COLING 2025 [arXiv:2412.11385]. Twenty-one focal words, e.g. delves 0.21 → 14.38 per million (+6697.14%), underscores +903.61%, intricate +611.24%. Human raters showed a small preference for delve-initial abstracts (coefficient 0.082, p = 0.023) and no significant preference for the other focal words. This is partial support for the hypothesis that human-feedback learning amplifies these words.
- **Liang et al. (2024).** ICML 2024, PMLR 235 [arXiv:2403.07183]. In ICLR 2024 reviews, commendable, meticulous and intricate show 9.8-, 34.7- and 11.2-fold increases in the probability of occurring in a sentence. Top-100 adjective and adverb lists. Estimated AI-modified share rose from 1.6% to 10.6% (ICLR) and was about 16.9% at EMNLP; Nature Portfolio journals showed no significant rise. Higher in reviews submitted near deadlines.
- **Liang et al. (2024).** *Mapping the Increasing Use of LLMs in Scientific Papers* [arXiv:2404.01268] venue UNVERIFIED. Realm, intricate, showcasing and pivotal are the top-4 words by log-odds. Estimated α for CS abstracts reached 17.5%.
- **Gray (2024).** [arXiv:2403.16887] arXiv preprint. Intricate +117%, commendable +83%, meticulous +59%, meticulously +137% (2022 → 2023). Estimated excess of 60,000–85,000 papers in 2023.
- **Geng & Trotta (2024).** [arXiv:2404.08627] arXiv preprint. In simulated ChatGPT revision, "significant" rose 99% (cs) to 308% (math) and "is"/"are" fell 14%/17% (cs).
- **Matsui (2025).** Perspectives on Medical Education [DOI:10.5334/pme.1929]. 103 of 135 candidate AI-influenced terms rose (modified Z ≥ 3.5) in 2024, led by "delve", "underscore", "primarily", "meticulous" and "boast". These terms began rising in 2020, before ChatGPT. (Crossref abstract only.)
- **Paech, Roush, Goldfeder, Shwartz-Ziv (2025).** *Antislop* [arXiv:2510.15061] arXiv preprint.
  - Over-use ratios: "Elara" 85,513×, "heart hammered ribs" 1,192× (gemma-3-12b creative writing); "It's not X, it's Y" 6.3× in some models
  - "flickered" is in the top over-represented words of 98.5% of 67 models
  - Their FTPO fine-tuning suppresses 94.4% of a 1,000-item banlist with under 1% writing-quality loss; DPO reduces diversity (74–92% of baseline); token banning drops quality to 28/100 at 8k banned patterns
- **Sun, Yin, Xu, Kolter, Liu (2025).** *Idiosyncrasies in Large Language Models*, ICML 2025 (arXiv comment) [arXiv:2502.12150]. Five chat models can be told apart at 97.1% accuracy. Paraphrased or translated outputs remain classifiable at similar accuracy, and word-shuffled text still does well. **Implication.** Paraphrasing with one LLM swaps one fingerprint for another.

### 4.5 Content homogenisation and diversity
- **Padmakumar & He (2024).** *Does Writing with Language Models Reduce Content Diversity?* [arXiv:2309.05196], ICLR 2024 (PDF header). Corpus homogenisation: 0.1536 (solo), 0.1578 (GPT-3), 0.1660 (InstructGPT). Key points contributed by InstructGPT are more homogeneous. The model contributes about 35% of the characters.
- **Doshi & Hauser (2024).** Science Advances [DOI:10.1126/sciadv.adn5290]. AI-assisted stories are rated more creative but are "more similar to each other".
- **Anderson, Shah, Kreminski (2024).** C&C 2024 [arXiv:2402.01536; DOI:10.1145/3635636.3656204]. Homogenisation of ideation.
- **Kirk et al. (2024).** [arXiv:2310.06452] venue UNVERIFIED. "RLHF significantly reduces output diversity compared to SFT".
- **Guo et al. (2024).** Findings of NAACL 2024 [arXiv:2311.09807; Crossref DOI 10.18653/v1/2024.findings-naacl.228]. Diversity declines over recursive training on synthetic text.
- **Sourati, Ziabari, Dehghani (2025).** [arXiv:2508.01491] venue UNVERIFIED. **Uhlmann (2025)**, *The Even Sheen of AI* [arXiv:2509.16794] venue UNVERIFIED. Both are conceptual or review treatments of homogenisation.
- **Kim, Chang, Pham, Iyyer (2026).** *Argument Collapse: LLMs Flatten Long-Form Public Debate* [arXiv:2606.01736] venue UNVERIFIED.
  - Unique main arguments: 65.3% human vs 3.4% vanilla LLM; in BR essays, 78.6% vs 18.4%
  - Unique sub-arguments: 41.0% human vs 9.1% LLM
  - Strong stance: 76.1% human vs 63.4% LLM
  - Support → proposal transitions: 29.4% (LLM) vs 12.3% (human)
  - Diversified prompting recovers only 50–55% of human argument coverage; argumentation still fills 89.7% of paragraphs vs 71.5% for humans

### 4.6 Discourse organisation, narrative and specificity
- **Russell, Rajendhran, Pham, Iyyer, Wieting (2026).** *StoryScope* [arXiv:2604.03136], COLM 2026 (PDF header). 10,272 prompts × 6 authors.
  - AI narrators explain the theme 77% of the time (human 52%)
  - AI dialogue serves philosophical debate more often (59% vs 34%)
  - AI uses vague allusions more (72% vs 50%); humans reference specific texts and authors more (47% vs 24%)
  - Humans use explicit emotion labels more (29% vs 8%), break the fourth wall more (67% vs 39%) and write morally ambivalent protagonists more (59% vs 38%)
  - Human narrative rarity percentile 0.71 vs 0.49 (d = 0.83)
  - After LAMP-style span-level artifact removal, detection was still 93.9% macro-F1, a drop of 1.6 points
- **Madler (2026).** *SlopShape* [arXiv:2609.15369] venue UNVERIFIED. Commercial web content. The structural model beats the style model by 8.8 macro-F1. Detection is nearly unchanged (96.1) when posts are reworded by their own model. Human rarity percentile 0.835 vs 0.436 for AI (d = 1.80). AI conclusions restate the thesis or reframe it.
- **Jiang & Hyland (2024).** *Does ChatGPT Argue Like Students? Bundles in Argumentative Essays*, Applied Linguistics [DOI:10.1093/applin/amae052]. ChatGPT uses fewer bundles but more rigid ones, and more noun- and preposition-based bundles. Students show more epistemic stance and authorial presence. (Crossref abstract.)
- **Jiang & Hyland (2025).** *Does ChatGPT Write Like a Student? Engagement Markers in Argumentative Essays*, Written Communication [DOI:10.1177/07410883251328311]. Across 145 essays per group, ChatGPT used fewer engagement markers, "particularly questions and personal asides". (Crossref abstract.)
- **Shaib et al. (2025).** *Measuring AI "Slop" in Text* [arXiv:2509.19163], COLM 2026 (PDF header). Annotators labelled 36% of articles as slop. The strongest predictors are relevance, density and tone. Syntax-only features reach AUROC 0.64–0.68. LLM judges under-predict slop (0.03–0.08 vs 0.35 for humans).

### 4.7 Creative writing and expert editing
- **Chakrabarty, Laban, Agarwal, Muresan, Wu (2024).** *Art or Artifice?* CHI 2024 [DOI:10.1145/3613904.3642731; arXiv:2309.14556]. LLM stories pass 9% (ChatGPT) to 30% (Claude) of TTCW tests; New Yorker stories were most preferred 89% of the time. LLM assessments do not correlate with experts. Experts noted repeated pet words ("tendrils", "inky darkness") and metaphors that "fell flat".
- **Chakrabarty, Laban, Wu (2025).** *Can AI writing be salvaged?* (LAMP), CHI 2025 [DOI:10.1145/3706598.3713559; arXiv:2409.14509].
  - Edit categories: awkward word choice (28%), poor sentence structure (20%), unnecessary exposition (18%), clichés (17%)
  - 74% of edits are replacements; 70% of non-deletion edits preserve meaning
  - "unspoken" occurs in about 15% of LLM paragraphs
  - Writer-edited text was most preferred 65% of the time; LLM-edited text beat LLM-generated text (p ≈ 1e-11)
  - Few-shot LLM span detection reaches precision 0.46, below expert agreement (0.57)

## 5. Coverage gaps and unverified items
- **Hedging and stance in LLM vs human academic prose (2024–2026).** No dedicated corpus study was found by arXiv search, and Crossref did not return a matching LLM-specific study. Evidence here comes from Herbold, Jiang & Hyland and Argument Collapse.
- **Zipf / rank-frequency profile of LLM text.** No dedicated study was found. Indirect evidence: GLTR rank buckets, Suvanto synonym entropy, Kobak/Antislop over-use ratios. The skill's Zipf check is therefore an engineering heuristic, not a literature-backed effect.
- **Nicks et al. (ICLR 2024).** Primary quantitative results not extracted (PDF not retrievable).
- **Evaluations of commercial humanizers** come from DAMAGE (19 tools), Karr et al. (Undetectable AI) and Russell et al. (a humanized o1-pro condition). No peer-reviewed multi-service benchmark beyond these was found.
- **Venues not confirmed:** listed inline as "venue UNVERIFIED".
