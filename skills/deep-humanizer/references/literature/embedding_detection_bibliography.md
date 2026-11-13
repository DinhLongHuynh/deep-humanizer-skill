# Distribution- and embedding-level detection of LLM-generated text: annotated bibliography

Prepared for the deep-humanizer skill (research track: distribution/embedding-level detection). Compiled 13 November 2026.

## How entries were verified

- **arXiv IDs**: every listed ID was fetched through the arXiv API (batch metadata lookup); the title printed below is the title returned by arXiv, so ID and title match by construction. 108 of 108 arXiv IDs resolved.
- **DOIs**: resolved through the Crossref REST API (title and container matched by inspection) or taken from the arXiv record's own DOI field. Proceedings DOIs for NeurIPS 2023-2025 use the Curran 10.52202 prefix as returned by Crossref.
- **Venues**: taken from Crossref container titles, the PDF header/footer of the arXiv version, the authors' arXiv comment field, or (Lastde, DNA-GPT) a web search hitting the conference's own pages. The source is given per entry. 'Venue not verified' means only the arXiv preprint was confirmed.
- **Numbers**: quantitative results come from the abstract or from text extracted from the paper PDF (25 arXiv PDFs plus the Text Fluoroscopy ACL Anthology PDF were searched). An automated pass confirmed that every number in every summary occurs in that paper's abstract or extracted PDF text, apart from DOIs and the Reinhart chunk counts, which were checked by eye. Nothing was filled in from memory; where an item rests on memory or title only, it is labelled UNVERIFIED in the entry.
- Entries in sections A-C and the key items of D, E and G have full summaries; peripheral items (surveys, datasets, defences) have shorter summaries.

Direction convention used throughout: 'AI-like' means the value typical of instruction-tuned LLM output; 'human-like' means the value typical of the human reference texts in the cited study.



## A. Representation geometry, embedding-space and hidden-state detectors

### 1. Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts

Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Serguei Barannikov, Irina Piontkovskaya, Sergey Nikolenko, and Evgeny Burnaev (2023). *Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts*. NeurIPS 2023 (Advances in NeurIPS 36).

- **Identifiers**: arXiv: [2306.04723](https://arxiv.org/abs/2306.04723) · DOI: [10.52202/075280-1706](https://doi.org/10.52202/075280-1706) (verified: Crossref DOI + PDF footer)
- **Code/data**: https://github.com/ArGintum/GPTID
- **Summary**: Treats the contextual token embeddings of one text as a point cloud and estimates its intrinsic dimension. Recipe: RoBERTa-base last-layer token embeddings for English (XLM-R base for other languages), <CLS>/<SEP> dropped; persistent-homology dimension (PHD) from the growth rate of the 0-dimensional persistence score (minimum-spanning-tree edge-length sum, weight alpha = 1.0) over random subsets of sizes n_i, i.e. regress log E(S_i) on log n_i and set PHD = 1/(1 - slope). The abstract reports that fluent human text has average PHD around 9 for several alphabet-based languages (about 7 for Chinese) and that AI text is about 1.5 lower; in the body, English human text lies mostly between 9 and 10 and generated text near 8 regardless of generator. Accuracy at 1% FPR is modest (25-40% on Wikipedia prompts; e.g. GPT-3.5: 40.0) but survives DIPPER paraphrasing (GPT-3.5 + DIPPER: 41.2) while DetectGPT and GPTZero collapse; in their DIPPER ablation the mean PHD of generated text moves only from 7.30 to 7.51. False-positive rate on TOEFL essays by non-native writers was 26% for PHD versus 58% (OpenAI) and 52% (GPTZero). Absolute values depend on the estimator: in their estimator-comparison table (human text) PHD gave 9.21-9.59, MLE 11.55-12.13 and TwoNN 5.70-5.95; MLE detects as well in-distribution but is less robust to paraphrase.
- **Implication for the humanizer**: PHD is a whole-text geometric property that lexical paraphrase barely moves, so it is the most direct test of whether a rewrite changed the 'shape' of the text rather than its words. It runs on CPU with roberta-base (125M). Compare against a human reference distribution of the same register and length, never against a fixed number, and never mix estimators. Texts under ~200 tokens give unstable estimates (see Short-PHD).

### 2. Short-PHD: Detecting Short LLM-generated Text with Topological Data Analysis After Off-topic Content Insertion

Dongjun Wei, Minjia Mao, Xiao Fang, and Michael Chau (2025). *Short-PHD: Detecting Short LLM-generated Text with Topological Data Analysis After Off-topic Content Insertion*. arXiv preprint (2025); venue not verified.

- **Identifiers**: arXiv: [2504.02873](https://arxiv.org/abs/2504.02873) (verified: arXiv)
- **Code/data**: https://github.com/djwei96/ShortPHD
- **Summary**: Shows that PHD estimation on short texts suffers from 'local density peaks' and notes that Tulchinskii et al. assume generations of at least 200 tokens. Short-PHD inserts the same off-topic content before every input (12 insertions, LLaMA-3-8B embeddings in their setup) so that human and machine texts are shifted equally while the estimate stabilizes, then thresholds the PHD. At 50 tokens the AUC rises from 0.632 (OPT/OPT-IML/GPT-NeoX) and 0.653 (GPT-3.5/GPT-4o) for plain PHD to 0.793 and 0.830.
- **Implication for the humanizer**: Plain PHD should not be reported below roughly 200 tokens; the skill should either abstain or use a fixed off-topic padding protocol that is identical for the reference and the candidate text.

### 3. Artificial Text Detection via Examining the Topology of Attention Maps

Laida Kushnareva, Daniil Cherniavskii, Vladislav Mikhailov, Ekaterina Artemova, Serguei Barannikov, Alexander Bernstein, et al. (2021). *Artificial Text Detection via Examining the Topology of Attention Maps*. EMNLP 2021.

- **Identifiers**: arXiv: [2109.04825](https://arxiv.org/abs/2109.04825) · DOI: [10.18653/v1/2021.emnlp-main.50](https://doi.org/10.18653/v1/2021.emnlp-main.50) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Extracts topological features (persistence barcodes, graph statistics) from BERT attention maps and trains a linear classifier. These features outperform count- and neural-based baselines by up to 10% on three datasets and are the most robust to unseen GPT-style generators; probing shows sensitivity to surface and syntactic properties.
- **Implication for the humanizer**: Earlier evidence that topology of internal representations carries an authorship signal. Heavier than PHD on token embeddings; low priority for a CPU toolkit.

### 4. AI-generated text boundary detection with RoFT

Laida Kushnareva, Tatiana Gaintseva, German Magai, Serguei Barannikov, Dmitry Abulkhanov, Kristian Kuznetsov, et al. (2023). *AI-generated text boundary detection with RoFT*. arXiv preprint (2023); venue not verified.

- **Identifiers**: arXiv: [2311.08349](https://arxiv.org/abs/2311.08349) (verified: arXiv)
- **Code/data**: https://github.com/SilverSolver/ai_boundary_detection
- **Summary**: Studies boundary detection in texts that start human-written and switch to machine-generated (RoFT data). Perplexity-based approaches are more robust to domain-specific peculiarities than fine-tuned RoBERTa, and the paper analyses which text properties confuse boundary detection.
- **Implication for the humanizer**: Mixed human/AI documents need localisation, not a single score. The humanizer's check round should score sentences or windows so that residual machine-like spans can be targeted.

### 5. Emergence of a High-Dimensional Abstraction Phase in Language Transformers

Emily Cheng, Diego Doimo, Corentin Kervadec, Iuri Macocco, Jade Yu, Alessandro Laio, and Marco Baroni (2025). *Emergence of a High-Dimensional Abstraction Phase in Language Transformers*. ICLR 2025.

- **Identifiers**: arXiv: [2405.15471](https://arxiv.org/abs/2405.15471) (verified: arXiv comment)
- **Code/data**: none found in paper/abstract
- **Summary**: Measures the intrinsic dimension of hidden representations across layers of several language models and finds a high-dimensional 'abstraction phase' in intermediate layers. Representations in this phase are the first full linguistic abstraction of the input, transfer best to downstream tasks and predict each other across models; an earlier onset predicts better language-modelling performance.
- **Implication for the humanizer**: If the skill computes embedding geometry from a decoder LM rather than RoBERTa, intermediate layers near the ID peak are the principled choice; this agrees with Text Fluoroscopy's preference for middle layers.

### 6. Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features

Xiao Yu, Kejiang Chen, Qi Yang, Weiming Zhang, Nenghai Yu (2024). *Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features*. EMNLP 2024.

- **Identifiers**: DOI: [10.18653/v1/2024.emnlp-main.885](https://doi.org/10.18653/v1/2024.emnlp-main.885) (verified: Crossref DOI)
- **Code/data**: https://github.com/Fish-and-Sheep/Text-Fluoroscopy
- **Summary**: Argues that last-layer features are too semantic (they overfit the training domain) and first-layer features are too shallow (fragile to paraphrase). For each text it projects every layer to a next-token distribution and selects the intermediate layer M with the largest summed KL divergence from the first and the last layer; that layer's embedding (encoder: gte-Qwen1.5-7B-instruct) feeds a 3-layer MLP (hidden sizes 1024 and 512, tanh). In their Table 1, AUROC is 0.996-0.9998 on ChatGPT, GPT-4 and Claude3 text, both original and after DIPPER paraphrase or back-translation, whereas Fast-DetectGPT falls to 0.806 on GPT-4 + DIPPER (supervised, in-distribution setting).
- **Implication for the humanizer**: Supports measuring the human-AI gap on intermediate-layer embeddings rather than on the final layer. A CPU adaptation (e.g. gte-Qwen2-1.5B-instruct or Qwen2.5-0.5B with the same KL layer selection) is plausible but untested.

### 7. RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns

Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Ziyang Luo, et al. (2025). *RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns*. TACL 2025.

- **Identifiers**: arXiv: [2508.13152](https://arxiv.org/abs/2508.13152) · DOI: [10.1162/tacl.a.61](https://doi.org/10.1162/tacl.a.61) (verified: Crossref DOI + arXiv comment)
- **Code/data**: https://github.com/NLP2CT/RepreGuard
- **Summary**: Feeds human and LLM texts through a surrogate LLM, observes systematically different activation patterns, extracts a discriminative activation direction from a small labelled set, and classifies a text by the projection of its representation onto that direction against a threshold. Reports 94.92% average AUROC across in-distribution and OOD settings, with robustness to text length and mainstream attacks. Directions are obtained per layer by PCA on paired activation differences (LLM text minus human text); a text's score is the projection of its token activations onto these directions, and LLM text shows a higher overall activation level. In the surrogate ablation Llama-3.1-8B was the chosen surrogate (94.74% AUROC), phi-2 (2.7B) reached 94.14% and Gemma-2B-Instruct 92.75%, instruction-tuned surrogates generally beat their base versions, and GPT-Neo-2.7B was at chance (49.73%).
- **Implication for the humanizer**: A 'machine-ness direction' in a small LM's residual stream is a cheap, interpretable gap metric: the check round can report the projection score of the original and the humanized text relative to human reference texts. Small (2-3B) instruction-tuned surrogates are competitive, but surrogate choice can make the method fail entirely (GPT-Neo-2.7B), so a sub-2B surrogate must be validated locally.

### 8. SV-Detect: AI-generated Text Detection with Steering Vectors

Mikhail Vishnyakov and Tatiana Gaintseva (2026). *SV-Detect: AI-generated Text Detection with Steering Vectors*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2606.07313](https://arxiv.org/abs/2606.07313) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Builds, at every layer of a frozen LM, a steering-vector direction separating human from AI text and represents each input by its layer-wise alignment with these directions; a light classifier on the projection features gives the score. Reports strong in-distribution and shift performance, including polishing and rewriting edits, and finds that the directions align with recognisable stylistic cues plus additional signal.
- **Implication for the humanizer**: Generalises RepreGuard to a per-layer profile; the skill could log a projection profile across layers to see at which depth a humanized text still looks machine-like.

### 9. DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning

Xun Guo, Shan Zhang, Yongxin He, Ting Zhang, Wanquan Feng, Haibin Huang, and Chongyang Ma (2024). *DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning*. NeurIPS 2024.

- **Identifiers**: arXiv: [2410.20964](https://arxiv.org/abs/2410.20964) · DOI: [10.52202/079017-2802](https://doi.org/10.52202/079017-2802) (verified: Crossref DOI + arXiv comment)
- **Code/data**: https://github.com/heyongxin233/DeTeCtive
- **Summary**: Recasts detection as distinguishing writing styles: a text encoder is trained with multi-level contrastive learning (human vs AI, model family, specific model) and classification is done by K-nearest-neighbour retrieval against a database of encoded training texts. Training-Free Incremental Adaptation (TFIA) adds OOD examples to the database without retraining (+0.84% AvgRec on unseen models, +7.03% on unseen domains). The paper reports gains of 5.58% and 14.20% over prior state of the art in its two OOD settings and compatibility with several encoders.
- **Implication for the humanizer**: Retrieval in a style-embedding space is a natural check-round metric: the share of a text's k nearest neighbours that are human in a reference bank. Updating the bank with new human samples needs no training.

### 10. Few-Shot Detection of Machine-Generated Text using Style Representations

Rafael Rivera Soto, Kailin Koch, Aleem Khan, Barry Chen, Marcus Bishop, and Nicholas Andrews (2024). *Few-Shot Detection of Machine-Generated Text using Style Representations*. ICLR 2024.

- **Identifiers**: arXiv: [2401.06712](https://arxiv.org/abs/2401.06712) (verified: PDF header)
- **Code/data**: https://github.com/LLNL/LUAR/tree/main/fewshot_iclr2024
- **Summary**: Uses authorship style representations (UAR/LUAR, trained for human authorship identification) for few-shot detection of machine text, including texts from LLMs unseen in training. Features that separate human authors also separate human from machine authors (Llama-2, ChatGPT, GPT-4); documents are about 128 tokens long throughout, and in their visualisation SBERT semantic embeddings do not separate the sources while the stylistic (UAR) embeddings do. Given a handful of samples from a specific LLM, the method attributes further texts to it.
- **Implication for the humanizer**: Separates 'what is said' (semantic embeddings) from 'how it is said' (style embeddings). The humanizer should move style embeddings toward the human reference while keeping semantic embeddings close to the source; LUAR (HF: rrivera1849/LUAR-MUD) runs on CPU.

### 11. StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples

Ajay Patel, Jiacheng Zhu, Justin Qiu, Zachary Horvitz, Marianna Apidianaki, Kathleen McKeown, and Chris Callison-Burch (2025). *StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples*. NAACL 2025.

- **Identifiers**: arXiv: [2410.12757](https://arxiv.org/abs/2410.12757) · DOI: [10.18653/v1/2025.naacl-long.436](https://doi.org/10.18653/v1/2025.naacl-long.436) (verified: Crossref DOI)
- **Code/data**: https://huggingface.co/StyleDistance/styledistance
- **Summary**: Trains content-independent style embeddings by contrastive learning on a synthetic dataset of near-exact paraphrases that differ in controlled style features (40 features).
- **Implication for the humanizer**: Candidate style encoder for the check round; because it is trained to ignore content, distances to human reference texts reflect style rather than topic.

### 12. Same Author or Just Same Topic? Towards Content-Independent Style Representations

Anna Wegmann, Marijn Schraagen, and Dong Nguyen (2022). *Same Author or Just Same Topic? Towards Content-Independent Style Representations*. RepL4NLP workshop at ACL 2022.

- **Identifiers**: arXiv: [2204.04907](https://arxiv.org/abs/2204.04907) · DOI: [10.18653/v1/2022.repl4nlp-1.26](https://doi.org/10.18653/v1/2022.repl4nlp-1.26) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Compares ways of controlling content when training style representations and finds that controlling for conversation yields representations that capture style better independently of content than domain control or no control.
- **Implication for the humanizer**: Background for choosing a content-independent style encoder (an HF model 'AnnaWegmann/Style-Embedding' exists; its correspondence to this paper was not verified).

### 13. Idiosyncrasies in Large Language Models

Mingjie Sun, Yida Yin, Zhiqiu Xu, J. Zico Kolter, and Zhuang Liu (2025). *Idiosyncrasies in Large Language Models*. ICML 2025.

- **Identifiers**: arXiv: [2502.12150](https://arxiv.org/abs/2502.12150) (verified: PDF header + arXiv comment)
- **Code/data**: https://github.com/locuslab/llm-idiosyncrasies
- **Summary**: Fine-tunes text-embedding models (default LLM2vec with LoRA and an N-way head, inputs truncated to 512 tokens) to tell which LLM wrote a response. Five-way accuracy on ChatGPT, Claude, Grok, Gemini and DeepSeek reaches 97.1%; other backbones also score highly on chat models (ELMo 90.8, BERT 91.1, T5 90.5, GPT-2 92.1) and base models are harder (87.3 with LLM2vec). Removing special characters gives 95.1%, shuffling words 88.9%, shuffling letters 39.1%, so most of the signal lives in the word-level distribution; accuracy stays above 90% after rephrasing or translation by another LLM and above chance after summarisation. Markdown formatting contributes moderately.
- **Implication for the humanizer**: Model fingerprints survive paraphrase through word choice and content, so a humanizer that keeps the bag of words keeps the fingerprint. A bag-of-words or small fine-tuned encoder classifier is a cheap proxy for 'which model wrote this', which is a separate question from 'human or AI'.

### 14. Your Large Language Models Are Leaving Fingerprints

Hope McGovern, Rickard Stureborg, Yoshi Suhara, and Dimitris Alikaniotis (2024). *Your Large Language Models Are Leaving Fingerprints*. arXiv preprint (2024); later venue not verified.

- **Identifiers**: arXiv: [2405.14057](https://arxiv.org/abs/2405.14057) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Shows that simple classifiers on character/word n-gram and part-of-speech features detect machine text robustly in and out of domain. LLM 'fingerprints' are small frequency differences in lexical and morphosyntactic features, persist within model families (e.g. llama-13b vs llama-65b), and chat-tuned models are easier to detect than base models.
- **Implication for the humanizer**: POS n-gram frequency vectors are a cheap, interpretable distributional gap metric that a humanizer can target directly.

### 15. Whose LLM is it Anyway? Linguistic Comparison and LLM Attribution for GPT-3.5, GPT-4 and Bard

Ariel Rosenfeld and Teddy Lazebnik (2026). *Whose LLM is it Anyway? Linguistic Comparison and LLM Attribution for GPT-3.5, GPT-4 and Bard*. Mathematics (MDPI) 2026.

- **Identifiers**: arXiv: [2402.14533](https://arxiv.org/abs/2402.14533) · DOI: [10.3390/math14101683](https://doi.org/10.3390/math14101683) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Compares vocabulary, POS distribution, dependency distribution and sentiment of GPT-3.5, GPT-4 and Bard outputs on the same inputs and attributes texts to their model with 88% accuracy using an off-the-shelf classifier.
- **Implication for the humanizer**: Further evidence that coarse POS and dependency distributions carry model identity.

### 16. Detecting Machine-Generated Texts by Multi-Population Aware Optimization for Maximum Mean Discrepancy

Shuhai Zhang, Yiliao Song, Jiahao Yang, Yuanqing Li, Bo Han, and Mingkui Tan (2024). *Detecting Machine-Generated Texts by Multi-Population Aware Optimization for Maximum Mean Discrepancy*. ICLR 2024.

- **Identifiers**: arXiv: [2402.16041](https://arxiv.org/abs/2402.16041) (verified: arXiv comment)
- **Code/data**: https://github.com/ZSHsh98/MMD-MP
- **Summary**: Frames detection as a two-sample test with Maximum Mean Discrepancy on learned deep-kernel features and introduces multi-population-aware optimisation to cope with the variance of machine text produced by several LLMs; reports better paragraph- and sentence-level detection than baselines on GPT-2 to ChatGPT text.
- **Implication for the humanizer**: MMD between a batch of humanized outputs and a human reference batch is the right statistical frame for a 'distribution gap'; it can be computed on any feature space (embeddings, Biber vectors, surprisal statistics).

### 17. Human Variability vs. Machine Consistency: A Linguistic Analysis of Texts Generated by Humans and Large Language Models

Sergio E. Zanotto and Segun Aroyehun (2024). *Human Variability vs. Machine Consistency: A Linguistic Analysis of Texts Generated by Humans and Large Language Models*. arXiv preprint (2024); see note.

- **Identifiers**: arXiv: [2412.03025](https://arxiv.org/abs/2412.03025) (verified: arXiv)
- **Code/data**: https://github.com/brucewlee/lftk (feature tool used)
- **Summary**: Extracts about 250 LFTK linguistic features plus syntactic depth (spaCy), semantic distance (pairwise sentence cosine with sentence-transformers paraphrase-MiniLM-L6-v2) and emotion measures for human and LLM texts, then projects them with PCA. Human texts show considerably higher variability in these features, especially in genres with looser style constraints; human texts are less cognitively demanding, with higher semantic content and richer emotional content. The same authors published 'Linguistic and Embedding-Based Profiling of Texts Generated by Humans and Large Language Models' at EMNLP 2025 (DOI 10.18653/v1/2025.emnlp-main.1163); whether it is the extended version of this preprint was not verified.
- **Implication for the humanizer**: The gap is in dispersion as much as in means: the check round should compare variances and within-text spread (sentence length, syntactic depth, sentence-to-sentence semantic distance), not only averages.

### 18. Are AI Detectors Good Enough? A Survey on Quality of Datasets With Machine-Generated Texts

German Gritsai, Anastasia Voznyuk, Andrey Grabovoy, and Yury Chekhovich (2025). *Are AI Detectors Good Enough? A Survey on Quality of Datasets With Machine-Generated Texts*. PDLM workshop at AAAI 2025 (per arXiv comment).

- **Identifiers**: arXiv: [2410.14677](https://arxiv.org/abs/2410.14677) (verified: arXiv comment)
- **Code/data**: https://github.com/Advacheck-OU/ai-dataset-analysing
- **Summary**: Reviews datasets from AI-text detection competitions and argues that reported accuracies up to 99.9% partly reflect dataset artefacts; proposes methods to evaluate dataset quality.
- **Implication for the humanizer**: The skill's validation set must avoid trivial artefacts (length, formatting, prompt leakage), otherwise calibration measures artefacts rather than style.


## B. Probability-based zero-shot detectors and their successors

### 19. GLTR: Statistical Detection and Visualization of Generated Text

Sebastian Gehrmann, Hendrik Strobelt, and Alexander M. Rush (2019). *GLTR: Statistical Detection and Visualization of Generated Text*. ACL 2019 (System Demonstrations).

- **Identifiers**: arXiv: [1906.04043](https://arxiv.org/abs/1906.04043) · DOI: [10.18653/v1/p19-3019](https://doi.org/10.18653/v1/p19-3019) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Visualises, per token, the rank of the observed token under a language model (colour-coded rank buckets), together with per-token probability and the entropy of the predicted distribution. Generated text concentrates in the highest-rank (most predictable) buckets. In a human study the visual aid raised untrained readers' detection rate from 54% to 72%.
- **Implication for the humanizer**: Token-rank histograms are the simplest per-token signal and give an interpretable heat-map for the humanizer to show where text is too predictable.

### 20. Release Strategies and the Social Impacts of Language Models

Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herbert-Voss, Jeff Wu, et al. (2019). *Release Strategies and the Social Impacts of Language Models*. OpenAI report (2019).

- **Identifiers**: arXiv: [1908.09203](https://arxiv.org/abs/1908.09203) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Release report for GPT-2 that, among other analyses, evaluates zero-shot detection by thresholding average log-probability and a fine-tuned RoBERTa classifier.
- **Implication for the humanizer**: Origin of the mean log-likelihood baseline used in all later comparisons.

### 21. DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*. ICML 2023.

- **Identifiers**: arXiv: [2301.11305](https://arxiv.org/abs/2301.11305) (verified: arXiv comment)
- **Code/data**: https://ericmitchell.ai/detectgpt (project page)
- **Summary**: Observes that model samples lie in regions of negative curvature of the model's log-probability function. The score compares log p(x) with the mean log p of perturbed versions of x produced by a mask-filling model (T5). On GPT-NeoX-20B fake news it raised AUROC from 0.81 (best zero-shot baseline) to 0.95.
- **Implication for the humanizer**: Historically important but needs on the order of 100 perturbations and model calls per text, too slow for an iterative CPU loop; Fast-DetectGPT replaces it.

### 22. DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text

Jinyan Su, Terry Yue Zhuo, Di Wang, and Preslav Nakov (2023). *DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text*. Findings of EMNLP 2023.

- **Identifiers**: arXiv: [2306.05540](https://arxiv.org/abs/2306.05540) · DOI: [10.18653/v1/2023.findings-emnlp.827](https://doi.org/10.18653/v1/2023.findings-emnlp.827) (verified: Crossref DOI)
- **Code/data**: https://github.com/mbzuai-nlp/DetectLLM
- **Summary**: Introduces two log-rank-based scores: LRR (Log-Likelihood Log-Rank Ratio, the ratio of mean negative log-likelihood to mean log-rank, needs one forward pass) and NPR (Normalized Log-Rank Perturbation, needs perturbations). Improves over prior state of the art by 3.9 and 1.75 AUROC points.
- **Implication for the humanizer**: LRR is a one-pass CPU score that complements log-likelihood; cheap to add to the check round.

### 23. Smaller Language Models are Better Black-box Machine-Generated Text Detectors

Niloofar Mireshghallah, Justus Mattern, Sicun Gao, Reza Shokri, and Taylor Berg-Kirkpatrick (2024). *Smaller Language Models are Better Black-box Machine-Generated Text Detectors*. EACL 2024 (short); published title: 'Smaller Language Models are Better Zero-shot Machine-Generated Text Detectors'.

- **Identifiers**: arXiv: [2305.09859](https://arxiv.org/abs/2305.09859) · DOI: [10.18653/v1/2024.eacl-short.25](https://doi.org/10.18653/v1/2024.eacl-short.25) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Tests cross-detection, where the scoring model differs from the generator, over GPT, OPT and Pythia models of different sizes and training stages. Smaller and partially trained models are better universal detectors: OPT-125M reaches AUC 0.81 on ChatGPT text whereas GPT-J-6B reaches 0.45 (OPT-6.7B 0.58). Larger models are more conservative in their likelihood and curvature, which hurts cross-model detection.
- **Implication for the humanizer**: Direct support for using 100M-1.5B scoring models in a CPU toolkit: small scorers are not a compromise for likelihood/curvature signals.

### 24. Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang (2024). *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature*. ICLR 2024.

- **Identifiers**: arXiv: [2310.05130](https://arxiv.org/abs/2310.05130) (verified: PDF header + arXiv comment)
- **Code/data**: https://github.com/baoguangsheng/fast-detect-gpt
- **Summary**: Replaces DetectGPT's perturbations with conditional probability curvature computed analytically from logits: d(x) = (log p_theta(x|x) - mu) / sigma, where mu and sigma^2 are the sums over positions of the mean and variance of log p_theta(x~_t | x_<t) with x~_t drawn from a sampling model q_phi(.|x_<t). With the source model as both sampler and scorer, mean AUROC over five models is 0.9887 vs 0.9554 for DetectGPT; on ChatGPT/GPT-4 text 0.9338 vs 0.7225; about 340x faster. In the black-box setting (GPT-J sampler, GPT-Neo-2.7B scorer) mean AUROC is 0.9677, and it flags about 80% of ChatGPT text at 1% FPR. Accuracy rises monotonically with passage length, and it remains ahead of baselines under T5 paraphrasing and a word-swap 'decoherence' attack.
- **Implication for the humanizer**: Core CPU metric: one forward pass if the same model samples and scores. Higher (more positive) curvature means more machine-like; the humanizer should report the curvature z-score relative to human references of the same length.

### 25. Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein (2024). *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*. ICML 2024 (PMLR 235).

- **Identifiers**: arXiv: [2401.12070](https://arxiv.org/abs/2401.12070) (verified: PDF header)
- **Code/data**: https://github.com/ahans30/Binoculars
- **Summary**: Score = log-perplexity of the text under one model divided by the cross-perplexity between two models' next-token distributions on the same text, which normalises away prompt-induced low perplexity. Default: perplexity from Falcon-7B-Instruct, cross-perplexity from Falcon-7B and Falcon-7B-Instruct; the two models must share a tokenizer; low scores mean machine text. The repository code (binoculars/detector.py) defines a low-FPR threshold of 0.8536 (chosen at 0.01% FPR) and an F1-optimised threshold of 0.9015; the paper's worked example uses 0.901 as the 'global threshold'. Over many document types it detects over 90% of ChatGPT samples at 0.01% FPR without ChatGPT training data; other related pairs also work (appendix Table 3, e.g. Llama-13B with Llama-2-13B: TPR 99.65% at 0.01% FPR on their benchmark), so the pair need not be base/instruct. Documents were evaluated at 128, 256 and 512 tokens.
- **Implication for the humanizer**: Base/instruct pair scoring is the best-documented zero-shot detector, but the 7B pair is too large for CPU (two 7B models). A same-tokenizer small pair (e.g. Qwen2.5-1.5B and Qwen2.5-1.5B-Instruct) is a plausible substitute that must be calibrated locally; neither published threshold (0.8536 low-FPR, 0.9015 F1) transfers.

### 26. DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text

Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, and Haifeng Chen (2024). *DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text*. ICLR 2024.

- **Identifiers**: arXiv: [2305.17359](https://arxiv.org/abs/2305.17359) (verified: web search (iclr.cc poster page, dblp))
- **Code/data**: https://github.com/Xianjun-Yang/DNA-GPT
- **Summary**: Truncates a text, regenerates the continuation several times from the prefix with the suspected model, and compares the regenerations with the original suffix by n-gram overlap (black-box) or probability divergence (white-box). Machine text is reproduced more closely. Evaluated on text-davinci-003, GPT-3.5-turbo, GPT-4, GPT-NeoX-20B and LLaMA-13B.
- **Implication for the humanizer**: Requires generation with the (suspected) source model; costly on CPU and less relevant when the source model is unknown.

### 27. Ghostbuster: Detecting Text Ghostwritten by Large Language Models

Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein (2024). *Ghostbuster: Detecting Text Ghostwritten by Large Language Models*. NAACL 2024.

- **Identifiers**: arXiv: [2305.15047](https://arxiv.org/abs/2305.15047) · DOI: [10.18653/v1/2024.naacl-long.95](https://doi.org/10.18653/v1/2024.naacl-long.95) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Passes documents through a series of weaker language models, runs a structured search over vector and scalar combinations of these features, and trains a classifier. Reaches 99.0 F1 across domains, 5.9 F1 above the best prior model, and generalises better across domains, prompts and models.
- **Implication for the humanizer**: Template for a feature-search classifier over cheap probability features; the API models it used are retired, so a local re-implementation would use small open LMs.

### 28. SeqXGPT: Sentence-Level AI-Generated Text Detection

Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, and Xipeng Qiu (2023). *SeqXGPT: Sentence-Level AI-Generated Text Detection*. EMNLP 2023.

- **Identifiers**: arXiv: [2310.08903](https://arxiv.org/abs/2310.08903) · DOI: [10.18653/v1/2023.emnlp-main.73](https://doi.org/10.18653/v1/2023.emnlp-main.73) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Builds a sentence-level detection task with documents polished by LLMs and detects AI sentences from the per-token log-probability lists of several white-box LLMs, treated like waveforms and processed with convolution and self-attention.
- **Implication for the humanizer**: Supports per-sentence scoring from log-prob sequences, useful for locating residual machine-like sentences after humanization.

### 29. Glimpse: Enabling White-Box Methods to Use Proprietary Models for Zero-Shot LLM-Generated Text Detection

Guangsheng Bao, Yanbin Zhao, Juncai He, and Yue Zhang (2025). *Glimpse: Enabling White-Box Methods to Use Proprietary Models for Zero-Shot LLM-Generated Text Detection*. ICLR 2025.

- **Identifiers**: arXiv: [2412.11506](https://arxiv.org/abs/2412.11506) (verified: arXiv comment)
- **Code/data**: https://github.com/baoguangsheng/glimpse
- **Summary**: Estimates full next-token distributions from the top-k log-probabilities exposed by proprietary APIs, which lets white-box scores (Entropy, Rank, Log-Rank, Fast-DetectGPT) run on proprietary models. With Fast-DetectGPT and GPT-3.5 it reaches average AUROC about 0.95 on five recent source models, a 51% improvement relative to the remaining headroom of the open-source baseline.
- **Implication for the humanizer**: Optional: lets the check round use a stronger API scorer when available; not needed for the CPU path.

### 30. Training-free LLM-generated Text Detection by Mining Token Probability Sequences

Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, and Fei Wu (2025). *Training-free LLM-generated Text Detection by Mining Token Probability Sequences*. ICLR 2025.

- **Identifiers**: arXiv: [2410.06072](https://arxiv.org/abs/2410.06072) (verified: web search (proceedings.iclr.cc, iclr.cc poster))
- **Code/data**: https://github.com/TrustMedia-zju/Lastde_Detector
- **Summary**: Treats the token-probability sequence as a time series. It computes diversity entropy over sliding-window segments at multiple scales (local statistic) and combines it with likelihood (global statistic); Lastde++ adds fast sampling. In the paper's black-box experiments GPT-J is the single proxy model for scoring.
- **Implication for the humanizer**: Local temporal texture of the log-probability sequence is a signal that surface-level humanizers rarely target; it can be computed on CPU from the same forward pass as Fast-DetectGPT.

### 31. Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness

Shixuan Ma and Quan Wang (2024). *Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness*. EMNLP 2024.

- **Identifiers**: arXiv: [2409.16914](https://arxiv.org/abs/2409.16914) · DOI: [10.18653/v1/2024.emnlp-main.971](https://doi.org/10.18653/v1/2024.emnlp-main.971) (verified: Crossref DOI)
- **Code/data**: https://github.com/Shixuan-Ma/TOCSIN
- **Summary**: Defines token cohesiveness: after a few rounds of random token deletion, how much the text's meaning changes. LLM text is more cohesive than human text, and the score is a plug-in that improves four base zero-shot detectors in black-box settings. A follow-up by the same authors, 'TCD: Empowering Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness', appears in EMNLP 2026 proceedings per Crossref (DOI as returned: 10.18653/2026.emnlp-main.1708; not read).
- **Implication for the humanizer**: Cohesiveness is cheap with a small sentence encoder and is a candidate auxiliary signal.

### 32. Detecting Subtle Differences between Human and Model Languages Using Spectrum of Relative Likelihood

Yang Xu, Yu Wang, Hao An, Zhichen Liu, and Yongyuan Li (2024). *Detecting Subtle Differences between Human and Model Languages Using Spectrum of Relative Likelihood*. EMNLP 2024.

- **Identifiers**: arXiv: [2406.19874](https://arxiv.org/abs/2406.19874) · DOI: [10.18653/v1/2024.emnlp-main.564](https://doi.org/10.18653/v1/2024.emnlp-main.564) (verified: Crossref DOI)
- **Code/data**: https://github.com/CLCS-SUSTech/FourierGPT
- **Summary**: Uses relative rather than absolute likelihood values and extracts features from the Fourier spectrum of the likelihood sequence; supervised and heuristic classifiers are competitive with zero-shot methods and set a new state of the art on short-text detection.
- **Implication for the humanizer**: Spectral features of the surprisal sequence measure the 'rhythm' of predictability; useful for short texts where PHD fails.

### 33. BiScope: AI-generated Text Detection by Checking Memorization of Preceding Tokens

Hanxi Guo, Siyuan Cheng, Xiaolong Jin, Zhuo Zhang, Kaiyuan Zhang, Guanhong Tao, Guangyu Shen, Xiangyu Zhang (2024). *BiScope: AI-generated Text Detection by Checking Memorization of Preceding Tokens*. NeurIPS 2024.

- **Identifiers**: DOI: [10.52202/079017-3307](https://doi.org/10.52202/079017-3307) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Bibliographic record verified via Crossref only; full text not read. Method summary from memory (UNVERIFIED): a surrogate LLM is given a summary-based prompt and the text, and forward and backward cross-entropy features are passed to a classifier.
- **Implication for the humanizer**: Listed for completeness; do not rely on the method summary without reading the paper.

### 34. Imitate Before Detect: Aligning Machine Stylistic Preference for Machine-Revised Text Detection

Jiaqi Chen, Xiaoye Zhu, Tianyang Liu, Ying Chen, Xinhui Chen, Yiwen Yuan, et al. (2025). *Imitate Before Detect: Aligning Machine Stylistic Preference for Machine-Revised Text Detection*. AAAI 2025.

- **Identifiers**: arXiv: [2412.10432](https://arxiv.org/abs/2412.10432) · DOI: [10.1609/aaai.v39i22.34525](https://doi.org/10.1609/aaai.v39i22.34525) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Targets machine-revised text (human text polished by an LLM). Style preference optimisation (SPO) aligns a scoring model to machine writing style before Fast-DetectGPT-style scoring. Reports +13% AUC on text revised by open-source LLMs and +5% and +19% on GPT-3.5- and GPT-4o-revised text, surpassing GPTZero with 1,000 training samples and five minutes of SPO. A 2026 follow-up, HLPD (AAAI 2026, DOI 10.1609/aaai.v40i36.40297), aligns to human language preference instead (title-level record only).
- **Implication for the humanizer**: Humanized text is itself machine-revised text; style-aligned scorers are the detectors most likely to catch it, so the check round should include a revision-sensitive score.

### 35. Raidar: geneRative AI Detection viA Rewriting

Chengzhi Mao, Carl Vondrick, Hao Wang, and Junfeng Yang (2024). *Raidar: geneRative AI Detection viA Rewriting*. ICLR 2024.

- **Identifiers**: arXiv: [2401.12970](https://arxiv.org/abs/2401.12970) (verified: arXiv comment)
- **Code/data**: none found in paper/abstract
- **Summary**: Observes that LLMs change human-written text more than AI-generated text when asked to rewrite it, and uses the edit distance between a text and its LLM rewrite as the detection feature. Improves F1 of academic and commercial detectors by up to 29 points across news, creative writing, essays, code, Yelp reviews and arXiv papers.
- **Implication for the humanizer**: 'Rewrite stability' check: if a small instruct model leaves the humanized text almost unchanged, the text still sits where models like to be. Feasible on CPU with a 0.5-1.5B instruct model but slow.

### 36. Learning to Rewrite: Generalized LLM-Generated Text Detection

Ran Li, Wei Hao, Weiliang Zhao, Junfeng Yang, and Chengzhi Mao (2024). *Learning to Rewrite: Generalized LLM-Generated Text Detection*. arXiv preprint (2024); later venue not verified.

- **Identifiers**: arXiv: [2408.04237](https://arxiv.org/abs/2408.04237) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Learning2Rewrite trains the rewriting LLM to minimise edits on AI text, amplifying the Raidar gap. Across 21 domains and four LLMs it outperforms prior methods by up to 23.04% AUROC in-distribution, with larger gains reported out of distribution.
- **Implication for the humanizer**: Same signal as Raidar with a trained rewriter; shows that the edit-distance gap is learnable and robust across domains.

### 37. AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees

Hongyi Zhou, Jin Zhu, Pingfan Su, Kai Ye, Ying Yang, Shakeel A O B Gavioli-Akilagun, and Chengchun Shi (2025). *AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees*. NeurIPS 2025.

- **Identifiers**: arXiv: [2510.01268](https://arxiv.org/abs/2510.01268) · DOI: [10.52202/085713-2980](https://doi.org/10.52202/085713-2980) (verified: Crossref DOI)
- **Code/data**: https://github.com/Mamba413/AdaDetectGPT
- **Summary**: Learns a witness function from training data that reweights logits-based statistics, with statistical guarantees on TPR, FPR, TNR and FNR; improves the state-of-the-art logits detector nearly uniformly, by up to 37%.
- **Implication for the humanizer**: A small amount of labelled local data can tune Fast-DetectGPT-type scores; relevant if the skill builds a calibration set.

### 38. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm

Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, and Yanan Cao (2025). *DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm*. NeurIPS 2025 (spotlight per arXiv comment).

- **Identifiers**: arXiv: [2509.15550](https://arxiv.org/abs/2509.15550) · DOI: [10.52202/085713-0065](https://doi.org/10.52202/085713-0065) (verified: Crossref DOI)
- **Code/data**: https://github.com/Xiaoweizhu57/DNA-DetectLLM
- **Summary**: Zero-shot detector that constructs an 'ideal' AI-generated sequence for the input, iteratively repairs non-optimal tokens toward it, and uses the cumulative repair effort as the score; motivated by growing overlap of human and AI feature distributions.
- **Implication for the humanizer**: Repair effort is another view of 'distance from the model's preferred text'; human text needs more repair.

### 39. DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models

Jiachen Fu, Chun-Le Guo, and Chongyi Li (2025). *DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models*. ACM Multimedia 2025.

- **Identifiers**: arXiv: [2509.14268](https://arxiv.org/abs/2509.14268) · DOI: [10.1145/3746027.3754862](https://doi.org/10.1145/3746027.3754862) (verified: Crossref DOI)
- **Code/data**: https://fjc2005.github.io/detectanyllm (project page)
- **Summary**: Direct Discrepancy Learning trains the scoring model with an objective aligned to the detection task. The MIRAGE benchmark samples human texts from 10 corpora in 5 domains and re-generates or revises them with 17 LLMs. Reports over 70% improvement over existing methods with the same training data and base scoring model.
- **Implication for the humanizer**: MIRAGE's 'revise' split is a realistic test set for humanized text.

### 40. RADAR: Robust AI-Text Detection via Adversarial Learning

Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho (2023). *RADAR: Robust AI-Text Detection via Adversarial Learning*. NeurIPS 2023.

- **Identifiers**: arXiv: [2307.03838](https://arxiv.org/abs/2307.03838) · DOI: [10.52202/075280-0662](https://doi.org/10.52202/075280-0662) (verified: Crossref DOI)
- **Code/data**: https://radar.vizhub.ai (project page)
- **Summary**: Trains a detector adversarially against a paraphraser; outperforms earlier detectors across 8 LLMs and 4 datasets, especially under paraphrasing, and transfers from instruction-tuned LLMs to others.
- **Implication for the humanizer**: Adversarially trained classifiers resist simple attacks (also seen in RAID) but fall to detector-guided paraphrasing (Cheng et al. 2025) and to LLM-rewritten human text (ARB).

### 41. Technical Report on the Pangram AI-Generated Text Classifier

Bradley Emi and Max Spero (2024). *Technical Report on the Pangram AI-Generated Text Classifier*. Technical report (arXiv 2024).

- **Identifiers**: arXiv: [2402.14873](https://arxiv.org/abs/2402.14873) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Describes a commercial transformer classifier trained with hard-negative mining and synthetic mirror prompts; claims over 38 times lower error than DetectGPT and leading commercial tools on a 10-domain benchmark, no bias against non-native English writers, and generalisation to unseen domains and models.
- **Implication for the humanizer**: Commercial reference detector; independent evaluations (Russell et al. 2025) found it robust to paraphrasing and humanizers, so it is a meaningful external check.

### 42. DAMAGE: Detecting Adversarially Modified AI Generated Text

Elyas Masrour, Bradley Emi, and Max Spero (2025). *DAMAGE: Detecting Adversarially Modified AI Generated Text*. arXiv preprint (2025); venue not verified.

- **Identifiers**: arXiv: [2501.03437](https://arxiv.org/abs/2501.03437) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Studies 19 humanizer and paraphrasing tools, assesses meaning preservation qualitatively, shows that many detectors fail on humanized text, and trains a detector with data-centric augmentation that keeps a low FPR; the detector stays robust when the authors fine-tune an attacker against it.
- **Implication for the humanizer**: Commercial detectors are being trained on humanizer outputs, so surface rewriting alone has a short shelf life.


## C. Surprisal dynamics, uniform information density and length effects

### 43. Revisiting the Uniform Information Density Hypothesis

Clara Meister, Tiago Pimentel, Patrick Haller, Lena Jäger, Ryan Cotterell, and Roger Levy (2021). *Revisiting the Uniform Information Density Hypothesis*. EMNLP 2021.

- **Identifiers**: arXiv: [2109.11635](https://arxiv.org/abs/2109.11635) · DOI: [10.18653/v1/2021.emnlp-main.74](https://doi.org/10.18653/v1/2021.emnlp-main.74) (verified: Crossref DOI)
- **Code/data**: https://github.com/rycolab/revisiting-uid (cited by GPT-who)
- **Summary**: Revisits the uniform information density hypothesis and compares operationalisations of UID (surprisal variance, local surprisal differences, super-linear surprisal costs) against acceptability judgements and reading times.
- **Implication for the humanizer**: Supplies the exact UID definitions used by GPT-who; human text is not maximally uniform, so variance-type measures are the relevant gap.

### 44. GPT-who: An Information Density-based Machine-Generated Text Detector

Saranya Venkatraman, Adaku Uchendu, and Dongwon Lee (2024). *GPT-who: An Information Density-based Machine-Generated Text Detector*. Findings of NAACL 2024.

- **Identifiers**: arXiv: [2310.06202](https://arxiv.org/abs/2310.06202) · DOI: [10.18653/v1/2024.findings-naacl.8](https://doi.org/10.18653/v1/2024.findings-naacl.8) (verified: Crossref DOI)
- **Code/data**: https://github.com/saranya-venkatraman/gpt-who
- **Summary**: Computes token surprisal with GPT2-XL and derives UID features: mean surprisal, UID variance (mean squared deviation from the mean surprisal), UID difference (mean absolute difference between consecutive surprisals), its squared variant, and the minimum/maximum UID over spans of N tokens (N = 20 maximised F1 in their ablation); a logistic regression maps them to authors. Outperforms GLTR, GPTZero, DetectGPT, the OpenAI detector and ZeroGPT by over 20% across domains on four benchmarks, at about 0.8 s inference per article.
- **Implication for the humanizer**: Surprisal-variance features are cheap, interpretable targets: the humanizer can report them per text and aim for the human reference range rather than simply raising perplexity.

### 45. Diversity Boosts AI-Generated Text Detection

Advik Raj Basani and Pin-Yu Chen (2026). *Diversity Boosts AI-Generated Text Detection*. TMLR 2026.

- **Identifiers**: arXiv: [2509.18880](https://arxiv.org/abs/2509.18880) (verified: PDF header + arXiv comment)
- **Code/data**: https://github.com/IBM/diveye
- **Summary**: Computes per-token surprisal with GPT-2 (default) and extracts a 9-dimensional diversity vector: mean, variance, skewness and kurtosis of surprisal; mean and variance of the first-order difference of surprisal; variance, binned entropy and lag-1 autocorrelation of the second-order difference. An XGBoost classifier gives the score. Human text shows wider spread and heavier tails; positive skew indicates rare, surprising tokens. Outperforms zero-shot detectors by up to 33.2%, improves other detectors by up to 18.7% as an auxiliary signal, and is reported robust to paraphrasing and adversarial attacks.
- **Implication for the humanizer**: The most CPU-friendly 'rhythm' detector (GPT-2 small). Its features double as humanizer targets: increase surprisal variance and tail weight while keeping fluency.

### 46. When AI Settles Down: Late-Stage Stability as a Signature of AI-Generated Text Detection

Ke Sun, Guangsheng Bao, Han Cui, and Yue Zhang (2026). *When AI Settles Down: Late-Stage Stability as a Signature of AI-Generated Text Detection*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2601.04833](https://arxiv.org/abs/2601.04833) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Analyses over 120k texts and finds 'late-stage volatility decay': the log-probability fluctuations of AI text stabilise as generation proceeds, with 24-32% lower volatility than human text in the second half. Two features from the second half of the sequence (from position floor(n/2)): derivative dispersion, the standard deviation of the absolute first difference of token negative log-probability, and local volatility, the mean of the standard deviation within a sliding window (w = 20). With Llama-3-8B-Instruct as surrogate (512-token truncation) it reports state-of-the-art results on EvoBench and MAGE and complementarity with global methods.
- **Implication for the humanizer**: Endings matter: humanizers tend to edit openings and leave the second half smooth. The check round should compute surprisal statistics separately for the second half.

### 47. How Short Is Too Short? Length-Conditional Calibration and Abstention for Detecting AI-Generated Text

Ritesh Janga and Justin Zhan (2026). *How Short Is Too Short? Length-Conditional Calibration and Abstention for Detecting AI-Generated Text*. arXiv preprint (Nov 2026); venue not verified.

- **Identifiers**: arXiv: [2611.10901](https://arxiv.org/abs/2611.10901) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Derives closed forms for how the false-positive rate of a threshold tuned on long text drifts as passages get shorter, and for the number of tokens needed to reach a target TPR. Calibrates thresholds within length bins with conformal p-values and abstains in bins where the detector cannot be informative. With log-likelihood and log-rank from GPT-2 and Binoculars from a Falcon-7B/Falcon-7B-Instruct pair, binned calibration holds FPR at 5.0% from 10 to 200 tokens; a single threshold set at 200 tokens gives 29% FPR for Binoculars at 10 tokens. No detector reaches 80% TPR; the best is Binoculars at 200 tokens with 59%. The minimum-length formula puts 80% power for GPT-2 log-likelihood near 1,100 tokens.
- **Implication for the humanizer**: Every score in the check round needs length-binned calibration against human references and an explicit abstain state for short texts.


## D. Corpus-level distribution shift

### 48. Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews

Weixin Liang, Zachary Izzo, Yaohui Zhang, Haley Lepp, Hancheng Cao, Xuandong Zhao, et al. (2024). *Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews*. ICML 2024 (PMLR 235).

- **Identifiers**: arXiv: [2403.07183](https://arxiv.org/abs/2403.07183) (verified: PDF header + arXiv comment)
- **Code/data**: https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
- **Summary**: Models a target corpus as the mixture (1 - alpha) P + alpha Q of human (P) and AI (Q) word-occurrence distributions estimated from reference texts, and estimates alpha by maximum likelihood instead of classifying documents. Estimates that 6.5-16.9% of peer-review text at ICLR 2024, NeurIPS 2023, CoRL 2023 and EMNLP 2023 was substantially LLM-modified. Adjectives such as 'commendable', 'meticulous' and 'intricate' became 9.8-, 34.7- and 11.2-fold more likely to occur in a sentence. The method is about seven orders of magnitude cheaper than per-document detectors.
- **Implication for the humanizer**: A distributional estimator over an adjective/adverb vocabulary is a cheap corpus-level check: applied to a batch of humanized outputs it estimates the fraction that still looks AI-modified.

### 49. Mapping the Increasing Use of LLMs in Scientific Papers

Weixin Liang, Yaohui Zhang, Zhengxuan Wu, Haley Lepp, Wenlong Ji, Xuandong Zhao, et al. (2025). *Mapping the Increasing Use of LLMs in Scientific Papers*. Nature Human Behaviour 2025, 'Quantifying large language model usage in scientific papers' (same 14 authors); COLM 2024 venue of the preprint not verified.

- **Identifiers**: arXiv: [2404.01268](https://arxiv.org/abs/2404.01268) · DOI: [10.1038/s41562-025-02273-8](https://doi.org/10.1038/s41562-025-02273-8) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Applies the same population-level framework to 950,965 papers (Jan 2020-Feb 2024) on arXiv, bioRxiv and Nature portfolio journals. LLM-modified content rises steadily, fastest in computer science (up to 17.5%) and least in mathematics and Nature portfolio papers (up to 6.3%).
- **Implication for the humanizer**: Register matters: base rates of AI-like vocabulary differ by field, so the human reference set must match the register of the text being humanized.

### 50. The Widespread Adoption of Large Language Model-Assisted Writing Across Society

Weixin Liang, Yaohui Zhang, Mihai Codreanu, Jiayu Wang, Hancheng Cao, and James Zou (2025). *The Widespread Adoption of Large Language Model-Assisted Writing Across Society*. Patterns 2025.

- **Identifiers**: arXiv: [2502.09747](https://arxiv.org/abs/2502.09747) · DOI: [10.1016/j.patter.2025.101366](https://doi.org/10.1016/j.patter.2025.101366) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Extends the framework to consumer complaints (687,241), corporate press releases (537,413), job postings (304.3 million) and UN press releases (15,919) from Jan 2022 to Sep 2024. By late 2024 about 18% of financial consumer-complaint text and up to 24% of corporate press-release text appear LLM-assisted.
- **Implication for the humanizer**: Reference corpora collected after 2023 are contaminated; human references for calibration should predate late 2022 or be otherwise vetted.

### 51. Delving into LLM-assisted writing in biomedical publications through excess vocabulary

Dmitry Kobak, Rita González-Márquez, Emőke-Ágnes Horvát, and Jan Lause (2025). *Delving into LLM-assisted writing in biomedical publications through excess vocabulary*. Science Advances 11(27), 2025.

- **Identifiers**: arXiv: [2406.07016](https://arxiv.org/abs/2406.07016) · DOI: [10.1126/sciadv.adt3813](https://doi.org/10.1126/sciadv.adt3813) (verified: Crossref DOI + arXiv journal_ref)
- **Code/data**: https://github.com/berenslab/llm-excess-vocab
- **Summary**: Builds the word-occurrence matrix of 15.1 million English PubMed abstracts (2010-2024) and compares each word's observed 2024 frequency p with a counterfactual expected frequency q obtained by linear extrapolation from 2021 and 2022, using the excess gap delta = p - q (sensitive for frequent words) and the excess ratio r = p / q (sensitive for rare words). Words such as 'delves' (r = 28.0), 'underscores' (r = 13.8) and 'showcasing' (r = 10.7) show unprecedented excess; the 2024 excess vocabulary consists almost entirely of style words, unlike the content words (e.g. covid) of earlier years. The excess implies that at least 13.5% of 2024 abstracts were processed with LLMs, up to 40% in some subcorpora.
- **Implication for the humanizer**: The excess-ratio list is a ready-made, data-driven 'AI vocabulary' lexicon for the surface layer of the humanizer; it should be regenerated per register rather than hand-curated.

### 52. Most biomedical publications show signs of LLM-assisted writing

Lena Holzwarth, Rita González-Márquez, and Dmitry Kobak (2026). *Most biomedical publications show signs of LLM-assisted writing*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2608.10715](https://arxiv.org/abs/2608.10715) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Proposes and validates a new unbiased word-frequency estimator of LLM usage and applies it to open-access PubMed Central full texts. By the end of 2025, 89% of papers show excess LLM-associated vocabulary; Discussion paragraphs are about twice as likely as Methods paragraphs to be LLM-assisted (68% vs 32%).
- **Implication for the humanizer**: Within-document variation: sections differ in how AI-like they read, so section-aware references help.

### 53. Is ChatGPT Transforming Academics' Writing Style?

Mingmeng Geng and Roberto Trotta (2024). *Is ChatGPT Transforming Academics' Writing Style?*. arXiv preprint (2024); venue not verified.

- **Identifiers**: arXiv: [2404.08627](https://arxiv.org/abs/2404.08627) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Uses word-frequency changes in one million arXiv abstracts (May 2018-Jan 2024) to estimate the share of abstracts with ChatGPT-style writing, about 35% in computer science under the authors' modelling assumptions.
- **Implication for the humanizer**: Independent replication of the excess-vocabulary approach on arXiv.

### 54. ChatGPT "contamination": estimating the prevalence of LLMs in the scholarly literature

Andrew Gray (2024). *ChatGPT "contamination": estimating the prevalence of LLMs in the scholarly literature*. arXiv preprint (2024).

- **Identifiers**: arXiv: [2403.16887](https://arxiv.org/abs/2403.16887) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Tracks disproportionate increases of indicative keywords in 2023 scholarly papers and estimates that at least 60,000 papers (slightly over 1%) were LLM-assisted.
- **Implication for the humanizer**: Keyword-based lower bound; shows how a few marker words already move corpus statistics.

### 55. Why Does ChatGPT "Delve" So Much? Exploring the Sources of Lexical Overrepresentation in Large Language Models

Tom S. Juzek and Zina B. Ward (2025). *Why Does ChatGPT "Delve" So Much? Exploring the Sources of Lexical Overrepresentation in Large Language Models*. COLING 2025 (per arXiv comment).

- **Identifiers**: arXiv: [2412.11385](https://arxiv.org/abs/2412.11385) (verified: arXiv comment)
- **Code/data**: none found in paper/abstract
- **Summary**: Develops a transferable method to identify 'focal words' overused because of LLMs (21 words in scientific abstracts) and tests explanations; no evidence points to architecture, algorithm choices or training data; comparative model testing is consistent with RLHF playing a role, while an exploratory online study suggests participants react differently to 'delve' than to other focal words.
- **Implication for the humanizer**: Overused vocabulary is plausibly a post-training artefact, consistent with base models reading as more human (Xu et al. 2026; Reinhart et al. 2025).


## E. Register, syntactic and stylometric differences

### 56. Do LLMs write like humans? Variation in grammatical and rhetorical styles

Alex Reinhart, Ben Markey, Michael Laudenbach, Kachatad Pantusen, Ronald Yurko, Gordon Weinberg, and David West Brown (2025). *Do LLMs write like humans? Variation in grammatical and rhetorical styles*. PNAS 122 (2025), e2422455122.

- **Identifiers**: arXiv: [2410.16107](https://arxiv.org/abs/2410.16107) · DOI: [10.1073/pnas.2422455122](https://doi.org/10.1073/pnas.2422455122) (verified: Crossref DOI + arXiv journal_ref)
- **Code/data**: pseudobibeR (R package on CRAN, v1.3 checked)
- **Summary**: Builds two parallel corpora (HAP-E: 66,320 chunks; CAP, built from COCA: 76,920 chunks; two human and six LLM chunks per source text) where GPT-4o, GPT-4o Mini and four Llama 3 variants continue human-written prompts, and tags 66 Biber features with pseudobibeR. Instruction-tuned models use present participial clauses at 2-5 times the human rate (GPT-4o 5.3x, paired Cohen's d = 1.38), 'that' clauses as subject 2.6x (d = 0.77), nominalisations 2.1x (d = 1.23) and phrasal coordination 1.9x (d = 0.81), and agentless passives at about half the human rate; Llama base models are closer to human rates. A random forest separates the seven sources with 66% accuracy (chance 14%), mostly confusing LLMs with each other; pairwise human-vs-LLM forests reach 93-98% even across corpora, but accuracy drops on M4 arXiv text.
- **Implication for the humanizer**: Gives quantitative targets for grammatical rewriting (fewer participial clauses and nominalisations, more agentless passives and clausal variety, in the register's human range). Instruction tuning, not model size, drives the gap.

### 57. Benchmark of stylistic variation in LLM-generated texts

Jiří Milička, Anna Marklová, and Václav Cvrček (2026). *Benchmark of stylistic variation in LLM-generated texts*. International Journal of Corpus Linguistics 2026 (journal title: 'Benchmark of linguistic variation in LLM-generated texts').

- **Identifiers**: arXiv: [2509.10179](https://arxiv.org/abs/2509.10179) · DOI: [10.1075/ijcl.25151.mil](https://doi.org/10.1075/ijcl.25151.mil) (verified: Crossref DOI)
- **Code/data**: https://osf.io/hs7xt/ (data and scripts)
- **Summary**: Applies Biber's multidimensional analysis to AI-Brown, an LLM counterpart of the BE-21 Brown-family corpus of British English, and replicates in Czech with AI-Koditex. Examines 16 frontier models, settings and prompts, with emphasis on base versus instruction-tuned models, and turns the dimensions on which LLMs differ most into a benchmark that ranks models in interpretable dimensions.
- **Implication for the humanizer**: MDA dimension scores (e.g. involved vs informational production) give a low-dimensional register profile to compare humanized text against.

### 58. Interpretable Stylistic Variation in Human and LLM Writing Across Genres, Models, and Decoding Strategies

Swati Rallapalli, Shannon Gallagher, Ronald Yurko, Tyler Brooks, Chuck Loughin, Michele Sezgin, and Violet Turri (2026). *Interpretable Stylistic Variation in Human and LLM Writing Across Genres, Models, and Decoding Strategies*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2604.14111](https://arxiv.org/abs/2604.14111) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Analyses Biber features of human text and of 11 LLMs across 8 genres and 4 decoding strategies. Key LLM differentiators are robust to prompts that nudge toward human-like writing and to continuing human text; genre influences style more than source; chat variants cluster together; model matters more than decoding strategy.
- **Implication for the humanizer**: Prompting for 'human style' does not remove Biber-level differences; the humanizer must edit grammatical features explicitly and check them after rewriting.

### 59. How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework

Björn Nieth, Marianna Gracheva, Michaela Mahlberg, Bjoern Eskofier, and Emmanuelle Salin (2026). *How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2605.23651](https://arxiv.org/abs/2605.23651) (verified: arXiv)
- **Code/data**: https://github.com/BjoernNieth/Register_Aware_LLMs
- **Summary**: Frames human-likeness as a two-sample problem between the distribution of 67 Biber features in a register-specific human reference corpus and in an LLM corpus, measured with Maximum Mean Discrepancy. Seven instruction-tuned open models deviate from the human baseline in all five English registers; which model is closest depends on register, not size.
- **Implication for the humanizer**: Closest published analogue of the check round the skill needs: register-matched MMD on Biber vectors, reportable as a single gap number with a permutation p-value.

### 60. Detection and Measurement of Syntactic Templates in Generated Text

Chantal Shaib, Yanai Elazar, Junyi Jessy Li, and Byron C. Wallace (2024). *Detection and Measurement of Syntactic Templates in Generated Text*. EMNLP 2024.

- **Identifiers**: arXiv: [2407.00211](https://arxiv.org/abs/2407.00211) · DOI: [10.18653/v1/2024.emnlp-main.368](https://doi.org/10.18653/v1/2024.emnlp-main.368) (verified: Crossref DOI)
- **Code/data**: https://cshaib.github.io/syntactic_templates/ (project page)
- **Summary**: Defines syntactic templates as repeated part-of-speech n-grams (lengths n = 4-8 analysed; n = 6 in the main tables) and measures CR-POS (gzip compression ratio of the POS-tag sequence; higher = less diverse), template rate (fraction of texts with at least one template) and templates per token. 76% of templates in model text occur in pre-training data versus 35% for human text, and templates are not overwritten by fine-tuning such as RLHF. A 2026 follow-up applies template-based unsupervised detection to Korean (Findings of EACL 2026, DOI 10.18653/v1/2026.findings-eacl.77; record only).
- **Implication for the humanizer**: POS-template rate and CR-POS are cheap with spaCy and capture repetitive sentence frames that lexical substitution leaves intact.

### 61. Contrasting Linguistic Patterns in Human and LLM-Generated News Text

Alberto Muñoz-Ortiz, Carlos Gómez-Rodríguez, and David Vilares (2024). *Contrasting Linguistic Patterns in Human and LLM-Generated News Text*. Artificial Intelligence Review 57, 265 (2024).

- **Identifiers**: arXiv: [2308.09067](https://arxiv.org/abs/2308.09067) · DOI: [10.1007/s10462-024-10903-2](https://doi.org/10.1007/s10462-024-10903-2) (verified: arXiv DOI + journal_ref)
- **Code/data**: none found in paper/abstract
- **Summary**: Compares human and LLM news text on morphological, syntactic, psychometric and sociolinguistic features. Human texts have more scattered sentence-length distributions, more varied vocabulary, distinct dependency and constituent types, shorter constituents and more optimised dependency distances; humans express stronger negative emotions; LLM outputs use more numbers, symbols and auxiliaries.
- **Implication for the humanizer**: Sentence-length dispersion, lexical variety and dependency-distance statistics are concrete, cheap targets.

### 62. How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection

Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng Wu (2023). *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection*. arXiv preprint (2023).

- **Identifiers**: arXiv: [2301.07597](https://arxiv.org/abs/2301.07597) (verified: arXiv)
- **Code/data**: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
- **Summary**: Releases the Human ChatGPT Comparison Corpus (HC3) of paired human expert and ChatGPT answers, with human evaluations, linguistic analyses and detection baselines.
- **Implication for the humanizer**: Common paired dataset for quick validation, now dated (ChatGPT 2022).

### 63. Stylometric Detection of AI-Generated Text in Twitter Timelines

Tharindu Kumarage, Joshua Garland, Amrita Bhattacharjee, Kirill Trapeznikov, Scott Ruston, and Huan Liu (2023). *Stylometric Detection of AI-Generated Text in Twitter Timelines*. arXiv preprint (2023); venue not verified.

- **Identifiers**: arXiv: [2303.03697](https://arxiv.org/abs/2303.03697) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Uses stylometric features to detect AI tweets and to find when an AI starts generating tweets in a timeline.
- **Implication for the humanizer**: Stylometry works on very short texts when aggregated over a timeline; per-text it is weak.

### 64. Stylometry recognizes human and LLM-generated texts in short samples

Karol Przystalski, Jan K. Argasiński, Iwona Grabska-Gradzińska, and Jeremi K. Ochab (2026). *Stylometry recognizes human and LLM-generated texts in short samples*. Expert Systems with Applications 2026.

- **Identifiers**: arXiv: [2507.00838](https://arxiv.org/abs/2507.00838) · DOI: [10.1016/j.eswa.2025.129001](https://doi.org/10.1016/j.eswa.2025.129001) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Builds a Wikipedia-based benchmark with human summaries, LLM texts, summariser outputs and paraphrases, and classifies 10-sentence texts with decision trees and LightGBM on StyloMetrix and n-gram stylometric features; reaches up to 0.87 Matthews correlation in a 7-class setting.
- **Implication for the humanizer**: Hand-crafted stylometric features remain competitive on short samples and are fully CPU-bound.

### 65. DependencyAI: Detecting AI Generated Text through Dependency Parsing

Sara Ahmed and Tracy Hammond (2026). *DependencyAI: Detecting AI Generated Text through Dependency Parsing*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2602.15514](https://arxiv.org/abs/2602.15514) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Detects AI text using only the sequence of dependency-relation labels; competitive in monolingual, multi-generator and multilingual settings, with systematic over-prediction for some generators on unseen domains.
- **Implication for the humanizer**: Dependency-label distributions are a content-free syntactic signal that survives lexical paraphrase.

### 66. Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy

Shushanta Pudasaini, Luis Miralles-Pechuán, David Lillis, and Marisa Llorens Salvador (2026). *Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2603.23146](https://arxiv.org/abs/2603.23146) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: A classifier on 30 linguistic features reaches F1 0.9734 on PAN CLEF 2025 and COLING 2025 data but degrades substantially across domains and generators; SHAP shows that the most influential features differ between datasets.
- **Implication for the humanizer**: Feature-based detectors learn dataset-specific cues, so the humanizer should be validated on several registers, not tuned to one.

### 67. Detecting the Machine: A Comprehensive Benchmark of AI-Generated Text Detectors Across Architectures, Domains, and Adversarial Conditions

Madhav S. Baidya, S. S. Baidya, and Chirag Chawla (2026). *Detecting the Machine: A Comprehensive Benchmark of AI-Generated Text Detectors Across Architectures, Domains, and Adversarial Conditions*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2603.17522](https://arxiv.org/abs/2603.17522) (verified: arXiv)
- **Code/data**: https://github.com/MadsDoodle/Human-and-LLM-Generated-Text-Detectability-under-Adversarial-Humanization
- **Summary**: Benchmarks classical classifiers, fine-tuned encoders, an XGBoost stylometric model, perplexity-based detectors and LLM-as-detector prompting on HC3 and ELI5 pairs. Encoders are near-perfect in distribution but degrade under domain shift; the stylometric model matches them while staying interpretable; LLM-as-detector underperforms; perplexity-based methods show a polarity issue that must be corrected; no method generalises robustly.
- **Implication for the humanizer**: Supports a feature-based (stylometric) component in the check round and warns against LLM-as-judge detection.

### 68. Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models

Samuel Paech, Allen Roush, Judah Goldfeder, and Ravid Shwartz-Ziv (2025). *Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models*. arXiv preprint (2025); venue not verified.

- **Identifiers**: arXiv: [2510.15061](https://arxiv.org/abs/2510.15061) (verified: arXiv)
- **Code/data**: https://github.com/sam-paech/auto-antislop
- **Summary**: Profiles model-specific over-represented strings ('slop') against human baselines; some patterns are over 1,000 times more frequent in LLM output than in human text. An inference-time sampler with backtracking suppresses 8,000+ patterns while maintaining quality, whereas token banning becomes unusable at about 2,000; a token-level preference-optimisation fine-tuning method (FTPO) uses the same profiles.
- **Implication for the humanizer**: A data-driven over-representation ratio against a human baseline is the right way to build the surface-pattern list, and backtracking suppression is a generation-time alternative to post-hoc rewriting.


## F. Semantic homogenization and diversity

### 69. Does Writing with Language Models Reduce Content Diversity?

Vishakh Padmakumar and He He (2024). *Does Writing with Language Models Reduce Content Diversity?*. ICLR 2024.

- **Identifiers**: arXiv: [2309.05196](https://arxiv.org/abs/2309.05196) (verified: arXiv comment)
- **Code/data**: none found in paper/abstract
- **Summary**: Controlled study of argumentative essays written alone, with GPT-3 (base) or with InstructGPT. Writing with InstructGPT, but not with the base model, significantly reduces the diversity of essays across users.
- **Implication for the humanizer**: Homogenisation is tied to instruction tuning; corpus-level diversity across many humanized outputs is a separate check from per-text human-likeness.

### 70. Homogenization Effects of Large Language Models on Human Creative Ideation

Barrett R. Anderson, Jash Hemant Shah, and Max Kreminski (2024). *Homogenization Effects of Large Language Models on Human Creative Ideation*. ACM Creativity & Cognition 2024.

- **Identifiers**: arXiv: [2402.01536](https://arxiv.org/abs/2402.01536) · DOI: [10.1145/3635636.3656204](https://doi.org/10.1145/3635636.3656204) (verified: arXiv DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: In a 36-participant study, users produced less semantically distinct ideas with ChatGPT than with an alternative creativity support tool.
- **Implication for the humanizer**: Measure across-output semantic dispersion (mean pairwise embedding distance) for batches.

### 71. Generative AI enhances individual creativity but reduces the collective diversity of novel content

Anil R. Doshi, Oliver P. Hauser (2024). *Generative AI enhances individual creativity but reduces the collective diversity of novel content*. Science Advances 2024.

- **Identifiers**: DOI: [10.1126/sciadv.adn5290](https://doi.org/10.1126/sciadv.adn5290) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Record verified via Crossref; content summarised from the title only: access to generative AI ideas raised the rated creativity of individual stories but made stories more similar to each other.
- **Implication for the humanizer**: Same batch-level diversity concern.

### 72. AI Suggestions Homogenize Writing Toward Western Styles and Diminish Cultural Nuances

Dhruv Agarwal, Mor Naaman, and Aditya Vashistha (2025). *AI Suggestions Homogenize Writing Toward Western Styles and Diminish Cultural Nuances*. CHI 2025.

- **Identifiers**: arXiv: [2409.11360](https://arxiv.org/abs/2409.11360) · DOI: [10.1145/3706598.3713564](https://doi.org/10.1145/3706598.3713564) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Cross-cultural experiment with 118 participants from India and the US: AI writing suggestions gave larger efficiency gains to Americans and led Indian participants to adopt Western writing styles, changing how, not only what, they wrote.
- **Implication for the humanizer**: 'Human' is not one distribution; the reference corpus should match the writer's variety, or the humanizer will push everyone toward one style.

### 73. We're Different, We're the Same: Creative Homogeneity Across LLMs

Emily Wenger and Yoed Kenett (2025). *We're Different, We're the Same: Creative Homogeneity Across LLMs*. arXiv preprint (2025); venue not verified.

- **Identifiers**: arXiv: [2501.19361](https://arxiv.org/abs/2501.19361) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Using standardised creativity tests, LLM responses are much more similar to other LLMs' responses than human responses are to each other, even after controlling for response structure.
- **Implication for the humanizer**: Switching the rewriting model does not by itself restore human-level diversity.

### 74. Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, et al. (2025). *Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)*. NeurIPS 2025 Datasets & Benchmarks (oral per arXiv comment).

- **Identifiers**: arXiv: [2510.22954](https://arxiv.org/abs/2510.22954) · DOI: [10.52202/085713-2732](https://doi.org/10.52202/085713-2732) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Introduces Infinity-Chat, 26K open-ended real-world queries with a 6-category, 17-subcategory taxonomy, and documents an 'Artificial Hivemind' effect: strong intra-model repetition and inter-model homogeneity in open-ended generation.
- **Implication for the humanizer**: Batch-level semantic homogeneity is a measurable signature even when single texts look human.

### 75. Understanding the Effects of RLHF on LLM Generalisation and Diversity

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu (2024). *Understanding the Effects of RLHF on LLM Generalisation and Diversity*. ICLR 2024 (UNVERIFIED in this pass).

- **Identifiers**: arXiv: [2310.06452](https://arxiv.org/abs/2310.06452) (verified: arXiv)
- **Code/data**: https://github.com/facebookresearch/rlfh-gen-div
- **Summary**: Analyses SFT, reward modelling and RLHF and finds that RLHF generalises better than SFT to new inputs but significantly reduces output diversity across a variety of measures.
- **Implication for the humanizer**: Mechanistic background for why instruction-tuned output is lower in variance; base models are a source of more human-like variance.

### 76. The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text

Yanzhu Guo, Guokan Shang, Michalis Vazirgiannis, and Chloé Clavel (2024). *The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text*. Findings of NAACL 2024.

- **Identifiers**: arXiv: [2311.09807](https://arxiv.org/abs/2311.09807) · DOI: [10.18653/v1/2024.findings-naacl.228](https://doi.org/10.18653/v1/2024.findings-naacl.228) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Recursively trains models on synthetic text and observes a consistent decrease in lexical, syntactic and semantic diversity over iterations, strongest for creative tasks.
- **Implication for the humanizer**: Diversity metrics (lexical, syntactic, semantic) are sensitive to model-generated text and suitable as gap measures.

### 77. Standardizing the Measurement of Text Diversity: A Tool and a Comparative Analysis of Scores

Chantal Shaib, Venkata S. Govindarajan, Joe Barrow, Jiuding Sun, Alexa F. Siu, Byron C. Wallace, and Ani Nenkova (2025). *Standardizing the Measurement of Text Diversity: A Tool and a Comparative Analysis of Scores*. IJCNLP-AACL 2025 (demo).

- **Identifiers**: arXiv: [2403.00553](https://arxiv.org/abs/2403.00553) · DOI: [10.18653/v1/2025.ijcnlp-demo.5](https://doi.org/10.18653/v1/2025.ijcnlp-demo.5) (verified: Crossref DOI)
- **Code/data**: Python library 'diversity' (PyPI v0.3.1); Shaib et al. 2024 (templates paper) state that they extract templates with this library
- **Summary**: Compares text-diversity scores and finds that fast compression ratios capture information similar to slow n-gram homogeneity scores; releases a tool.
- **Implication for the humanizer**: Compression ratio (gzip) over words and POS tags is the cheapest diversity measure for batches.


## G. Robustness benchmarks, humanizers and false positives

### 78. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer (2023). *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*. NeurIPS 2023.

- **Identifiers**: arXiv: [2303.13408](https://arxiv.org/abs/2303.13408) · DOI: [10.52202/075280-1195](https://doi.org/10.52202/075280-1195) (verified: Crossref DOI)
- **Code/data**: https://github.com/martiansideofthemoon/ai-detection-paraphrases
- **Summary**: Builds DIPPER, an 11B paraphraser with control over lexical diversity and content reordering, which evades watermarking, GPTZero, DetectGPT and OpenAI's classifier; DetectGPT accuracy falls from 70.3% to 4.6% at 1% FPR. A retrieval defence over 15M generations still detects 80-97% of paraphrases at 1% FPR.
- **Implication for the humanizer**: Paraphrase destroys token-probability signals but not geometric ones (PHD) or provider-side retrieval; the skill should report which signal families moved.

### 79. Can AI-Generated Text be Reliably Detected?

Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi (2023). *Can AI-Generated Text be Reliably Detected?*. TMLR (per arXiv comment).

- **Identifiers**: arXiv: [2303.11156](https://arxiv.org/abs/2303.11156) (verified: arXiv comment)
- **Code/data**: https://github.com/vinusankars/Reliability-of-AI-text-detectors
- **Summary**: Introduces recursive paraphrasing attacks against watermarking, neural, zero-shot and retrieval detectors on passages of about 300 tokens, and gives a theoretical bound linking the best possible detector to the total-variation distance between human and model text distributions.
- **Implication for the humanizer**: The theoretical target of a deep humanizer is to reduce that distributional distance, which is what the embedding- and distribution-level checks estimate.

### 80. GPT detectors are biased against non-native English writers

Weixin Liang, Mert Yuksekgonul, Yining Mao, Eric Wu, and James Zou (2023). *GPT detectors are biased against non-native English writers*. Patterns 2023.

- **Identifiers**: arXiv: [2304.02819](https://arxiv.org/abs/2304.02819) · DOI: [10.1016/j.patter.2023.100779](https://doi.org/10.1016/j.patter.2023.100779) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Several widely used GPT detectors consistently misclassify non-native English writing samples as AI-generated while classifying native writing correctly; simple prompting strategies both mitigate the bias and bypass the detectors.
- **Implication for the humanizer**: Low lexical diversity and low perplexity are not uniquely 'AI'; the humanizer must not push non-native or plain-style writers toward a single 'native' style, and the check round must report false positives on human references.

### 81. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors

Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch (2024). *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors*. ACL 2024.

- **Identifiers**: arXiv: [2405.07940](https://arxiv.org/abs/2405.07940) · DOI: [10.18653/v1/2024.acl-long.674](https://doi.org/10.18653/v1/2024.acl-long.674) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Over 6 million generations from 11 models across 8 domains, 11 adversarial attacks, 4 decoding strategies and with/without repetition penalty; 12 detectors evaluated at a fixed 5% FPR. Findings: default thresholds of open-source detectors give dangerously high FPRs; repetition penalty lowers accuracy by up to 32 points for all detectors; strong detectors (Fast-DetectGPT, Binoculars, Originality) can go from near-perfect to 95%+ error when generator or decoding changes; Binoculars and other metric-based methods drop by up to 36.1% under synonym swaps; homoglyphs drop five detectors by 40.6% on average; adversarially trained RADAR is more robust.
- **Implication for the humanizer**: Decoding choices (sampling with repetition penalty) are already a strong 'humanizer' for probability-based detectors; the check round must use several signal families and fixed-FPR thresholds.

### 82. MAGE: Machine-generated Text Detection in the Wild

Yafu Li, Qintong Li, Leyang Cui, Wei Bi, Zhilin Wang, Longyue Wang, et al. (2024). *MAGE: Machine-generated Text Detection in the Wild*. ACL 2024.

- **Identifiers**: arXiv: [2305.13242](https://arxiv.org/abs/2305.13242) · DOI: [10.18653/v1/2024.acl-long.3](https://doi.org/10.18653/v1/2024.acl-long.3) (verified: Crossref DOI)
- **Code/data**: https://github.com/yafuly/MAGE
- **Summary**: Large testbed of human and machine text across domains and generators; the authors attribute detection difficulty in the wild to decreasing linguistic distinctions between sources, yet the best detector still identifies 86.54% of out-of-domain texts from a new LLM.
- **Implication for the humanizer**: Useful public data for calibrating the check round across domains.

### 83. M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection

Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, et al. (2024). *M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection*. EACL 2024.

- **Identifiers**: arXiv: [2305.14902](https://arxiv.org/abs/2305.14902) · DOI: [10.18653/v1/2024.eacl-long.83](https://doi.org/10.18653/v1/2024.eacl-long.83) (verified: Crossref DOI)
- **Code/data**: https://github.com/mbzuai-nlp/M4
- **Summary**: Multi-generator, multi-domain, multilingual benchmark for black-box machine-text detection.
- **Implication for the humanizer**: Multilingual human references for non-English use.

### 84. M4GT-Bench: Evaluation Benchmark for Black-Box Machine-Generated Text Detection

Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, et al. (2024). *M4GT-Bench: Evaluation Benchmark for Black-Box Machine-Generated Text Detection*. ACL 2024.

- **Identifiers**: arXiv: [2402.11175](https://arxiv.org/abs/2402.11175) · DOI: [10.18653/v1/2024.acl-long.218](https://doi.org/10.18653/v1/2024.acl-long.218) (verified: Crossref DOI)
- **Code/data**: https://github.com/mbzuai-nlp/M4GT-Bench
- **Summary**: Benchmark with three tasks: mono- and multilingual binary detection, multi-way generator attribution, and boundary detection in mixed human-machine text.
- **Implication for the humanizer**: Boundary-detection task matches the per-span check the skill needs.

### 85. MGTBench: Benchmarking Machine-Generated Text Detection

Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, and Yang Zhang (2024). *MGTBench: Benchmarking Machine-Generated Text Detection*. ACM CCS 2024.

- **Identifiers**: arXiv: [2303.14822](https://arxiv.org/abs/2303.14822) · DOI: [10.1145/3658644.3670344](https://doi.org/10.1145/3658644.3670344) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Unified benchmark framework comparing metric-based and model-based detectors against powerful LLMs under common architectures, datasets and settings.
- **Implication for the humanizer**: Reference implementation suite for baselines.

### 86. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios

Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, and Lidia S. Chao (2024). *DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios*. NeurIPS 2024 Datasets & Benchmarks.

- **Identifiers**: arXiv: [2410.23746](https://arxiv.org/abs/2410.23746) · DOI: [10.52202/079017-3186](https://doi.org/10.52202/079017-3186) (verified: Crossref DOI)
- **Code/data**: https://github.com/NLP2CT/DetectRL
- **Summary**: Real-world benchmark with human text from misuse-prone domains and LLM text perturbed by heuristic rules simulating prompt variation, human revisions (word substitutions) and writing noise (spelling errors); state-of-the-art detectors still underperform.
- **Implication for the humanizer**: Shows that cheap noise (typos, substitutions) lowers detector scores, which is why the skill must not count such noise as genuine humanization.

### 87. Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks

Yichen Wang, Shangbin Feng, Abe Bohan Hou, Xiao Pu, Chao Shen, Xiaoming Liu, Yulia Tsvetkov, and Tianxing He (2024). *Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks*. ACL 2024.

- **Identifiers**: arXiv: [2402.11638](https://arxiv.org/abs/2402.11638) · DOI: [10.18653/v1/2024.acl-long.160](https://doi.org/10.18653/v1/2024.acl-long.160) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Stress-tests detectors under editing, paraphrasing, prompting and co-generating attacks with limited generator access; almost no detector is robust to all attacks, and performance drops by 35% on average across attacks.
- **Implication for the humanizer**: Attack-specific weaknesses differ by detector family, so a single detector is not an adequate check.

### 88. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?

Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, and Zhouwang Yang (2024). *Are AI-Generated Text Detectors Robust to Adversarial Perturbations?*. ACL 2024.

- **Identifiers**: arXiv: [2406.01179](https://arxiv.org/abs/2406.01179) · DOI: [10.18653/v1/2024.acl-long.327](https://doi.org/10.18653/v1/2024.acl-long.327) (verified: Crossref DOI)
- **Code/data**: https://github.com/CarlanLark/Robust-AIGC-Detector
- **Summary**: Proposes a Siamese calibrated reconstruction network (SCRN) that improves robustness to adversarial perturbations by 6.5-18.25% absolute accuracy over the best baseline on four datasets.
- **Implication for the humanizer**: Defence side; reconstruction-based detectors target the same perturbations humanizers use.

### 89. Beemo: Benchmark of Expert-edited Machine-generated Outputs

Ekaterina Artemova, Jason Lucas, Saranya Venkatraman, Jooyoung Lee, Sergei Tilga, Adaku Uchendu, and Vladislav Mikhailov (2025). *Beemo: Benchmark of Expert-edited Machine-generated Outputs*. NAACL 2025.

- **Identifiers**: arXiv: [2411.04032](https://arxiv.org/abs/2411.04032) · DOI: [10.18653/v1/2025.naacl-long.357](https://doi.org/10.18653/v1/2025.naacl-long.357) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: 6.5k texts written by humans, generated by ten instruction-tuned LLMs and edited by experts, plus 13.1k machine-generated and LLM-edited texts; benchmarks 33 detector configurations across edit types.
- **Implication for the humanizer**: Expert-edited machine text is the closest public proxy for 'well-humanized' text; good held-out test data.

### 90. People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text

Jenna Russell, Marzena Karpinska, and Mohit Iyyer (2025). *People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text*. ACL 2025.

- **Identifiers**: arXiv: [2501.15654](https://arxiv.org/abs/2501.15654) · DOI: [10.18653/v1/2025.acl-long.267](https://doi.org/10.18653/v1/2025.acl-long.267) (verified: Crossref DOI)
- **Code/data**: https://github.com/jenna-russell/human_detectors
- **Summary**: Annotators who frequently use ChatGPT for writing detect AI-generated articles (300 articles) nearly perfectly, outperforming almost all automatic detectors; only Pangram matches them. On articles generated and humanized by O1-PRO, the expert majority vote has 100% TPR, while Binoculars detects 6.7% and Fast-DetectGPT 23.3%. Experts rely heavily on 'AI vocabulary' but also on more complex phenomena that automatic detectors struggle to assess.
- **Implication for the humanizer**: Open zero-shot detectors are weak judges of humanized text; the check round should include an expert-style rubric (via an LLM judge or human review) and, where allowed, a strong commercial detector.

### 91. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack

Ying Zhou, Ben He, and Le Sun (2024). *Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack*. LREC-COLING 2024.

- **Identifiers**: arXiv: [2404.01907](https://arxiv.org/abs/2404.01907) · DOI: [10.63317/5ocyse7sux8y](https://doi.org/10.63317/5ocyse7sux8y) (verified: Crossref DOI (as returned) + arXiv comment)
- **Code/data**: none found in paper/abstract
- **Summary**: Adversarial attacks against fine-tuned detectors can make machine text be classified as human in as little as 10 seconds.
- **Implication for the humanizer**: Classifier-guided edits are cheap; evasion of one classifier is not evidence of human-likeness.

### 92. AuthorMist: Evading AI Text Detectors with Reinforcement Learning

Isaac David and Arthur Gervais (2025). *AuthorMist: Evading AI Text Detectors with Reinforcement Learning*. arXiv preprint (2025); venue not verified.

- **Identifiers**: arXiv: [2503.08716](https://arxiv.org/abs/2503.08716) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Fine-tunes a 3B model with GRPO using detector feedback as reward to paraphrase AI text; attack success rates of 78.6-96.2% against individual detectors with semantic similarity above 0.94.
- **Implication for the humanizer**: Detector-reward RL works at small scale, but optimising against detectors is exactly what the skill should avoid if the goal is genuine human-likeness rather than evasion.

### 93. Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text

Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, and Soheil Feizi (2025). *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text*. NeurIPS 2025.

- **Identifiers**: arXiv: [2506.07001](https://arxiv.org/abs/2506.07001) · DOI: [10.52202/085713-1588](https://doi.org/10.52202/085713-1588) (verified: Crossref DOI)
- **Code/data**: https://github.com/chengez/Adversarial-Paraphrasing
- **Summary**: Training-free attack: an instruction-tuned LLM paraphrases while, at each step, a detector picks the candidate token with the lowest AI score. Simple paraphrasing actually increases TPR at 1% FPR by 8.57% (RADAR) and 15.03% (Fast-DetectGPT), whereas detector-guided paraphrasing (guided by OpenAI-RoBERTa-Large) reduces it by 64.49% and 98.96%, with an average reduction of 87.88% across eight detectors.
- **Implication for the humanizer**: Plain LLM paraphrasing makes text more machine-like for some detectors. Any detector used to guide rewriting must be excluded from the evaluation set (held-out detectors only).

### 94. Base Models Look Human To AI Detectors

Yixuan Even Xu, Ziqian Zhong, Aditi Raghunathan, Fei Fang, and J. Zico Kolter (2026). *Base Models Look Human To AI Detectors*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2605.19516](https://arxiv.org/abs/2605.19516) (verified: arXiv)
- **Code/data**: https://github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing
- **Summary**: GPTZero and Pangram often judge text from base models as overwhelmingly human but text from their instruction-tuned counterparts as AI. Humanization by Iterative Paraphrasing (HIP) minimally fine-tunes a base model into a paraphraser and applies it repeatedly; across Llama-3 and Qwen-3 families from 0.6B to 70B it improves detector human-likeness with a better semantic-preservation trade-off than baselines. The authors conclude that detectors track instruction-tuning artefacts and local context more than an invariant notion of machine text.
- **Implication for the humanizer**: Base-model likelihood is the natural 'human-likeness' reference: the observer/performer gap in Binoculars-style scoring and the base-vs-instruct preference are measurable on CPU with Qwen3-0.6B or Qwen2.5-0.5B pairs.

### 95. ARB: A Matched Authorship-Rewriting Benchmark Dataset for AI-Text Detector Evaluation

Gaetano Perrone and Simon Pietro Romano (2026). *ARB: A Matched Authorship-Rewriting Benchmark Dataset for AI-Text Detector Evaluation*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2607.29539](https://arxiv.org/abs/2607.29539) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: 1,800 human texts (XSum, WritingPrompts, OpenWebText) and four open generators (Llama-3.2-3B, Qwen2.5-7B, Mistral-7B, Gemma-2-9B) yield matched human, direct-LLM, LLM-rewritten-human and LLM-rewritten-LLM variants. At 1% FPR, Fast-DetectGPT and Binoculars (Falcon-7B) detect 91.2% and 93.5% of direct LLM text but only 30.8% and 15.1% of LLM-rewritten human text; same-model rewriting of LLM text keeps 78.3% and 83.0%. RADAR drops from 66.8% to 12.2%; BERT- and RoBERTa-based defences stay below 3%.
- **Implication for the humanizer**: Probability-based detectors key on content origin more than on surface style; for a humanizer this means rewriting with an LLM does not by itself make AI-origin text pass, and human-origin text revised by an LLM is largely missed.

### 96. Humanized but Still True? Auditing Facts and Detection in AI-Text Humanizers

Ziyuan Wang, Genliang Zhu, and Xiaobo Chen (2026). *Humanized but Still True? Auditing Facts and Detection in AI-Text Humanizers*. arXiv preprint (Oct 2026); venue not verified.

- **Identifiers**: arXiv: [2611.00359](https://arxiv.org/abs/2611.00359) (verified: arXiv)
- **Code/data**: https://github.com/zwang-devv/humanizer-fact-audit
- **Summary**: Checks whether numbers, dates, person/organisation names and quoted spans survive humanization. In the October 2026 cycle of a public humanizer leaderboard all 11 commercial tools changed or dropped at least one such unit in 35-65% of documents containing one (September cycle 19-50%), mostly quoted spans; counting only numbers and names, 10 of 11 tools still altered at least one in 20% or more of documents. An open 12B humanizer kept at least 97% of numbers and names but reworded 67-78% of quoted spans. Retention and evasion were not clearly related (Spearman rho = -0.46, 95% CI -0.65 to 0.08).
- **Implication for the humanizer**: The skill needs a hard fact-preservation gate (numbers, dates, names, quotes verbatim) in its check round.

### 97. On the Indistinguishability of Human v/s AI Generated Text

Jaee Ponde, Aritra Das, Mihir More, and Debayan Gupta (2026). *On the Indistinguishability of Human v/s AI Generated Text*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2608.26797](https://arxiv.org/abs/2608.26797) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Theoretical analysis of using human writing samples to paraphrase machine responses toward the human distribution; under mixing and stability conditions repeated paraphrasing converges to the human target, with explicit rates that depend on semantic-preservation and quality constraints and on the amount of human data.
- **Implication for the humanizer**: Formal support for a reference-guided iterative loop (rewrite, measure distance to human reference, repeat) with a semantic-preservation constraint.

### 98. Beyond Easy Wins: A Text Hardness-Aware Benchmark for LLM-generated Text Detection

Navid Ayoobi, Sadat Shahriar, and Arjun Mukherjee (2025). *Beyond Easy Wins: A Text Hardness-Aware Benchmark for LLM-generated Text Detection*. arXiv preprint (2025); venue not verified.

- **Identifiers**: arXiv: [2507.15286](https://arxiv.org/abs/2507.15286) (verified: arXiv)
- **Code/data**: https://github.com/navid-aub/SHIELD-Benchmark
- **Summary**: Benchmark (SHIELD) that folds reliability at low FPR and stability across domains and attacks into one metric, plus a model-agnostic 'humanification' framework with a controllable hardness parameter that challenges zero-shot detectors.
- **Implication for the humanizer**: A hardness-controlled humanizer is a useful baseline to compare against.

### 99. Group-Adaptive Threshold Optimization for Robust AI-Generated Text Detection

Minseok Jung, Cynthia Fuertes Panizo, Liam Dugan, Yi R., Fung, Pin-Yu Chen, and Paul Pu Liang (2025). *Group-Adaptive Threshold Optimization for Robust AI-Generated Text Detection*. arXiv preprint (2025); venue not verified.

- **Identifiers**: arXiv: [2502.04528](https://arxiv.org/abs/2502.04528) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: With a fixed global threshold, detectors make more false positives on shorter human-written texts and more positive calls on neurotic writing styles among long texts; FairOPT learns group-specific thresholds.
- **Implication for the humanizer**: Threshold by length (and, where possible, by register) rather than globally.

### 100. Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI

Yuxia Wang, Rui Xing, Jonibek Mansurov, Giovanni Puccetti, Zhuohan Xie, Minh Ngoc Ta, et al. (2026). *Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI*. ACL 2026.

- **Identifiers**: arXiv: [2502.11614](https://arxiv.org/abs/2502.11614) · DOI: [10.18653/v1/2026.acl-long.639](https://doi.org/10.18653/v1/2026.acl-long.639) (verified: Crossref DOI)
- **Code/data**: https://github.com/xnlp-lab/HumanEval-MGT
- **Summary**: Across 16 datasets in 9 languages and 9 domains, 19 annotators reach 87.6% average accuracy at telling human from machine text. The main gaps are concreteness, cultural nuance and diversity; explicitly prompting LLMs about these partially closes the gap in over 50% of cases; humans do not always prefer human text.
- **Implication for the humanizer**: Concreteness and cultural specificity are rewrite targets that no embedding detector measures directly; an LLM-judge rubric can.

### 101. Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods

Kathleen C. Fraser, Hillary Dawkins, and Svetlana Kiritchenko (2025). *Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods*. JAIR 82 (2025).

- **Identifiers**: arXiv: [2406.15583](https://arxiv.org/abs/2406.15583) · DOI: [10.1613/jair.1.16665](https://doi.org/10.1613/jair.1.16665) (verified: arXiv DOI + journal_ref)
- **Code/data**: none found in paper/abstract
- **Summary**: Survey of watermarking, statistical, stylistic and classifier-based AI-text detection and of the factors that determine detectability.
- **Implication for the humanizer**: Background survey.

### 102. A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions

Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, and Lidia S. Chao (2025). *A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions*. Computational Linguistics 2025.

- **Identifiers**: arXiv: [2310.14724](https://arxiv.org/abs/2310.14724) · DOI: [10.1162/coli_a_00549](https://doi.org/10.1162/coli_a_00549) (verified: Crossref DOI)
- **Code/data**: https://github.com/NLP2CT/LLM-generated-Text-Detection
- **Summary**: Survey of LLM-generated text detection: necessity, methods (watermarking, zero-shot, fine-tuned, adversarial), benchmarks and open problems.
- **Implication for the humanizer**: Background survey.

### 103. A Measurement Perspective on Benchmarking LLM-Generated Text Detectors using Item Response Theory

Yanbin Zhao, Zhipeng Chen, Guangsheng Bao, Guandong Zhang, and Yan Li (2026). *A Measurement Perspective on Benchmarking LLM-Generated Text Detectors using Item Response Theory*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2610.24972](https://arxiv.org/abs/2610.24972) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Applies multidimensional item response theory to 32K human and machine texts in four domains scored by 14 detectors; the item-parameter space shows distinct but overlapping human and machine distributions with strong variation across domains and lengths.
- **Implication for the humanizer**: Per-text difficulty varies widely; report distributions of scores, not single averages.

### 104. 'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions

Nils Dycke, Marina Sakharova, Nico Daheim, and Iryna Gurevych (2026). *'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions*. arXiv preprint (2026); venue not verified.

- **Identifiers**: arXiv: [2606.04906](https://arxiv.org/abs/2606.04906) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Defines several notions of AI-generated text and releases AITDNA, human-machine co-constructed texts with full edit and AI-interaction histories; detectors work for specific notions only.
- **Implication for the humanizer**: The skill should state which notion of 'AI text' its check targets (fully generated vs AI-edited human text).

### 105. Understanding the Effects of Human-written Paraphrases in LLM-generated Text Detection

Hiu Ting Lau and Arkaitz Zubiaga (2024). *Understanding the Effects of Human-written Paraphrases in LLM-generated Text Detection*. arXiv preprint (2024); venue not verified.

- **Identifiers**: arXiv: [2411.03806](https://arxiv.org/abs/2411.03806) (verified: arXiv)
- **Code/data**: none found in paper/abstract
- **Summary**: Collects HLPC, combining human and LLM texts with human-written and LLM paraphrases, and studies how human paraphrases affect OpenAI RoBERTa and watermark detectors.
- **Implication for the humanizer**: Human paraphrase is a distinct perturbation from LLM paraphrase; useful negative-control data.

### 106. Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training

Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, and Xiaoming Liu (2025). *Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training*. ACL 2025 (per arXiv comment).

- **Identifiers**: arXiv: [2502.12734](https://arxiv.org/abs/2502.12734) (verified: arXiv comment)
- **Code/data**: https://github.com/Liyuuuu111/GREATER
- **Summary**: Adversarial training framework in which an adversary perturbs critical tokens in embedding space with greedy search and pruning while a detector learns to defend, generalising to other attacks.
- **Implication for the humanizer**: Defence side of token-level attacks.

### 107. OpenTuringBench: An Open-Model-based Benchmark and Framework for Machine-Generated Text Detection and Attribution

Lucio La Cava and Andrea Tagarelli (2025). *OpenTuringBench: An Open-Model-based Benchmark and Framework for Machine-Generated Text Detection and Attribution*. arXiv preprint (2025); venue not verified.

- **Identifiers**: arXiv: [2504.11369](https://arxiv.org/abs/2504.11369) (verified: arXiv)
- **Code/data**: https://huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench
- **Summary**: Benchmark built on open LLMs for detection and attribution, including human/machine-manipulated, out-of-domain and unseen-model tasks, with a contrastive detector (OTBDetector).
- **Implication for the humanizer**: Open-model data suited to local calibration.

### 108. Counter Turing Test ($CT^2$): Investigating AI-Generated Text Detection for Hindi -- Ranking LLMs based on Hindi AI Detectability Index ($ADI_{hi}$)

Ishan Kavathekar, Anku Rani, Ashmit Chamoli, Ponnurangam Kumaraguru, Amit Sheth, and Amitava Das (2024). *Counter Turing Test ($CT^2$): Investigating AI-Generated Text Detection for Hindi -- Ranking LLMs based on Hindi AI Detectability Index ($ADI_{hi}$)*. Findings of EMNLP 2024 (per arXiv comment).

- **Identifiers**: arXiv: [2407.15694](https://arxiv.org/abs/2407.15694) (verified: arXiv comment)
- **Code/data**: https://github.com/ishank31/Counter_Turing_Test
- **Summary**: Hindi AI-text detection study covering 26 LLMs and a new Hindi news dataset; evaluates ConDA, J-Guard, RADAR, Raidar and intrinsic-dimension estimation.
- **Implication for the humanizer**: One of the few non-English evaluations of the PHD detector.


## H. Foundations used by the signals

### 109. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings

Kawin Ethayarajh (2019). *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings*. EMNLP-IJCNLP 2019.

- **Identifiers**: arXiv: [1909.00512](https://arxiv.org/abs/1909.00512) · DOI: [10.18653/v1/d19-1006](https://doi.org/10.18653/v1/d19-1006) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Shows that contextual representations in BERT, ELMo and GPT-2 are anisotropic (they occupy a narrow cone), increasingly so in upper layers, and that less than 5% of the variance in a word's representations can be explained by a static embedding.
- **Implication for the humanizer**: Any cosine-based embedding comparison must correct for anisotropy (mean-centering or whitening against the reference corpus) before distances are interpreted.

### 110. Representation Degeneration Problem in Training Natural Language Generation Models

Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, and Tie-Yan Liu (2019). *Representation Degeneration Problem in Training Natural Language Generation Models*. ICLR 2019.

- **Identifiers**: arXiv: [1907.12009](https://arxiv.org/abs/1907.12009) (verified: arXiv comment)
- **Code/data**: none found in paper/abstract
- **Summary**: Identifies representation degeneration: likelihood training pushes word embeddings into a narrow cone, and proposes a regulariser.
- **Implication for the humanizer**: Background for anisotropy; no paper found that uses anisotropy itself as an AI-text detector (see gaps).

### 111. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

Nils Reimers and Iryna Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP 2019.

- **Identifiers**: arXiv: [1908.10084](https://arxiv.org/abs/1908.10084) · DOI: [10.18653/v1/d19-1410](https://doi.org/10.18653/v1/d19-1410) (verified: Crossref DOI)
- **Code/data**: none found in paper/abstract
- **Summary**: Siamese/triplet fine-tuning of BERT for sentence embeddings comparable with cosine similarity; basis of the sentence-transformers library.
- **Implication for the humanizer**: Tooling for semantic-preservation checks (source vs humanized) and for semantic dispersion measures.


## Gaps, negative results of the search, and UNVERIFIED items

1. **Anisotropy as a detector**: an arXiv search for abstracts containing both 'anisotropy' and 'machine-generated text' returned no results, and no paper in this survey uses embedding anisotropy itself as an AI-text signal. Anisotropy enters only as a correction (Ethayarajh 2019; Gao et al. 2019). Any anisotropy-based gap metric in the skill is exploratory and must be validated locally.
2. **Small-model Binoculars**: no paper found that evaluates Binoculars with sub-2B model pairs (the published pairs are Falcon-7B/Falcon-7B-Instruct and LLaMA/Llama-2 7B and 13B). The CPU pair recommended in signals_table.md is untested.
3. **Kirk et al. (2310.06452)**: ICLR 2024 venue not confirmed in this pass (arXiv record has no venue).
4. **Liang et al. (2404.01268)**: the COLM 2024 venue for the preprint was not confirmed; the Nature Human Behaviour 2025 version was confirmed via Crossref (same 14 authors).
5. **BiScope**: bibliographic record verified (Crossref), method summary from memory and UNVERIFIED.
6. **Doshi and Hauser (2024)**: record verified; content summarised from the title only.
7. **Zanotto and Aroyehun**: relationship between arXiv 2412.03025 and the EMNLP 2025 paper with a different title is not verified.
8. **Milicka et al.**: the IJCL 2026 title ('linguistic variation') differs from the arXiv title ('stylistic variation'); same authors and topic, treated as the journal version but not confirmed.
9. **Zhou et al. (LREC-COLING 2024)**: Crossref returned DOI 10.63317/5ocyse7sux8y, an unusual prefix for this venue; reported as returned.
10. **TCD (EMNLP 2026)**: Crossref returned DOI 10.18653/2026.emnlp-main.1708 (no 'v1' segment); reported as returned, not opened.
11. **Levina and Bickel (2004), MLE intrinsic dimension; Facco et al. (2017), TwoNN**: cited through Tulchinskii et al.'s reference list only; not independently verified here.
12. **Hugging Face model IDs** named in signals_table.md were checked to exist through the Hugging Face API (HTTP 200); their suitability as detectors is a recommendation, not a published result, unless stated.
13. Several 2026 preprints (late-stage volatility, length calibration, ARB, HIP, humanizer fact audit, SV-Detect, DependencyAI, IRT benchmarking, AITDNA) are recent and unreviewed; their numbers should be treated as provisional.
