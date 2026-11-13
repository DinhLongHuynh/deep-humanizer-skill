# Validation of the deep-humanizer toolkit (November 2026)

## 1. Corpus and calibration

- Human: 120 texts written before ChatGPT. 40 arXiv abstracts (2018-2020, 8 categories), 40 CNN articles (CNN/DailyMail test split), 40 WritingPrompts stories (Reddit, 2018 dataset). Each is truncated at a sentence boundary to about 220-380 words.
- AI: 120 texts matched one-to-one by prompt (paper title, news highlights, story prompt), rotating claude-haiku-4-5, claude-sonnet-5, gpt-4.1-mini and gpt-5.4-mini.
- `assets/reference_default` uses all 240 texts. The validation below uses `bank_train`, calibrated without the 30 held-out topic groups (10 per register, generators balanced).

Grouped-CV AUROC, human vs AI (default bank): combined local model **0.978**; per layer {"surface": 0.78, "stylometry": 0.947, "lm": 0.87, "embedding": 0.868}. Train bank combined: 0.980. The CLEAN cut-off (human 75th percentile of out-of-fold p) is 0.149.

| feature | AUROC | direction | human median | AI median |
|---|---|---|---|---|
| `det.tabularisai` | 0.985 | higher_is_ai | 0.000573 | 0.978 |
| `det.fakespot` | 0.965 | higher_is_ai | 0.0353 | 1 |
| `det.desklib` | 0.948 | higher_is_ai | 0.0356 | 1 |
| `emb.ref.emb_lr_p_ai` | 0.831 | higher_is_ai | 0.133 | 0.823 |
| `srf.surface_score_per_1k` | 0.813 | higher_is_ai | 6.77 | 21.2 |
| `sty.mtld` | 0.803 | higher_is_ai | 98.5 | 137 |
| `sty.hapax_ratio` | 0.789 | higher_is_ai | 0.446 | 0.504 |
| `lm.late_local_volatility` | 0.787 | lower_is_ai | 2.51 | 2.16 |
| `lm.late_deriv_dispersion` | 0.783 | lower_is_ai | 2.51 | 2.17 |
| `sty.mattr50` | 0.783 | higher_is_ai | 0.821 | 0.856 |
| `lm.surprisal_sd` | 0.778 | lower_is_ai | 2.69 | 2.37 |
| `srf.ai_vocab` | 0.754 | higher_is_ai | 3.83 | 16.4 |
| `lm.surprisal_p95` | 0.750 | lower_is_ai | 8.13 | 7.07 |
| `lm.gltr_beyond` | 0.745 | lower_is_ai | 0.0129 | 0.00638 |
| `lm.binoculars` | 0.741 | lower_is_ai | 1.03 | 0.98 |
| `sty.mean_word_len` | 0.729 | higher_is_ai | 4.81 | 5.52 |
| `sty.long_word_ratio` | 0.727 | higher_is_ai | 0.238 | 0.368 |
| `lm.uid_span_range` | 0.727 | lower_is_ai | 3.03 | 2.53 |
| `lm.uid_span_max` | 0.726 | lower_is_ai | 4.35 | 3.78 |
| `lm.uid_local` | 0.720 | lower_is_ai | 14.3 | 11.5 |

Departures from the 2023 literature, now reflected in SKILL.md:
- PHD intrinsic dimension is higher for 2025-26 generators than for human text (median 10.10 vs 9.70, AUROC 0.57). Tulchinskii et al. reported lower PHD for 2023 models. The implementation follows their recipe.
- Current models show higher lexical diversity than humans (MTLD median 137 vs 98, AUROC 0.80), so the skill tells the editor to reuse key terms.
- The 2019 GPT-2 RoBERTa detector is near chance (AUROC 0.58). The RAID-trained open classifiers separate raw AI text almost perfectly (tabularisai 0.985, fakespot 0.965, desklib 0.948).

## 2. Humanization experiment (30 held-out AI texts)

Conditions:
- **ai**: the raw generation.
- **shallow**: one "make it human" LLM prompt with a banned-word list, no dashes and contractions.
- **deep_v1**: an LLM editor following SKILL.md Steps 1-4. It gets the check report each pass and makes up to 3 passes. A meaning gate follows each pass with one fix allowed. The kept version has the lowest combined p among those passing the gate. Detectors were held out throughout. Both shallow and deep used the same rewriter model.

### 2.1 Deep loop: local layers on the train bank, median over 30 documents

|   pass |   combined |   L.surface |   L.stylometry |   L.lm |   L.embedding |
|-------:|-----------:|------------:|---------------:|-------:|--------------:|
|      0 |      0.966 |       0.76  |          0.947 |  0.824 |         0.841 |
|      1 |      0.819 |       0.416 |          0.874 |  0.659 |         0.749 |
|      2 |      0.764 |       0.337 |          0.908 |  0.59  |         0.684 |
|      3 |      0.706 |       0.321 |          0.92  |  0.524 |         0.707 |

Verdicts by pass:

|   pass |   BORDERLINE |   CLEAN |   FLAGGED |
|-------:|-------------:|--------:|----------:|
|      0 |            3 |       0 |        27 |
|      1 |            6 |       1 |        23 |
|      2 |            5 |       0 |        24 |
|      3 |            3 |       1 |        25 |

Selected versions: median combined p 0.669 (mean 0.616), against a raw-AI median of 0.966. Verdicts: {'FLAGGED': 22, 'CLEAN': 2, 'BORDERLINE': 6}. Meaning gate on selected versions: {'PASS': 30}.

### 2.2 Check round and held-out detectors by condition

Documents scored: {'human': 30, 'ai': 30, 'shallow': 30, 'deep_v1': 1}. Scoring stopped at the deadline, so conditions with fewer than 30 documents are partial.

Median scores:

| cond    |   combined |   layer.surface |   layer.stylometry |   layer.lm |   layer.embedding |   det.tabularisai |   det.desklib |   det.fakespot |   det.tropa_mini |
|:--------|-----------:|----------------:|-------------------:|-----------:|------------------:|------------------:|--------------:|---------------:|-----------------:|
| human   |      0.086 |           0.296 |              0.08  |      0.352 |             0.138 |             0.001 |         0.032 |          0.019 |            0.004 |
| ai      |      0.966 |           0.76  |              0.947 |      0.824 |             0.841 |             0.982 |         1     |          1     |            1     |
| shallow |      0.784 |           0.347 |              0.682 |      0.644 |             0.737 |             0.114 |         0.999 |          0.998 |            0.999 |
| deep_v1 |      0.851 |           0.354 |              0.993 |      0.577 |             0.722 |             0.001 |         0.096 |          0.716 |            1     |

Share of documents with detector P(AI) >= 0.5:

| cond    |   det.tabularisai |   det.desklib |   det.fakespot |   det.tropa_mini |
|:--------|------------------:|--------------:|---------------:|-----------------:|
| human   |              0    |          0.17 |           0.07 |             0    |
| ai      |              0.67 |          0.9  |           0.93 |             0.97 |
| shallow |              0.4  |          0.93 |           0.97 |             0.83 |
| deep_v1 |              0    |          0    |           1    |             1    |

Meaning preservation (mean document cosine to source, length ratio):

| cond    |   doc_cos |   len_ratio |
|:--------|----------:|------------:|
| ai      |   nan     |     nan     |
| deep_v1 |     0.924 |       0.986 |
| shallow |     0.973 |       1.068 |

### 2.3 Register features (test-set medians)

| feature | human | raw AI | deep v1 |
|---|---|---|---|
| nominalisations per 1k | 26.7 | 57.1 | 42.6 |
| phrasal coordination per 1k | 13.8 | 19.2 | 21.5 |
| nouns per 1k | 223.7 | 272.2 | 271.0 |
| MTLD | 106.5 | 146.2 | 139.7 |
| POS compression ratio | 4.55 | 4.72 | 4.59 |

Deep v1 lowered nominalisations and template repetition. Coordination, noun density and lexical diversity stayed near the raw-AI values. In response, check.py now lists **syntax hotspots** (register markers by sentence, plus long words used only once), and Pass B requires working through them. A second run with this change (deep_v2) completed pass 1 before the deadline, with pass-1 median combined p 0.819, the same as v1. Its full effect is **not yet measured**.

### 2.4 Batch test (`batch_compare.py`, 30 held-out documents vs the bank's human texts)

- Human test batch: INDISTINGUISHABLE (C2ST on embeddings 0.36, C2ST on features 0.46, 0 features significant after Bonferroni).
- Raw AI test batch: SEPARABLE (C2ST on embeddings 0.63, C2ST on features 0.90, 12 significant features).

## 3. Summary

- The check round separates human from current-model AI text well (combined AUROC about 0.98). Stylometry is the strongest local layer, surface the weakest.
- Skill-guided LLM editing moved the surface and LM layers toward the human band and preserved meaning. It moved stylometry and embedding much less, and only 2 of 30 documents reached CLEAN within 3 passes. Held-out detector scores for the deep rewrites were **not assessed** (1 of 30 documents scored before the deadline). For the shallow rewrite, desklib and fakespot still flagged 93-97 percent of documents, while tabularisai's flag rate fell from 67 to 40 percent, a sign of how fragile a single detector's verdict is. This is consistent with Reinhart et al. 2025 and Rallapalli et al. 2026: register fingerprints survive LLM editing.
- Consequence for use: substantive change needs author samples, specific content from the user, or human editing of the syntax-hotspot sentences. The skill reports what remains and never promises a detector pass.
- Limitations: 240-text bank, three registers, 150-400 words, one rewriter model, English only. Commercial APIs were not exercised (no keys). deep_v2 was not completed.

Code: `validation_code/deep_loop.py` (agent emulation) and `validation_code/analyze.py` (scoring).
