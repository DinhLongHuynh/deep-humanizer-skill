# Detectors: what the check round runs and how to read it

## How `detectors.py` uses this catalogue

| Set | Members | When |
|---|---|---|
| Default open (`check.py`) | `tabularisai` (ModernBERT, RAID-trained), `desklib` (DeBERTa-v3-large), `fakespot` (RoBERTa, not RAID-trained), `tropa_mini` (4-class, added Nov 2026 after it alone agreed with ZeroGPT) | Every check; a score >= 0.5 from an informative one overrides CLEAN |
| Final open (`check.py --final`) | same set as the default since Nov 2026 (kept for compatibility) | Step 5 confirmation |
| Legacy (`--open all`) | `openai_gpt2` (2019), `hc3_chatgpt` (2023) | Only to show how old detectors behave; near chance on current models |
| Commercial (`--external NAME[,NAME]` or `--external available`) | gptzero, pangram, originality, sapling, winston, hive, copyleaks, zerogpt (unverified) | Only with keys in the environment and the user's consent to send the text |

Environment variables: `GPTZERO_API_KEY` (optional `GPTZERO_MODEL_VERSION`), `PANGRAM_API_KEY` (optional `PANGRAM_MODEL`), `ORIGINALITY_API_KEY`, `SAPLING_API_KEY` (optional `SAPLING_MODEL_VERSION`, default pinned to 20261104), `WINSTON_API_KEY`, `HIVE_API_KEY`, `COPYLEAKS_EMAIL` + `COPYLEAKS_API_KEY`, `ZEROGPT_API_KEY`. Each adapter normalises to P(AI) explicitly (Winston and ZeroGPT return human scores and are inverted; Copyleaks and Pangram return the share of text classified AI; Originality's model-5.0 response has no document probability, so the mean sentence score is reported as a proxy). Adapters request no-storage options where the API offers them (`storeScan: false`, `zero_retention: true`, `public_dashboard_link: false`) and record model versions. They follow the official documentation as read on 2026-11-13 but were not exercised with live keys; inspect the output of the first call.

Rules for using detector scores in this skill:

1. They are held out. Never select or iterate rewrites on their scores; use the local layers for that.
2. Pin model versions for before/after comparisons; commercial models change without notice.
3. Score whole documents. Sentence scores are diagnostics; detectors are unreliable below their minimum lengths (Copyleaks 255 characters, Sapling 300, Winston 600).
4. Report humanizer-specific outputs separately (tropa-mini `p_humanized`, Pangram `humanized_windows`, GPTZero `ai_subclass`). Moving from "AI" to "humanized" is not moving to "human".
5. Disagreement is normal. RAID leaders are mostly RAID-tuned and RAID's generators are 2023-era; the validation in `validation.md` shows how the open set behaves on 2025-2026 model output.

The full catalogue compiled by the detector research track follows (endpoints, schemas, licences, RAID numbers, independent accuracy evidence, verification status). Machine-readable specs are in `assets/detector_api_specs.json`.

---

## Detector catalogue (research track, 2026-11-13)

Status date: 2026-11-13. Companion file: `detector_api_specs.json`, which holds the machine-readable request and response templates for every tool below.

Sources were checked as follows. Official vendor API documentation was fetched directly from docs.gptzero.me, docs.pangram.com, docs.originality.ai, docs.copyleaks.com, sapling.ai, docs.gowinston.ai, docs.thehive.ai and developer.grammarly.com. HuggingFace model IDs were checked through the HF API, using config.json and the model card. RAID leaderboard numbers were parsed from the `leaderboard/submissions/*/results.json` files in github.com/liamdugan/raid (repository last pushed 2026-11-12). Papers were verified via the arXiv export API and Crossref. Anything that could not be confirmed is marked **UNVERIFIED**.

---

## 1. Recommendation for the check round

### 1.1 Default open-source set (CPU, no keys)

| Rank | Detector | Type | Params | Why it is in the set | Measured CPU time* |
|---|---|---|---|---|---|
| 1 | `tabularisai/ai-text-detection` | ModernBERT-base classifier, MIT | 149.6 M | Highest-scoring open classifier on RAID with a standard loader (TPR 0.990 at 5 % FPR, 0.959 at 1 %). | ~1.7–2.4 s |
| 2 | `desklib/ai-text-detector-v1.01` | DeBERTa-v3-large + mean-pool head, MIT | 434 M | Different backbone and training data from rank 1, which gives ensemble diversity (RAID 0.912). | ~5–10 s |
| 3 | `fakespot-ai/roberta-base-ai-text-detection-v1` | RoBERTa-base, Apache-2.0 | 124.6 M | Not trained on RAID, so it acts as an out-of-distribution check on ranks 1–2. | ~1.2–1.7 s |
| 4 | `wasitaigeneratedcom/tropa-mini-ai-text-detector` | DeBERTa-v3-large 4-class head, Apache-2.0 | 434 M | The only open model with an explicit `humanized` class; report `p[humanized]` separately. Its accuracy evidence comes only from the vendor. | ~5–9 s |
| 5 | Binoculars with `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen2.5-1.5B-Instruct` | zero-shot, BSD-3 | 2 × 1.54 B | Training-free and token-level, so it can be read against the humanizer's own surprisal features. The published thresholds apply only to the Falcon-7B pair, so the threshold must be **recalibrated locally**. | not run (est. tens of s) |
| 6 | Fast-DetectGPT (analytic) with a Qwen2.5-1.5B pair | zero-shot, MIT | 1.54 B | A second zero-shot criterion (conditional probability curvature). Its probability mapping must be refit for non-default models. | not run |
| fast mode | `MayZhou/e5-small-lora-ai-generated-detector` | e5-small (BERT) + merged LoRA, MIT | 33.4 M | About 0.6 s per text. Fine for inner-loop screening. It collapses on homoglyphs (RAID 0.111), so apply NFKC normalisation first. | ~0.6–0.7 s |

\*One ~120-token text, 8 threads, transformers/torch in the `humanizer-nlp` env. The first call includes warm-up. See `cpu_smoke_test` in the JSON.

Smoke-test sanity check (P(AI) on one LLM-style paragraph vs one informal human paragraph):

| Model | P(AI), AI-style text | P(AI), human-style text | Direction correct |
|---|---|---|---|
| e5-small-lora | 0.9546 | 0.1816 | yes |
| fakespot RoBERTa | 0.9999 | 0.0113 | yes |
| Oxidane tmr | 0.9873 | 0.0284 | yes |
| tabularisai ModernBERT | 0.9863 | 0.0013 | yes |
| desklib v1.01 | 1.0 | 0.0122 | yes |
| tropa-mini (1 − p[human]) | 1.0 | 0.2973 | yes |
| openai RoBERTa-base (index 0 = Fake) | 0.0002 | 0.0002 | no: scores both as "Real", the known failure on post-2019 text |

These two probes confirm label direction and CPU load only. They are not an accuracy estimate.

Do not use these as verdicts: `openai-community/roberta-*-openai-detector` (RAID 0.518 / 0.507), `Hello-SimpleAI/chatgpt-detector-roberta` (RAID 0.266, no licence tag), `TrustSafeAI/RADAR-Vicuna-7B` (RAID 0.639, non-commercial), `SuperAnnotate/ai-detector` (RAID 0.649, custom licence and pip package). They are acceptable as historical baselines in a validation report.

### 1.2 Commercial detectors, when the user supplies a key

| Rank | Service | Why | Main caveat |
|---|---|---|---|
| 1 | **Pangram** (`pangram-4`) | Strongest independent evidence (Jabarian & Imas 2025, NBER w34223): the only detector at "policy-grade" FNR/FPR across four frontier LLMs, robust to StealthGPT. Humanizer head per window (`is_humanized`, `humanizer_score`). | Async (POST then poll). Not on RAID. Price only via pangram.com/pricing (UNVERIFIED here). Minor versions change inside a series. |
| 2 | **GPTZero** | Synchronous. Three-way document classes (human / ai / mixed), `ai_paraphrased` subclass, per-sentence `generated_prob`. RAID 0.972 at 5 % FPR, 0.891 under paraphrase. Second tier in Jabarian & Imas. | Pin `modelVersion`. `completely_generated_prob` is deprecated. A mis-sent key silently drops to free-plan rate limits. |
| 3 | **Sapling** | Only vendor here with a per-character price in its API docs ($0.005 per 1,000 characters). Free-trial keys get 50,000 characters per 24 h. Model-based sentence scores with offsets, `ai_fraction`, and an explicit `version` pin. | Default model changed 2026-11-03. No RAID entry or independent evaluation found. |
| 4 | **Originality.ai** (`model-5.0`) | Second tier in Jabarian & Imas (best AUROC among the runners-up). Sentence `ai_score` with offsets. | Enterprise plan required. **Auto credit top-up is on by default.** The documented v3 response has a label and an `ai_assisted_percent` but no document probability. |
| 5 | **Winston AI** (`5.2`) | Top RAID entry (vendor submission 2026-09-24, 1.000 / 0.999). Flags zero-width and homoglyph attacks. | Score is **human**-oriented, 0–100 (invert it). Minimum 300 characters; under 600 is unreliable. RAID test texts are public, so a vendor RAID score may be in-distribution. |
| 6 | Copyleaks | Section-level probabilities, `sensitivity` 1–3, free sandbox mode intended for integration tests. | Login-token flow (login 429 triggers a 5-minute block). Minimum 255 characters. Sandbox output is for wiring tests, not evaluation. |
| 7 | Grammarly AI Detection API (Beta) | RAID 0.995. | Organisation OAuth credentials. Three-step file upload. Document-level only. |
| 8 | Hive | Chunk-level scores over 2,048-character windows. | Moderation-platform project setup. Response envelope path needs a live check. |
| 9 | ZeroGPT (zerogpt.com) | Has a historical RAID harness wrapper. | **UNVERIFIED**: official docs not reachable. Look-alike zerogpt.net and zerogpt.org services have different APIs. |
| – | Turnitin, QuillBot, Scribbr | — | No self-serve public API found. Not wrappable. QuillBot has a RAID entry (0.996). |

### 1.3 How the check round should use them

1. **Normalise every output to P(AI) in [0, 1]** using the `score_semantics` field. Winston and ZeroGPT return human scores, and the OpenAI RoBERTa, RADAR and ADAL detectors put AI at index 0. Store the raw value too.
2. **Pin and log versions** for each call: GPTZero `modelVersion`, Sapling `version`, Pangram `model` plus the returned `version`, Originality `aiModelVersion`, Winston `version`. Sapling changed its default on 2026-11-03, and Originality retires its AI-Allowance and multilang models on 2026-12-11. Unpinned before/after comparisons are therefore not interpretable.
3. **Score documents, not sentences.** Respect the minimum lengths (Copyleaks 255 characters, Winston 300, Sapling recommends ≥ 300, Grammarly ≥ 30 words). Treat sentence scores as diagnostics. In the PISA 2025 study, an embedding detector that was robust on whole texts fell to 0.59 AI recall on single sentences.
4. **Calibrate locally.** For each open detector, fix the threshold at a low FPR (1 % or 5 %) on a held-out human reference set in the same genre and register as the user's text. This is mandatory for Binoculars and Fast-DetectGPT with non-default models. Report rank or percentile against that human set as well as the raw probability.
5. **Report humanizer-specific heads separately**: Pangram `humanizer_score`, GPTZero `subclass.ai.class_probabilities.ai_paraphrased`, tropa-mini `p[humanized]`. Moving text from the `ai` class into a `humanized` or `paraphrased` class is not a move toward human text.
6. **Unicode hygiene.** Apply NFKC and strip zero-width characters before scoring, and fail the check if the rewrite contains homoglyphs or invisible characters. These are attack artefacts, Winston flags them, and they break small classifiers (e5-small homoglyph TPR 0.111) without making text more human-like.
7. **Privacy and cost defaults**: Originality `storeScan: false` and auto top-up disabled; Sapling `zero_retention: true`; Pangram `public_dashboard_link: false`; Copyleaks `sandbox: true` only for integration tests. Commercial calls must be opt-in.

---

## 2. Commercial APIs (official docs read 2026-11-13)

### GPTZero
- **Endpoint**: `POST https://api.gptzero.me/v2/predict/text`. Header `x-api-key: $GPTZERO_API_KEY`.
- **Request**: `{"document": TEXT, "modelVersion": optional}`. List model versions with `GET /v2/model-versions/ai-scan`. Usage stats are at `GET /v3/usage-stats`.
- **Response** (`documents[0]`):
  - `class_probabilities` {human, ai, mixed}, `predicted_class`
  - `confidence_category` (high / medium / low / reject; "high" targets < 1 % error)
  - `document_classification` (HUMAN_ONLY / MIXED / AI_ONLY)
  - `subclass.ai.class_probabilities` {pure_ai, ai_paraphrased}; mixed subclasses {concatenated, polished}
  - `sentences[*].generated_prob`, `perplexity`, `highlight_sentence_for_ai`
  - `paragraphs[*].completely_generated_prob`
  - Deprecated: `completely_generated_prob`, `average_generated_prob`
  - Example response version: `2026-10-07-base` / `4.1o`
- **Limits**: 150,000 characters on API plans (400 if exceeded). Language is auto-detected and non-English text uses a multilingual model.
- **Other**: an AI-patterns stream endpoint (`/v3/ai/patterns/stream`) exists but is allow-listed.
- **Pricing / free tier**: not stated in the API docs (UNVERIFIED).
- **Docs**: https://docs.gptzero.me/api-reference/POST/v2/predict/text.md

### Pangram Labs
- **Endpoint**: `POST https://text.external-api.pangram.com/task`. Header `x-api-key`.
- **Request**: `{"text", "model": "pangram-4" | "default", "public_dashboard_link": false}`. Returns `{task_id}`.
- **Polling**: `GET /task/{task_id}` until `stage` is `STAGE_SUCCESS` or `STAGE_FAILED`.
- **Response**:
  - `version`, `headline`, `prediction_short`
  - `fraction_ai`, `fraction_ai_assisted`, `fraction_human`
  - `windows[*]` {label, ai_assistance_score, confidence, start_index, end_index, word_count, is_humanized, humanizer_score}
- **Errors**: 401, 402 (credits), 403 (model not enabled), 422, 429 (QPS; respect `Retry-After`), 503.
- **Billing**: one started 100-word block per item for Pangram 4.
- **Bulk API**: `POST /bulk`, up to 1,000 billable units per request.
- **Python SDK**: `pangram`.
- **Docs**: https://docs.pangram.com/api-reference/ai-detection

### Originality.ai
- **Endpoint**: `POST https://api.originality.ai/api/v3/scan`. Header `X-OAI-API-KEY`.
- **Request**: `title`, `content`, `check_ai`, `check_plagiarism`, `check_facts`, `check_readability`, `check_grammar`, `storeScan`, `aiModelVersion` (`model-5.0` or `model-5.0-max`).
- **Response**: `results.ai.model`, `results.ai.prediction_label`, `results.ai.ai_assisted_percent`, and `results.ai.sentences[*]` {start_index, end_index, text, ai_score}.
- **Limits**: 500 requests per minute.
- **Account**: Enterprise plan required; auto credit top-up on by default.
- **Legacy**: the old `/api/v1/scan/ai` endpoint (used by the RAID harness) returned `score.ai`. Do not build on it.
- **Docs**: https://docs.originality.ai/scan.md

### Copyleaks
- **Auth**: `POST https://id.copyleaks.com/v3/account/login/api` with {email, key}, which returns `access_token`.
- **Endpoint**: `POST https://api.copyleaks.com/v2/writer-detector/{scanId}/check` with `Authorization: Bearer <token>`.
- **Request**: `text` (255–100,000 characters), `sandbox`, `language`, `explain`, `sensitivity` 1–3. `scanId` is 3–36 characters.
- **Response**: `summary.ai`, `summary.human`, `results[*]` {classification 1 = human / 2 = AI, probability, matches[...chars.starts / lengths]}, `modelVersion` (e.g. v11.0), `scannedDocument.actualCredits`.
- **Billing**: 1 credit per 250 words.
- **Docs**: https://docs.copyleaks.com/reference/actions/writer-detector/check

### Sapling
- **Endpoint**: `POST https://api.sapling.ai/api/v1/aidetect`. Key as JSON `key` or `Authorization: Bearer`.
- **Request**: `text` (≤ 200,000 characters), `sent_scores`, `score_string`, `lang`, `version`, `zero_retention`, `create_report`.
- **Response**: `score` (0–1, threshold 0.5), `ai_fraction`, `verdict`, `segments`, `sentence_scores[*]` {score, sentence, start, end}, `token_probs`, `low_confidence`, `low_confidence_reasons`.
- **Model versions**: 20251027, 20260820, 20261102, 20261104 (current default), plus `ml…` multilingual variants.
- **Quota**: free trial 50,000 characters per 24 h; subscribed keys 10 M per 24 h by default.
- **Pricing**: from $0.005 per 1,000 characters.
- **Docs**: https://sapling.ai/docs/api/detector/

### Winston AI
- **Endpoint**: `POST https://api.gowinston.ai/v2/ai-content-detection`. `Authorization: Bearer`.
- **Request**: `text` (300–150,000 characters; under 600 unreliable), `file`, `website`, `version` (5.2 default … 3.0, or `latest`), `sentences`, `language`.
- **Response**: `score`, a **human** score 0–100; `sentences[*]` {text, score}; `attack_detected` {zero_width_space, homoglyph_attack}; `readability_score`.
- **Docs**: https://docs.gowinston.ai/api-reference/v2/ai-content-detection/post

### Hive
- **Endpoint**: `POST https://api.thehive.ai/api/v2/task/sync` (the sync-task page also shows `api.hivemoderation.com`). Header `Authorization: token <key>`. Body `{"text_data": TEXT}`.
- **Response**: `aggregate_score[{class: "ai_generated", score}]` and `output[*]` per 2,048-character chunk.
- **Unconfirmed**: the outer envelope and pricing were not confirmed. Search the response recursively for `aggregate_score`.
- **Docs**: https://docs.thehive.ai/reference/ai-generated-text-detection-1

### Grammarly AI Detection API (Beta)
- **Step 1**: `POST https://api.grammarly.com/ecosystem/api/v1/ai-detection` with {filename}. Returns `score_request_id` and `file_upload_url`.
- **Step 2**: PUT the file to that URL within 120 s.
- **Step 3**: `GET …/ai-detection/{score_request_id}` until `status` is `COMPLETED`.
- **Score**: `score.average_confidence`, `score.ai_generated_percentage`.
- **Auth**: OAuth 2.0 bearer with scopes `ai-detection-api:read` and `ai-detection-api:write`.
- **Limits**: 30 words minimum, 100,000 characters maximum, 4 MB files, 10 POST/s and 50 GET/s.
- **Docs**: https://developer.grammarly.com/ai-detection-api.html

### ZeroGPT (UNVERIFIED)
- **Endpoint**: `POST https://api.zerogpt.com/api/detect/detectText`, header `ApiKey`, body `{"input_text"}`, read `data.isHuman` (0–100, human-oriented).
- **Source**: the RAID repository's wrapper only. Official docs were not reachable.

### Turnitin, QuillBot, Scribbr
No self-serve public API was found. Turnitin's AI writing indicator is delivered inside institutional licences. A "Turnitin Core API" appears in third-party API directories, but its access terms are UNVERIFIED.

---

## 3. Open-source classifiers (HF IDs verified via the HF API)

Label maps were read from config.json and the model card. "Smoke" means the label direction was confirmed empirically on CPU.

| Model ID | Base / params | Licence | Max tokens | Label mapping (AI =) | Loader | RAID TPR@5 % (all / paraphrase / homoglyph) | Notes |
|---|---|---|---|---|---|---|---|
| tabularisai/ai-text-detection | ModernBERT-base, 149.6 M | MIT | 512 (model 8,192) | softmax[1] (smoke) | AutoModelForSequenceClassification, transformers ≥ 4.48 | 0.990 / 0.981 / 0.979 | ~11 M training rows incl. RAID train; card recommends NFKC. |
| desklib/ai-text-detector-v1.01 | DeBERTa-v3-large, 434 M | MIT | 768 | sigmoid(single logit) (smoke) | custom `DesklibAIDetectionModel` (mean pool + Linear(h, 1)) | 0.912 / 0.837 / 0.997 | Most downloaded modern detector. |
| fakespot-ai/roberta-base-ai-text-detection-v1 | RoBERTa-base, 124.6 M | Apache-2.0 | 512 | id2label {0 Human, 1 AI} (smoke) | standard | none | Training data not stated in card. |
| wasitaigeneratedcom/tropa-mini-ai-text-detector | desklib DeBERTa-v3-large, 434 M | Apache-2.0 | 768 | 4-class [human, ai, ai_edited, humanized]; P(AI) = 1 − p0 (smoke) | custom `AIDetectionModel` (detector_num_labels = 4) | none | Vendor benchmark only. |
| Oxidane/tmr-ai-text-detector | RoBERTa-base, 124.6 M | MIT | 512 | {0 human, 1 ai} (smoke) | standard | 0.958 / 0.956 / 0.886 | 50 k RAID samples, hard-negative mining. |
| MayZhou/e5-small-lora-ai-generated-detector | e5-small, 33.4 M | MIT | 512 | LABEL_1 = AI (smoke) | standard (LoRA merged) | 0.857 / 0.855 / **0.111** | Fastest. |
| Shushant/ADAL-detector-large | RoBERTa-large, 355.4 M | Apache-2.0 | 512 | **index 0 = AI** (card) | standard | 0.963 / 0.731 / 1.000 | Weak under paraphrase. |
| GeorgeDrayson/modernbert-ai-detection-raid-mage | ModernBERT-base, 149.6 M | Apache-2.0 | 512 | softmax[1] assumed (UNVERIFIED) | standard | 0.941 / 0.929 / 0.834 | Sibling `…/modernbert-ai-detection` is MAGE-only. |
| rasbt/ai-text-detector-modernbert | ModernBERT-base, 149.6 M | Apache-2.0 | 512 | {0 HUMAN, 1 AI} | standard | none | Trained on 50 k rows, single source. |
| ShantanuT01/vanguard-ai-text-detector | ModernBERT-large, 395.8 M | MIT | 512 | sigmoid(single logit) | standard (num_labels = 1) | none | Trained on DACTYL 2.0, LLMTrace, MAGA-Bench. |
| ShantanuT01/BERT-tiny-RAID | BERT-tiny, 4.4 M | MIT | 512 | sigmoid assumed (UNVERIFIED) | standard | 0.842 / 0.748 / 0.275 | — |
| openai-community/roberta-base-openai-detector | RoBERTa-base, 125.2 M | MIT | 512 | **index 0 = Fake** | standard | 0.518 / 0.680 / 0.101 | GPT-2 era; failed smoke test. |
| Hello-SimpleAI/chatgpt-detector-roberta | RoBERTa-base | none declared | 512 | {0 Human, 1 ChatGPT} | standard | 0.266 / 0.458 / 0.000 | HC3, 2022. |
| TrustSafeAI/RADAR-Vicuna-7B | RoBERTa classifier ("7B" names the generator) | non-commercial (Vicuna) | 512 | **index 0 = AI** (IBM/RADAR README, RAID harness) | standard | 0.639 / 0.624 / 0.448 | — |
| SuperAnnotate/ai-detector | RoBERTa-large, 354.3 M | other | 512 | sigmoid(single) | pip `generated_text_detector@v1.1.0` | 0.649 / 0.472 / 0.703 | — |
| yaful/MAGE | Longformer | Apache-2.0 | 4,096 | UNVERIFIED (dataset: 0 = machine, 1 = human) | standard | none | — |
| Lindarixon/Linda-RAID, nakas/gpttrace-*, anon-review-meld-2026/meld | DeBERTa-v3-large / Ettin encoders | CC-BY-NC / CC-BY-SA / MIT (gated) | ? | UNVERIFIED | UNVERIFIED (no architectures in config) | 0.995 / 0.998–0.999 / 0.998 | High RAID scores but loaders undocumented; RAID-tuned. |

## 4. Zero-shot detectors

| Tool | Repo (licence) | Default models | CPU option | Score direction | Notes |
|---|---|---|---|---|---|
| Binoculars | ahans30/Binoculars (BSD-3-Clause; last push 2024-05-14) | falcon-7b (observer) + falcon-7b-instruct (performer); max 512 tokens | Qwen2.5-0.5B or 1.5B base + instruct (same tokenizer) | **lower = AI**. Thresholds 0.9015 (F1) and 0.8537 (low FPR) are valid only for the Falcon pair. | Paper: > 90 % of ChatGPT text detected at 0.01 % FPR. RAID no-attack 0.790, synonym 0.421. |
| Fast-DetectGPT | baoguangsheng/fast-detect-gpt (MIT) | `local_infer.py` default: gpt-neo-2.7B for both roles; README (2026-01-31) recommends Llama3-8B / -Instruct | Qwen2.5-1.5B pair or gpt2-xl | higher = AI. The built-in probability mapping is fit for the default models. | ~340× faster than DetectGPT (abstract). |
| Glimpse | baoguangsheng/glimpse (MIT) | API LLM top-k log-probs | remote | higher = AI | Could reuse the user's OPENAI credential if top-k logprobs are still returned. |
| DivEye | IBM/diveye (NOASSERTION) | surprisal-diversity features + classifier | small LM | P(AI) | RAID 0.806. Conceptually closest to the humanizer's own "burstiness" signals. |
| Lastde / Lastde++ | TrustMedia-zju/Lastde_Detector (no licence) | any LM | small LM | see code | No licence file. |
| AdaDetectGPT | Mamba413/AdaDetectGPT (no licence) | LM + learned witness function | small LM | higher = AI | NeurIPS 2025. |
| ImBD | Jiaqi-Chen-00/ImBD (Apache-2.0) | style-aligned scoring LM | UNVERIFIED | higher = AI | Targets machine-*revised* text, which is the humanizer's output type. |
| DetectGPT | eric-mitchell/detect-gpt (MIT) | LM + T5 perturbations | impractical | higher = AI | Superseded. |
| ZipPy | thinkst/zippy (MIT) | compression ratios, no LM | yes | lower delta = AI | Weak on modern text. |

---

## 5. Independent accuracy evidence and failure conditions

**RAID leaderboard** (63 submissions parsed). TPR is measured at a per-domain 5 % FPR over 8 domains, 11 generators and 11 attacks. The generators are GPT-2/3/4, ChatGPT, Cohere, Llama-chat, Mistral and MPT, all 2023-era.
- **Top**: Winston AI 1.000; GPTTrace Ettin-1B 0.999; XYZ Science 0.999; Zhuque 0.998; MELD 0.998; Grammarly 0.995; QuillBot 0.996; GPTZero 0.972.
- **Ceiling caveat**: the top of the board is saturated and dominated by RAID-tuned or vendor self-submitted systems. RAID test texts are public, so a high score does not establish robustness to 2025–2026 frontier models or to commercial humanizers.
- **Missing vendors**: Pangram, Originality.ai, Sapling and Copyleaks have no submission. The Binoculars and FastDetectGPT entries lack aggregate scores in `results.json`.

**Liang et al. 2023 (Patterns)**
- Seven detectors gave an average **61.22 % FPR on 91 TOEFL essays** but were near-perfect on US 8th-grade essays.
- 89/91 essays were flagged by at least one detector. Flagged essays had significantly lower perplexity.
- Rewriting the essays with richer vocabulary cut FPR to 11.77 %.
- A "literary language" self-edit dropped detection of ChatGPT essays from 100 % to 13 %.
- Implication: perplexity-driven detectors penalise constrained, low-variance writing and are moved by lexical diversification alone.

**Jabarian & Imas 2025 (NBER w34223, not peer-reviewed)**
- Setup: 1,992 passages across six genres, four frontier LLMs (GPT-4.1, Claude Opus 4, Claude Sonnet 4, Gemini 2.0 Flash), plus StealthGPT humanization.
- Pangram was the only detector at policy-grade FPR/FNR, and its FNR stayed low on humanized text.
- GPTZero (lower FPR) and Originality.ai (higher AUROC) form a second tier. Neither suits very short text.
- RoBERTa-base was weakest.

**Karr et al. 2026 (arXiv:2608.11256)**
- Published abstracts tested with commercial detectors: light "refine abstract only" AI edits were flagged at **38–80 %**.
- Unmodified 2023–2025 originals were flagged at 9–15 %, with non-STEM rates higher. Flags tracked long-token and Academic Word List density.
- After Undetectable AI humanization, **< 4 %** of rewrites were still flagged.
- Detector scores therefore conflate editing with authorship and are easily evaded.

**Russell et al. 2025 (ACL)**
- Five expert annotators (frequent LLM users), by majority vote, misclassified 1 of 300 articles, beating most commercial and open detectors even under paraphrasing and humanization.
- The experts relied on "AI vocabulary" and also on formality, originality and clarity.
- Implication: lexical and stylistic signals belong in the check round alongside detector scores.

**Schaller et al. 2026 (arXiv:2611.12487, PISA 2025 student essays)**
- Detectors were near-perfect on whole texts.
- A student-style rewriting prompt cut the feature-based detector's AI recall from 0.92 to 0.60 (gpt-4o-mini rewrite) and 0.41 (Qwen3).
- An embedding-based detector stayed ≥ 0.94 on whole texts but fell to 0.59 on single sentences.

**Pangram technical reports and DAMAGE.** These are vendor reports, not independent.
- Pangram 4 claims AUROC 0.9916, FPR 0.0041 %, FNR 0.3396 %, with a dedicated humanizer head.
- DAMAGE studies 19 humanizers and paraphrasers and reports that many detectors fail on humanized text.

**General failure conditions**
- Short text (under ~300 characters or 50 words).
- Single sentences.
- Non-native or formulaic human prose, which produces false positives.
- Light AI editing, which is flagged as AI.
- Paraphrase, synonym and humanizer rewriting.
- Homoglyph and zero-width tricks, which break small classifiers.
- Domain or genre shift away from the training data (RAID-trained models on non-RAID genres).
- Unpinned model updates.
- Mixed human/AI documents, where document-level scores dilute. Use Pangram `fraction_*`, Sapling `ai_fraction` or GPTZero `mixed`.

---

## 6. Bibliography (verification status)

Verified via arXiv export API and/or Crossref on 2026-11-13:

1. Liang, W. et al. (2023). GPT detectors are biased against non-native English writers. *Patterns*. doi:10.1016/j.patter.2023.100779; arXiv:2304.02819.
2. Dugan, L. et al. (2024). RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. *ACL 2024*. doi:10.18653/v1/2024.acl-long.674; arXiv:2405.07940.
3. Hans, A. et al. (2024). Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text. arXiv:2401.12070 (ICML 2024 per repository).
4. Bao, G. et al. (2024). Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature. arXiv:2310.05130 (ICLR 2024).
5. Mitchell, E. et al. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. arXiv:2301.11305 (ICML 2023).
6. Krishna, K. et al. (2023). Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. *NeurIPS 36*. doi:10.52202/075280-1195; arXiv:2303.13408.
7. Sadasivan, V. S. et al. Can AI-Generated Text be Reliably Detected? arXiv:2303.11156 (TMLR).
8. Hu, X. et al. (2023). RADAR: Robust AI-Text Detection via Adversarial Learning. *NeurIPS 36*. doi:10.52202/075280-0662; arXiv:2307.03838.
9. Emi, B. & Spero, M. (2024). Technical Report on the Pangram AI-Generated Text Classifier. arXiv:2402.14873.
10. Glickenhaus, B. et al. (2026). Pangram 4 Technical Report. arXiv:2607.27183.
11. Masrour, E., Emi, B. & Spero, M. (2025). DAMAGE: Detecting Adversarially Modified AI Generated Text. arXiv:2501.03437.
12. Thai, K. et al. (2025). EditLens: Quantifying the Extent of AI Editing in Text. arXiv:2510.03154.
13. Russell, J., Karpinska, M. & Iyyer, M. (2025). People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text. *ACL 2025*. doi:10.18653/v1/2025.acl-long.267; arXiv:2501.15654.
14. Li, Y. et al. (2024). MAGE: Machine-generated Text Detection in the Wild. *ACL 2024*. doi:10.18653/v1/2024.acl-long.3; arXiv:2305.13242.
15. Wang, Y. et al. (2024). M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection. *EACL 2024*. doi:10.18653/v1/2024.eacl-long.83; arXiv:2305.14902.
16. Wu, J. et al. A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions. arXiv:2310.14724.
17. Wang, Y. et al. (2024). Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks. *ACL 2024*. doi:10.18653/v1/2024.acl-long.160; arXiv:2402.11638.
18. Fraser, K. C. et al. (2025). Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods. *JAIR*. doi:10.1613/jair.1.16665.
19. Xu, Y. et al. (2024). Training-free LLM-generated Text Detection by Mining Token Probability Sequences (Lastde). arXiv:2410.06072.
20. Chen, J. et al. (2025). Imitate Before Detect: Aligning Machine Stylistic Preference for Machine-Revised Text Detection. *AAAI 2025*. doi:10.1609/aaai.v39i22.34525; arXiv:2412.10432.
21. Bao, G. et al. Glimpse: Enabling White-Box Methods to Use Proprietary Models for Zero-Shot LLM-Generated Text Detection. arXiv:2412.11506 (ICLR 2025 per repository).
22. Zhou, H. et al. (2025). AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees. *NeurIPS 38*. doi:10.52202/085713-2980; arXiv:2510.01268.
23. Basani, A. R. et al. Diversity Boosts AI-Generated Text Detection (DivEye). arXiv:2509.18880 (TMLR 2026).
24. Wu, J. et al. (2024). DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios. *NeurIPS 37*. doi:10.52202/079017-3186.
25. Verma, V. et al. (2024). Ghostbuster: Detecting Text Ghostwritten by Large Language Models. *NAACL 2024*. doi:10.18653/v1/2024.naacl-long.95; arXiv:2305.15047.
26. Dugan, L. et al. (2025). GenAI Content Detection Task 3: Cross-Domain Machine-Generated Text Detection Challenge. arXiv:2501.08913 (COLING 2025 workshop).
27. Weber-Wulff, D. et al. (2023). Testing of detection tools for AI-generated text. *International Journal for Educational Integrity*. doi:10.1007/s40979-023-00146-z; arXiv:2306.15666.
28. Perkins, M. et al. (2024). GenAI Detection Tools, Adversarial Techniques and Implications for Inclusivity in Higher Education. arXiv:2403.19148.
29. Perkins, M. et al. (2024). Simple techniques to bypass GenAI text detectors: implications for inclusive education. *International Journal of Educational Technology in Higher Education*. doi:10.1186/s41239-024-00487-w.
30. Tufts, B. et al. (2024). A Practical Examination of AI-Generated Text Detectors for Large Language Models. arXiv:2412.05139.
31. Karr, J. A. et al. (2026). Why AI Detection Fails for Academic Integrity. arXiv:2608.11256.
32. Schaller, N.-J. et al. (2026). Detecting AI-Authored Text in Authentic Student Writing: Evidence from PISA 2025. arXiv:2611.12487.
33. Jabarian, B. & Imas, A. (2025). Artificial Writing and Automated Detection. NBER Working Paper 34223. doi:10.3386/w34223.

UNVERIFIED or not looked up: the original GLTR paper (Gehrmann et al., 2019) and its repository and licence; vendor benchmark claims on the tropa-mini model card (including its citation of a Jabarian & Imas dataset subset); all third-party review-site pricing figures.
