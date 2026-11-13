# L1 surface patterns

Catalogue of the surface families counted by `scripts/surface_scan.py`, with the edit that fixes each. Families follow Wikipedia's "Signs of AI writing" (WikiProject AI Cleanup) and the vocabulary studies listed in `assets/ai_vocabulary.json`. Weight 3 = act on one sighting; 2 = strong in company; 1 = weak alone (a careful human makes it on purpose).

Surface fixes are necessary and cheap but not sufficient: on RAID, synonym substitution moved Binoculars by 36 points and RADAR by 3, and narrative/structural detectors stayed above 90 percent after span-level cleanup (see `research_digest.md`). Do this pass last, after the structural passes, because LLM-written structural edits reintroduce these patterns.

## Staging instead of stating (weight 3)

| Family (`surface_scan`) | Looks like | Fix |
|---|---|---|
| `not_x_but_y` | "It's not just X, it's Y"; "not merely X but Y"; "This does not mean X. It means Y." | State Y. Keep a contrast only if someone actually holds X. |
| `closer_line` | One-line paragraph that restates: "That's the real win." "Let that sink in." | Delete, or replace with a new fact or consequence. |
| `staged_opener` | "Let's dive in", "Here's the thing", "Honestly?" | Start with the point. |
| `deep_saying` | "At its core", "the real question is", "X is the language of Y" | Replace with the specific claim. |
| `arguing_with_no_one` (2) | "To be clear,", "This is not to say", "One might be tempted to" | Remove the rebuttal of an objection nobody made. |

## Leftovers (weight 3)

| Family | Looks like | Fix |
|---|---|---|
| `chatbot_residue` | "I hope this helps", "Great question", "Let me know if", "Here is a revised..." | Delete the wrapper, keep the content. |
| `knowledge_limit` | "As of my last update", "not widely documented", "based on available information" | State what the source shows, or cut. |

## Inflation (weight 2)

| Family | Looks like | Fix |
|---|---|---|
| `inflated_significance` | "stands as a testament", "plays a pivotal role", "marking a pivotal moment", "evolving landscape", "continues to thrive" | Keep the fact, drop the significance claim. End on the last concrete fact. |
| `ing_rider` | ", highlighting its importance", ", ensuring that...", ", reflecting a broader..." | Finite clause with a subject, a separate sentence, or delete. Same edit as Pass B participial clauses. |
| `vague_authority` | "Experts argue", "studies suggest" without a source | Name the source the user gave, or cut. |
| `sales_language` | "nestled", "breathtaking", "seamless", "cutting-edge", "unlock the potential" | Say what it is. |
| `copula_avoidance` (1) | "serves as", "stands as", "boasts", "features" | Use is / has. |
| `signpost_transition` (1) | Sentence-initial "Additionally,", "Moreover,", "Furthermore,", "In conclusion," | Delete, or use the real relation (so, but, because). |
| `hedge_stack` (1) | "could potentially", "it is important to note that" | One hedge where the evidence is weak; none elsewhere. |

## Vocabulary

`assets/ai_vocabulary.json` holds 669 items (625 words, 11 phrases, 24 stopword-removed trigrams, 5 POS templates, 2 constructions), each matched to its source: Kobak et al. 2025 excess words (PubMed), Liang et al. ICML 2024 adjective/adverb lists (peer reviews), Juzek and Ward 2025 focal words, Reinhart et al. 2025 PNAS over-represented words by model, Antislop 2025 words and trigrams across 67 models, Gray 2024, Matsui 2025, LAMP 2025 and others. `surface_scan.py` counts words and phrases (lemma plus regular inflections) and the trigram cliches after stopword removal, and drops the "common excess" function-like words unless several sources list them.

Rules:
- Words are register-dependent. "Camaraderie", "palpable", "unease" are normal in fiction and conspicuous in a news report. Judge rate against the register's human band (the check report gives it), not presence.
- The usual right edit is deletion: the word was decoration ("a meticulous analysis" -> "an analysis", or say what made it careful).
- Do not swap one listed word for another listed synonym. LLMs spread probability over synonyms; a human repeats the plain term.
- Fiction cliche trigrams (Antislop): "voice barely above a whisper", "took a deep breath", "air thick with the scent of", "heart pounding in chest", "casting long shadows". Rewrite the moment instead of replacing the words.

## Punctuation and format

| Family | Looks like | Fix |
|---|---|---|
| `em_dash` rate | Dashes as the universal connector | Choose the relation: comma, colon, parentheses, full stop. If the author's samples use dashes, keep their rate. |
| `triad` rate | "X, Y, and Z" in every other sentence | Keep three items only when there are three. |
| `bold_label_list` | "- **Performance:** Performance improved..." | Prose, or a plain list. |
| `title_case_heading`, `emoji_or_arrow` | "## Strategic Partnerships And Growth", "🚀 Launch" | Sentence case, no decoration. |
| `one_line_closers` (count) | Short one-sentence paragraph after a long one | Merge or delete unless it adds a fact. |
| `repeated_sentence_openers` | "She noted... She saw... She filed..." | Merge or vary, unless the repetition is deliberate. |
| curly quotes | Curly quotes where the target uses straight ones | Match the target format (weak alone). |

## Integrity flags (not humanization)

`surface_scan.py` reports `integrity_warning` when the text contains zero-width characters, non-breaking spaces or Cyrillic look-alike letters. These are detector-gaming artefacts (RAID homoglyph and whitespace attacks). Normalise the text (NFKC) and remove them; never add them.

## When not to act

Leave a pattern alone inside quotations, titles, names, and passages that discuss the pattern. Letter salutations predate chatbots. Text written before late 2022 is not LLM output. Human writing increasingly absorbs these habits, so several tells together are the signal, not one.
