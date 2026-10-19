# Data protocol and item-bank schema

## Status
No approved/validated question corpus is present. The JSONL example is a schema demonstration only. Build the English item bank first. Translations remain candidate translations until qualified fluent-speaker review has been obtained for each target language.

## Proposed physics units
Motion/kinematics; Newton's laws; work and energy; waves and sound; electricity/circuits; geometrical optics. Map each to an explicit secondary curriculum and cite the curriculum source. Target 15 questions per unit, with five each at easy/medium/hard, but do not prioritize quota over correctness or coverage.

## Item requirements
One unambiguous stem; four plausible mutually exclusive options; exactly one verified key; correct units and significant figures; concise explanation; source/author and verification record. Do not use automated generation as final authority. Record whether each item is original, adapted, or derived and resolve licensing/copyright before publication.

Required fields: `item_id`, `topic`, `difficulty`, `language`, `stem`, `options` (four strings), `answer_index` (0–3), `explanation`, `source`, `review_status`, `reviewer_role`, `translation_model` (if applicable), `translation_status`, `version`.

## Synthetic response simulation
`src/simulate.py` generates 500 simulated profiles × 6 topics × 15 responses = 45,000 rows by default. It includes a synthetic response probability and latent mastery **for audit/simulation metadata only**. Those hidden fields must never be supplied to the predictive model. The simple response update is a transparent toy mechanism, not an empirically validated student-learning model. Vary its assumptions and state them in the paper.

Do not present simulation rows as real students, real engagement, measured learning gains, or validation. For a credible simulation study, test recovery/robustness under multiple generator settings and compare against an analytically simple baseline. For an educational efficacy claim, real prospective data and appropriate ethics review/consent would be required.

## Translation review gate
Machine translation can alter physics meaning, distractor distinctions, negation, numbers/units, terminology, dialect and orthography. Use qualified fluent reviewers; ideally two independent reviews followed by adjudication. Keep English source alongside translated text, and ask reviewers to flag “untranslatable/ambiguous” items rather than force a literal rendering. Record reviewer qualifications and process without exposing personal data. BLEU/back-translation alone does not establish correctness. If review is not secured, report only a translation feasibility audit, not language-specific model efficacy.
