# End-to-end execution plan — IJCNN 2027

**Working title:** Continual Learning for Adaptive STEM Tutoring in Low-Resource Nigerian Language Settings  
**Target stated by authors:** IJCNN 2027 regular paper, six IEEE two-column pages; verify the official 2027 call for papers, page limits, template, anonymization, and deadline when published. The planning deadline supplied is 31 January 2027.  
**Current status:** proposal/scaffold only. No language-validated corpus, trained models, or measured results exist in this repository.

## The critical design decision: validate the language before making language claims

There are no fluent-speaker reviewers currently available. Consequently, do **not** claim the system works in Yoruba, Igbo, or Hausa, and do not treat machine translations/back-translations as ground truth. Make the first study an English-language controlled continual-learning experiment with original English STEM items. In parallel, create a clearly labelled *candidate* machine-translated set for feasibility/error auditing only. A multi-language claim is gated on recruiting qualified language reviewers and documenting review. If reviewers cannot be recruited by the project decision date, retitle/narrow the paper to synthetic English-only benchmarking and frame Nigerian languages as motivation/future work—not as evaluated settings.

## A defensible research question and claims

**RQ:** Under a controlled stream of secondary-school physics questions and student-response histories, how do regularization and replay approaches trade off retention of earlier topics and rapid student-specific response prediction, relative to sequential fine-tuning and joint-training references?

Keep two separate claims distinct:
1. **Task-level continual learning:** learning topic units sequentially while evaluating retention on held-out questions/students.
2. **Personalization:** estimating a student's next response from that student's prior interactions. This is not automatically shown by a topic-level benchmark; evaluate it on held-out students and report a cold-start vs. increasing-history curve.

Use cautious terminology: a synthetic generator can test whether algorithms recover behavior under the generator's assumptions, not demonstrate real learner adaptation or classroom benefit. Do not call a simulated corpus a student dataset without the qualifier “synthetic.”

## Work plan and gates (13-week schedule; adapt to actual start date)

| Phase | Timing | Work and concrete deliverables | Gate / stop condition |
|---|---|---|---|
| 0. Protocol | Week 1 | Freeze RQs, task order, splits, metrics, seeds, compute budget, model and baselines in a dated protocol. Inspect IJCNN CFP/template. Create bibliography with verified DOI/arXiv metadata. | Do not train until labels, split, and metric definitions are unambiguous. |
| 1. Literature and scope | Weeks 1–2 | Read foundational continual learning, knowledge tracing, multilingual NLP, and STEM education evidence. Build claim-to-citation map; identify closest work. | Avoid “first” or “no prior work” unless a documented search supports it. |
| 2. Item bank | Weeks 2–3 | Draft 6 units (motion, Newton's laws, energy, waves, electricity, optics); target 15 items/unit with balanced difficulty. Check every answer, unit, distractor, explanation, and syllabus fit. Store source, author, reviewer, revision history. | No item enters an evaluation set without answer verification. Secure qualified physics review where possible. |
| 3. Synthetic generator | Weeks 3–4 | Run the included deterministic simulator; conduct sensitivity checks over mastery, slip, guess, learning-rate, topic order, and response noise. Split by student; ensure no identity/history leakage. | Synthetic results are explicitly benchmark/simulation evidence only. Do not treat 500 × 90 records as 45,000 independent learners. |
| 4. Language feasibility | Weeks 3–5 (parallel) | Machine-translate question, options, explanation, and terminology; retain source and model/version. Seek fluent reviewers for each language. Use bilingual item-level review and adjudication; log issues (meaning, physics, dialect, script/diacritics, units). | If no qualified reviewer: exclude translated-language accuracy/learning claims; label translations unvalidated and exploratory, or defer them. |
| 5. Baselines and implementation | Weeks 5–7 | Implement most simple and reproducible baselines first: majority/prior predictor, sequential fine-tuning, joint training reference, replay. Then EWC. Consider PNN only if a tested, compatible implementation is available; define its task-specific output and parameter budget. | Compare models with matched data, tuning budget, backbone, and compute. Report parameters and memory. |
| 6. Pilot | Week 7 | Tiny end-to-end run, inspect data leakage, calibration, class balance, loss curves, replay sampling, Fisher estimation, checkpointing, and GPU usage. | Fix pipeline before the full multi-seed run; never select a test set based on pilot outcomes. |
| 7. Main experiments | Weeks 8–9 | Run preregistered seeds (at least 3, ideally more if feasible), fixed task order plus order sensitivity, held-out students and questions, all stage × task evaluations, per-language only for validated translations. Save configs, logs, predictions, checkpoints, environment. | Report uncertainty and failures; do not omit unfavorable seeds. |
| 8. Analysis | Week 10 | Compute task-wise accuracy and BWT/FWT with explicit definitions, adaptation curves, per-topic and subgroup error; perform ablations (memory size, EWC strength, history length), robust sensitivity analyses. | Distinguish point estimates from uncertainty; explain any synthetic generator dependence. |
| 9. Writing | Weeks 10–11 | Write methods/data/results first; then related work, discussion, intro, abstract. Add only results reproduced from saved artifacts. Keep limitation/ethics statements specific. | No fabricated, inferred, or illustrative values in results tables. |
| 10. Reproduction and submission | Weeks 12–13 | Fresh-environment run, unit tests, artifact audit, citations/DOIs, language-review documentation, PDF compile, page/format check, anonymization, proofreading, official submission checklist. | Submit only after 2027 CFP confirms venue details and all statements match evidence. |

## Protocol details to lock before training

### Tasks, data and splits
- Define each topic unit as one continual-learning experience; keep the order fixed for the primary result and test at least one alternate order as a sensitivity analysis.
- Maintain separate item authoring and evaluation question pools where possible. If the same question is seen in training and test, memorization can masquerade as transfer.
- Split **students**, not interaction rows, into train/validation/test. For a personalized evaluation, test students must be disjoint from training students. Use prior history for each test student only where the protocol permits it; ensure labels from the future are never input.
- Keep synthetic generator parameters and random seeds in metadata. Evaluate on both the matched generator and altered parameters to reveal simulator overfitting.
- Store data version, provenance, language, translator/model/version, reviewer and adjudication status for each item. Strip identifying information; this planned synthetic data contain no real student records.

### Model and baselines
- Select one encoder only after pilot feasibility. A multilingual encoder's African-language pretraining does not guarantee Yoruba/Igbo/Hausa performance. Record exact Hugging Face model revision, tokenizer, license, and preprocessing.
- Inputs must contain only information available before the response (question and response history). Predict correctness; never feed simulator's hidden mastery, true probability, or future correctness to the learner model.
- Compare majority/prior baseline, sequential fine-tuning (no CL protection), joint training (non-stream upper reference), replay, and EWC. PNN is structurally different and may grow capacity with tasks; report total parameters, compute, and task-specific inference assumptions. Do not describe it as a fair same-capacity comparison without qualification.
- Personalization: explicitly define adapter initialization, update rule, data available, and whether updates occur online. Prevent test labels from leaking into shared model training. Compare shared-only, shared + per-student adapter, and a simple student-history baseline.

### Metrics and analysis
- Accuracy and preferably log loss/AUROC or balanced accuracy if class proportions warrant; report per-task values and macro average. Report confidence intervals across independent seeds and, where applicable, students.
- BWT: for task j, final accuracy on j minus accuracy immediately after first learning j; average across tasks learned before the final stage. Negative means loss. Be explicit about whether this is accuracy difference or a forgetting score.
- FWT: performance on not-yet-trained task before its training relative to a clearly defined untrained/reference baseline. Use the same evaluation students/items and specify baseline; do not compute FWT from a matrix without pre-training evaluations.
- Adaptation speed: define a threshold before evaluation (e.g., interactions until a fixed improvement over cold-start baseline) and compute on per-student held-out streams. Also show the full adaptation curve; report cases that never reach threshold as censored/not reached, not silently dropped.
- Compare seeds with paired summaries where possible. Avoid treating the 45,000 response rows as independent observations; uncertainty must respect student and seed clustering.

## Paper assembly and six-page discipline

Suggested allocation (confirm venue rules): Introduction 0.5 page; related work 0.5; data 0.75; method 1.25; experiments/results 1.75; discussion/conclusion 0.5; abstract, figures, references fit remaining space. Prioritize a clear methods/results table over broad claims. The 20–25 reference target is not a requirement if it crowds out reproducibility or verified citations. Verify each citation against the actual publication; do not cite secondary summaries as if primary papers.

Recommended figure/table set: (1) task/data/model pipeline diagram, (2) accuracy over training stages on earlier topics, (3) compact stage-by-task accuracy matrix, (4) primary metrics table with mean and variation. Every figure must be generated from saved experiment outputs, not hand-entered values.

## Submission checklist

- [ ] Official IJCNN 2027 call, scope, deadline, page limit, format and submission policy verified.
- [ ] Research question and contributions match evidence; no unsupported “first” claim.
- [ ] Question answers and distractors independently checked; language claims supported by documented qualified review.
- [ ] No real learner or synthetic-generated hidden variables leak into model inputs.
- [ ] Student-disjoint splits; task ordering and all baselines documented.
- [ ] Results reproducible from checked-in configs/scripts; seeds, variance and failed runs accounted for.
- [ ] Table values trace to raw predictions and analysis scripts; no placeholder results remain.
- [ ] Ethics/privacy, licenses, translation limitations and synthetic-data caveat included.
- [ ] References, anonymization, PDF rendering and exact page count checked against official instructions.
