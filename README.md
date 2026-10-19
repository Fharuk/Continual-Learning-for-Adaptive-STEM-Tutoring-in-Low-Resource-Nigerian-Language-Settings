# Continual Learning for Adaptive STEM Tutoring in Low-Resource Nigerian Language Settings

Research scaffold for an IJCNN 2027 study. **This repository currently contains no real student data, validated translations, trained models, or empirical results.** Synthetic interaction data are for testing the pipeline and studying controlled hypotheses only; they do not establish learning gains or language suitability.

## Research status and scope

- Proposed subject: secondary-school physics, six topic units.
- Proposed languages: English plus Yoruba, Igbo, and Hausa.
- Translation validation: **not currently available**. Treat translated content/model comparisons as exploratory until qualified speakers review the questions, answer choices, explanations, and terminology. Machine translation or back-translation alone is not adequate validation.
- Compute: GPU presumed available, but the data simulator and metric utilities run on Python's standard library.
- No empirical claims should be written until the experiments have run and been checked.

## Quick start

```bash
python -m src.simulate --students 500 --seed 2027 --out data/synthetic/interactions.csv
python -m src.experiment --out experiments/results --seeds 11 22 33
python -m unittest discover -s tests -v
```

Generated records are explicitly marked `synthetic`. Do not use them as evidence about real students or language proficiency. See [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) for the end-to-end workflow and decision gates, [`docs/DATA_PROTOCOL.md`](docs/DATA_PROTOCOL.md) for dataset design, and [`paper/main.tex`](paper/main.tex) for the honest manuscript skeleton.

## Repository map

- `src/simulate.py` — reproducible synthetic student-response stream generator.
- `src/metrics.py` — small, dependency-free BWT/FWT/adaptation metric helpers.
- `data/questions/template.jsonl` — question-bank schema/example (illustrative, not validated).
- `data/synthetic/` — generated outputs; ignored by Git.
- `docs/` — protocol, experiment plan, risks, and submission checklist.
- `paper/` — IEEE conference LaTeX starter; results and references must be verified before submission.

## Reproducibility requirements

Record the source/approval status of every question, split by student (not interaction), lock seeds and software versions, save raw predictions and per-task evaluation matrices, and report mean and variation across independent seeds. Never tune on the final test students. See the protocol before interpreting any metric.
