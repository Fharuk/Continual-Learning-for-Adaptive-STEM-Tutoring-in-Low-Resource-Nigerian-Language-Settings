"""Generate explicitly synthetic student-response sequences for pipeline tests.

This simulator is not a model of real Nigerian learners and must not be used to
claim educational efficacy, language quality, or real-world adaptation.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path

TOPICS = ["motion", "newtons_laws", "energy", "waves", "electricity", "optics"]
DIFFICULTIES = ("easy", "medium", "hard")


def simulate(n_students: int = 500, questions_per_topic: int = 15, seed: int = 2027):
    if n_students < 1 or questions_per_topic < 1:
        raise ValueError("n_students and questions_per_topic must be positive")
    rng = random.Random(seed)
    rows = []
    for s in range(n_students):
        # Independent topic mastery, deliberately simple and fully specified.
        mastery = {t: rng.betavariate(2.0, 2.0) for t in TOPICS}
        error = {t: rng.uniform(0.02, 0.18) for t in TOPICS}
        learning_rate = rng.uniform(0.005, 0.04)
        step = 0
        for ti, topic in enumerate(TOPICS):
            for qi in range(questions_per_topic):
                difficulty = DIFFICULTIES[qi % 3]
                penalty = {"easy": 0.0, "medium": 0.15, "hard": 0.30}[difficulty]
                # Guessing floor and slip/error parameter are explicit simulator assumptions.
                p = max(0.05, min(0.95, 0.20 + 0.75 * mastery[topic] - penalty - error[topic]))
                before = mastery[topic]
                correct = int(rng.random() < p)
                rows.append({
                    "student_id": f"S{s:04d}", "sequence_index": step,
                    "task_index": ti, "topic": topic,
                    "question_id": f"{topic.upper()}_{qi+1:03d}",
                    "difficulty": difficulty, "language": "en",
                    "correct": correct, "simulated_p_correct": round(p, 6),
                    "simulated_mastery_before": round(before, 6),
                    "simulated_learning_rate": round(learning_rate, 6),
                    "data_status": "synthetic_not_real_student_data",
                })
                step += 1
                # Feedback-induced mastery movement; bounded to [0,1].
                mastery[topic] = min(0.99, max(0.01, mastery[topic] + learning_rate * (correct - p)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=500)
    parser.add_argument("--questions-per-topic", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/interactions.csv"))
    args = parser.parse_args()
    rows = simulate(args.students, args.questions_per_topic, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    meta = {"seed": args.seed, "students": args.students, "topics": TOPICS,
            "questions_per_topic": args.questions_per_topic, "rows": len(rows),
            "status": "synthetic_not_real_student_data"}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows):,} explicitly synthetic interactions to {args.out}")

if __name__ == "__main__":
    main()
