import json
from pathlib import Path

from src.retrieval import Retriever, MODES

TOP_K = 5
HERE = Path(__file__).resolve().parent


def first_correct_rank(results, question):
    """Position (1, 2, 3...) of the first result from a correct page, or None."""
    for rank, r in enumerate(results, start=1):
        if r["doc"] == question["doc"] and r["page"] in question["pages"]:
            return rank
    return None


def main():
    questions = json.loads((HERE / "questions.json").read_text())
    retriever = Retriever()
    print(f"\n{len(questions)} questions, top {TOP_K} results each\n")

    table = []
    misses = {}
    for mode in MODES:
        hits = 0
        reciprocal = 0.0
        misses[mode] = []
        for q in questions:
            results = retriever.search(q["question"], mode=mode, top_k=TOP_K)
            rank = first_correct_rank(results, q)
            if rank:
                hits += 1
                reciprocal += 1 / rank
            else:
                got = ", ".join(f"{r['doc']} p.{r['page']}" for r in results[:3]) or "nothing"
                misses[mode].append((q["question"], got))
        table.append((mode, hits / len(questions), reciprocal / len(questions)))

    lines = [f"| Method | Hit Rate@{TOP_K} | MRR |", "|---|---|---|"]
    for mode, hit_rate, mrr in table:
        lines.append(f"| {mode} | {hit_rate:.2f} | {mrr:.2f} |")
    report = "\n".join(lines)
    print(report)

    print("\nQuestions each method missed (and what it returned instead):")
    for mode in MODES:
        print(f"\n[{mode}]")
        if not misses[mode]:
            print("  none")
        for text, got in misses[mode]:
            print(f"  - {text}")
            print(f"      returned: {got}")

    (HERE / "results.md").write_text(report + "\n")
    print("\nSaved table to eval/results.md")


if __name__ == "__main__":
    main()