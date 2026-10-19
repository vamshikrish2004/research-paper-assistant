import re

from langchain_ollama import ChatOllama

from src import config
from src.retrieval import Retriever

NO_ANSWER = "I don't know based on the documents."
MAX_TURNS = 3   # how many earlier turns the assistant remembers
TOP_K = 5       # how many chunks are given to the AI

# Words that suggest a question depends on the earlier chat
FOLLOWUP_WORDS = {
    "it", "its", "that", "this", "they", "them", "their", "those", "these",
    "he", "she", "his", "her", "one", "ones", "former", "latter", "also", "too",
}


def looks_like_followup(question):
    """True if the question probably needs the earlier chat to make sense."""
    words = re.findall(r"[a-z']+", question.lower())
    if not words:
        return False
    if len(words) <= 4:
        return True
    if words[0] in {"and", "also", "but", "so"}:
        return True
    return any(w in FOLLOWUP_WORDS for w in words)


def verify(answer, hits):
    """Check the citations in an answer. Returns (valid cited numbers, problems)."""
    problems = []
    valid = set(range(1, len(hits) + 1))

    found = set()
    for group in re.findall(r"\[([^\]]*)\]", answer):
        found.update(int(n) for n in re.findall(r"S(\d+)", group))
    cited = sorted(found & valid)
    unknown = sorted(found - valid)

    if unknown:
        problems.append(f"cites sources that were not provided: {unknown}")
    if not cited:
        problems.append("no valid citation given")

    # Every number in the answer should appear in the text it cites
    if cited:
        check_text = " ".join(hits[n - 1]["text"] for n in cited)
    else:
        check_text = " ".join(h["text"] for h in hits)
    plain = re.sub(r"\[[^\]]*\]", "", answer)
    for num in re.findall(r"\d[\d,.]*\d", plain):
        if num not in check_text:
            problems.append(f"the number {num} does not appear in the cited text")

    return cited, problems


class Assistant:
    def __init__(self, retriever=None):
        self.retriever = retriever or Retriever()
        self.llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
        self.history = []   # list of (question, answer)

    def rewrite(self, question):
        """Turn a follow-up question into a complete question.
        Questions that already make sense on their own are left unchanged."""
        if not self.history or not looks_like_followup(question):
            return question
        recent = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in self.history[-MAX_TURNS:]
        )
        prompt = f"""Rewrite the last question so it is a complete question that makes sense on its own.
Replace words like "it", "that", "they", "the other one" with what they refer to.
If the last question is already complete, repeat it exactly as it is.
Do not add anything that the last question does not ask about.
Reply with ONLY the question.

Conversation:
{recent}

Last question: {question}
Complete question:"""
        lines = [l.strip() for l in self.llm.invoke(prompt).content.strip().splitlines() if l.strip()]
        return lines[0].strip('"') if lines else question

    def generate(self, question, hits):
        """Ask the AI to answer from labeled sources and cite them."""
        blocks = [
            f"[S{i}] ({h['doc']}, page {h['page']})\n{h['text']}"
            for i, h in enumerate(hits, 1)
        ]
        sources_text = "\n\n".join(blocks)
        prompt = f"""You answer questions using only the sources below.

Rules:
1. Use ONLY the sources. Do not use your own knowledge.
2. After each fact, cite the source label, like [S1]. Use only labels that appear below.
3. Numbers in square brackets inside the source text (like [12]) are reference numbers, not source labels. Ignore them.
4. If the sources do not contain the answer, reply exactly: {NO_ANSWER}
5. Keep the answer short, 1 to 4 sentences.

Sources:
{sources_text}

Question: {question}
Answer:"""
        return self.llm.invoke(prompt).content.strip()

    def ask(self, question, mode="hybrid_rerank", doc=None):
        search_query = self.rewrite(question)
        hits = self.retriever.search(search_query, mode=mode, top_k=TOP_K, doc=doc)
        answer = self.generate(search_query, hits) if hits else NO_ANSWER

        normalized = answer.replace("\u2019", "'").lower()
        refused = "don't know" in normalized or "do not know" in normalized
        cited, problems = ([], []) if refused else verify(answer, hits)

        self.history.append((question, answer))
        return {
            "answer": answer,
            "search_query": search_query,
            "hits": hits,
            "cited": cited,
            "problems": problems,
            "refused": refused,
        }


def main():
    assistant = Assistant()
    print("Ready. Type a question, or 'quit'.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        result = assistant.ask(question)
        if result["search_query"] != question:
            print(f"(searched for: {result['search_query']})")
        print(f"\nAI: {result['answer']}")

        if result["cited"]:
            print("\nSources cited:")
            for n in result["cited"]:
                h = result["hits"][n - 1]
                print(f"  [S{n}] {h['doc']}, page {h['page']}")

        if not result["refused"]:
            if result["problems"]:
                print("\nCitation check: WARNING")
                for p in result["problems"]:
                    print("  -", p)
            else:
                print("\nCitation check: OK")
        print()


if __name__ == "__main__":
    main()