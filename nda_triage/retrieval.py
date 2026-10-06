"""Stage 3: retrieval (the "R" in RAG).

Given a clause that was flagged, find the playbook language that should
replace it. The playbook's fallbacks are embedded once into a local Chroma
index; at query time the flagged clause is embedded the same way and the
nearest fallbacks come back.

    index:  fallback  ->  index document  ->  embedding  ->  Chroma (.chroma/)
    query:  clause text  ->  embedding  ->  nearest fallbacks for this posture

Design choices:

- What gets embedded. A query is a problem clause ("neither party will
  solicit any employee..."), but a fallback's own text is often the fix
  ("Delete the non-solicitation provision"). Those don't read alike. So each
  index document describes the problem the fallback solves (rule name,
  description, and the descriptions of the patterns it applies to) followed
  by the fallback text. We embed what the problem looks like and return the
  fix.
- Posture is a hard filter, not a similarity signal. A recipient-only
  fallback must never be offered to a discloser, however similar it looks.
- Embedding model: Chroma's default, all-MiniLM-L6-v2 (384 dimensions, runs
  on CPU via ONNX). It is small and local: no API key, no text leaves the
  machine, and nothing extra to install. It is weaker on legal phrasing than
  large hosted models; the eval harness measures whether that matters.
- Chunking. A fallback is one chunk (they are a paragraph or less). Reference
  NDAs added to corpus/ are chunked by clause using the Stage 2 splitter,
  so a chunk is always a whole section, never half of one.
- Exact lookup vs. search. When classification has already named the rule,
  pattern, and posture, `lookup_fallbacks` returns the fallback directly from
  the rubric with no embeddings involved. Semantic search earns its place
  when the classifier is unsure, and for the reference corpus. The evals
  compare the two.
"""

import argparse
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from nda_triage.ingest import SUPPORTED, parse_nda
from nda_triage.models import RetrievedText
from nda_triage.rubric import Fallback, Posture, Rubric, Rule, load_rubric

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / ".chroma"
CORPUS_DIR = ROOT / "corpus"


def index_document(rule: Rule, fallback: Fallback) -> str:
    """The text we embed for one fallback: the problem it fixes, then the fix."""
    problems = [p.description for p in rule.patterns if p.id in fallback.applies_to]
    return "\n".join([
        f"{rule.name}: {rule.description}",
        "Problem: " + " ".join(problems),
        "Fallback: " + fallback.text,
    ])


def lookup_fallbacks(rubric: Rubric, rule_id: str, pattern_id: str, posture: Posture) -> list[Fallback]:
    """Exact lookup: the fallbacks the playbook assigns to this rule, pattern, and posture."""
    return [
        fb for fb in rubric.rule(rule_id).fallbacks
        if pattern_id in fb.applies_to and posture in fb.postures
    ]


class Retriever:
    def __init__(self, rubric: Rubric, index_dir: Path = INDEX_DIR, embedding_function=None):
        self.rubric = rubric
        self.client = chromadb.PersistentClient(path=str(index_dir))
        self.embed = embedding_function or DefaultEmbeddingFunction()
        self.playbook = self._sync_playbook_index()

    def _collection(self, name: str, fingerprint: str = ""):
        return self.client.get_or_create_collection(
            name,
            embedding_function=self.embed,
            metadata={"fingerprint": fingerprint} if fingerprint else None,
            configuration={"hnsw": {"space": "cosine"}},  # distance = 1 - cosine similarity
        )

    def _sync_playbook_index(self):
        """Rebuild the index only when the rubric's fallbacks have changed.

        The fingerprint is a hash of every index document. Edit a fallback in
        rubric.yaml and the hash changes, so the stale index is replaced.
        """
        docs = {
            fb.id: (rule, fb, index_document(rule, fb))
            for rule in self.rubric.rules for fb in rule.fallbacks
        }
        fingerprint = hashlib.sha256("".join(d for _, _, d in docs.values()).encode()).hexdigest()

        existing = self._collection("playbook")
        if existing.metadata and existing.metadata.get("fingerprint") == fingerprint:
            return existing

        self.client.delete_collection("playbook")
        collection = self._collection("playbook", fingerprint)
        collection.add(
            ids=list(docs),
            documents=[d for _, _, d in docs.values()],
            metadatas=[
                {
                    "rule_id": rule.id,
                    "action": fb.action,
                    # Chroma metadata values must be scalars, so each posture is its own flag.
                    **{f"posture_{p.value}": p in fb.postures for p in Posture},
                }
                for rule, fb, _ in docs.values()
            ],
        )
        return collection

    def search(self, clause_text: str, posture: Posture, rule_id: str | None = None, k: int = 3) -> list[RetrievedText]:
        """Nearest playbook fallbacks for a flagged clause, limited to the posture
        (and to one rule, if the caller knows it)."""
        conditions = [{f"posture_{posture.value}": True}]
        if rule_id:
            conditions.append({"rule_id": rule_id})
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        result = self.playbook.query(query_texts=[clause_text], n_results=k, where=where)
        fallbacks = {fb.id: fb for rule in self.rubric.rules for fb in rule.fallbacks}
        return [
            RetrievedText(
                source="playbook",
                id=fb_id,
                rule_id=meta["rule_id"],
                action=meta["action"],
                text=fallbacks[fb_id].text,
                distance=round(distance, 4),
            )
            for fb_id, meta, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0])
        ]

    # ---- reference corpus (e.g. Bonterms / oneNDA-style language) ----------

    def index_corpus(self, corpus_dir: Path = CORPUS_DIR) -> int:
        """Chunk every NDA in corpus_dir by clause and index the chunks.
        Returns the number of chunks. Rebuilt from scratch on each call."""
        if "corpus" in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection("corpus")
        collection = self._collection("corpus")
        count = 0
        for path in sorted(corpus_dir.glob("*")):
            if path.suffix.lower() not in SUPPORTED:
                continue
            doc = parse_nda(path)
            chunks = [c for c in doc.clauses if c.id not in ("preamble", "signature")]
            if chunks:
                collection.add(
                    ids=[f"{path.stem}:{c.id}" for c in chunks],
                    documents=[c.text for c in chunks],
                    metadatas=[{"source_name": path.name, "clause_id": c.id} for c in chunks],
                )
                count += len(chunks)
        return count

    def search_corpus(self, clause_text: str, k: int = 3) -> list[RetrievedText]:
        collection = self._collection("corpus")
        if collection.count() == 0:
            return []
        result = collection.query(query_texts=[clause_text], n_results=min(k, collection.count()))
        return [
            RetrievedText(source="corpus", id=chunk_id, text=text, distance=round(distance, 4))
            for chunk_id, text, distance in zip(result["ids"][0], result["documents"][0], result["distances"][0])
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find playbook fallbacks for a clause.")
    parser.add_argument("--posture", required=True, choices=[p.value for p in Posture])
    parser.add_argument("--text", help="clause text to search with")
    parser.add_argument("--file", help="an NDA file; use with --clause")
    parser.add_argument("--clause", help="clause id within --file, e.g. sec-8.3")
    parser.add_argument("--rule", help="limit results to one rubric rule")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    if args.file and args.clause:
        text = parse_nda(args.file).clause(args.clause).text
    elif args.text:
        text = args.text
    else:
        parser.error("give --text, or --file with --clause")

    retriever = Retriever(load_rubric())
    print(f"Query ({args.posture}): {text[:120]}{'...' if len(text) > 120 else ''}\n")
    for r in retriever.search(text, Posture(args.posture), rule_id=args.rule, k=args.k):
        print(f"  {r.distance:.3f}  {r.id:<32} [{r.rule_id}, {r.action}]")


if __name__ == "__main__":
    main()
