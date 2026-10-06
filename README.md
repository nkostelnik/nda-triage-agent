# NDA Triage Agent

Checks an inbound NDA against an attorney-authored playbook and returns a GREEN / YELLOW / RED classification, flagged clauses with the rule each one violates, and proposed redlines grounded in the playbook's fallback language.

**Output from this tool requires attorney review. It is not legal advice.** Use test data only. Never run real client or confidential NDAs through it.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Playbook rubric (`playbook/rubric.yaml`) | Done |
| 2 | Ingestion: parse, split into clauses, normalize | Done |
| 3 | Retrieval (RAG) over playbook fallbacks | Done |
| 4 | Reasoning: Claude tool use + Pydantic validation | Not started |
| 5 | Agent loop: retrieve, draft redline, self-check | Not started |
| 6 | Eval harness | Not started |
| 7 | CLI | Not started |

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # Windows; on macOS/Linux use .venv/bin/python
.venv/Scripts/python -m pytest -q
```

## Stage 1: the rubric

`playbook/rubric.yaml` holds 9 rules: mutuality, term, survival, non-solicit, non-compete, IP assignment, residuals, indemnification, and the definition of confidential information. Every position came from a structured interview with the playbook owner. The tool did not invent any of them.

Design choices:

- **Posture-dependent criteria.** Each rule sorts a clause into a named fact pattern, and each pattern has a color for each of three postures: discloser, recipient, and mutual. The same clause can be fine on one side of a deal and a problem on the other. For example, a residuals clause is GREEN for a recipient and RED for a discloser.
- **Posture comes from the user, with a cross-check.** The lawyer states the posture with `--posture`. The agent also reads the document's structure and warns if the two conflict.
- **Modifiers.** A few positions are adjustments rather than patterns. For example, a missing termination-on-notice right bumps the term color up one level.
- **Fallbacks are addressable.** Each fallback has an `id` and lists the patterns it fixes (`applies_to`). This lets the eval harness check whether retrieval returned the right fallback, not just a plausible-looking one.
- **Overall verdict: the worst rule wins.**
- **Provenance notes.** Where an interview answer was skipped, corrected, or changed on review, the rule's `notes` field records it.

`nda_triage/rubric.py` loads the YAML into Pydantic models and rejects a bad rubric when it loads: unknown keys (typos), a pattern missing a posture, a fallback pointing at a pattern that doesn't exist, or duplicate ids.

## Stage 2: ingestion

`nda_triage/ingest.py` turns a file into a `Document` of `Clause`s (`nda_triage/models.py`) in three steps:

1. **Read** `.txt` or `.docx` into paragraphs. Word keeps automatic section numbers ("3.2", "(a)") outside the paragraph text, so a plain text extraction loses them. The reader rebuilds the visible numbers from the document's numbering definitions. Tables are read in place. In `.txt` files, hard-wrapped lines are rejoined.
2. **Normalize** the Unicode: curly quotes, non-breaking spaces, long dashes, ligatures, and extra whitespace all get one standard form, so later stages see consistent text.
3. **Split** on the document's own structure, not fixed-size chunks. A new clause starts at each numbered section ("3.", "3.2", "Section 3", "Article IV"). Lettered subparts like "(a)" stay with their parent, because an exclusions list is one clause. Text before the first section becomes `preamble`, and "IN WITNESS WHEREOF" starts `signature`. Unnumbered documents fall back to headings, then to one clause per paragraph.

Each clause gets a stable id (`sec-8.3`), its heading, and character offsets into the full document text. This is what lets every flag trace back to the exact source text. The `Document` also records a SHA-256 hash of the original file.

Try it:

```bash
.venv/Scripts/python -m nda_triage.ingest samples/sample_nda.txt
.venv/Scripts/python -m nda_triage.ingest samples/sample_nda.txt --json
```

`samples/sample_nda.txt` is synthetic. Section 8.3 ("Cooperation") hides an employee non-solicit inside the miscellaneous provisions, which is the kind of trap the later stages need to catch.

Known limitations:

- The Word numbering rebuild covers how NDAs are normally numbered. It does not model every Word rule: list restarts through overrides, and two lists that share a numbering definition and continue each other.
- Headers, footers, footnotes, comments, and tracked-change markup in `.docx` are not read.
- Normalization changes characters (quotes, dashes). Clause text is therefore faithful to the wording of the source but not byte-identical to it. The file hash identifies the exact original.
- A paragraph that happens to begin like a section number ("2.5 million dollars...") would be treated as a new section.

## Stage 3: retrieval

`nda_triage/retrieval.py` embeds the playbook's fallbacks into a local Chroma index (`.chroma/`, gitignored). A flagged clause is embedded the same way, and its nearest fallbacks come back.

- **What gets embedded.** A query is a problem clause, but a fallback is the fix, and the two don't read alike ("neither party will solicit..." vs. "Delete the non-solicitation provision"). Each index document therefore holds the problem the fallback solves (the rule plus the descriptions of the patterns it applies to) followed by the fallback text.
- **Posture is a hard filter.** A recipient-only fallback is never offered to a discloser, however similar it looks. The rule can also be used as a filter when the caller knows it.
- **Embedding model: Chroma's default, all-MiniLM-L6-v2.** It's small (384 dimensions), runs locally on CPU, needs no API key, sends no text off the machine, and requires no extra package. It's weaker on legal phrasing than large hosted models. Swapping it is a one-line change in `Retriever.__init__`.
- **The index rebuilds itself** when any fallback in `rubric.yaml` changes. The collection stores a hash of its documents and is rebuilt on a mismatch.
- **Chunking.** Each fallback is one chunk. Reference NDAs placed in `corpus/` are chunked by clause using the Stage 2 splitter, so a chunk is always a whole section.
- **Exact lookup vs. search.** Once classification (Stage 4) has named the rule, pattern, and posture, `lookup_fallbacks` gets the right fallback straight from the rubric, with no embeddings. Semantic search is for cases where the classifier is unsure, and for the reference corpus. With only 12 fallbacks, exact lookup is the more reliable path. Embedding search is in this build to show the technique and to scale to a real corpus. The evals will compare the two.

Try it:

```bash
.venv/Scripts/python -m nda_triage.retrieval --posture discloser --file samples/sample_nda.txt --clause sec-5
.venv/Scripts/python -m nda_triage.retrieval --posture mutual --text "Recipient may use information retained in the unaided memory of its employees."
```

**A first observation, not yet a measured result:** for the sample's buried non-solicit (`sec-8.3`), unfiltered search ranks the term fallback first (distance 0.553) and the non-solicit fallback second (0.559). The clause mixes cooperation and duration language ("during the term and for twelve (12) months"), and a small embedding model latches onto the duration. Clear single-topic clauses (survival, term, residuals) rank the right fallback first. This is why the agent will pass the classified rule as a filter, or use exact lookup, instead of trusting unfiltered similarity. Stage 6 will measure retrieval accuracy properly.
