# NDA Triage Agent

Checks an inbound NDA against an attorney-authored playbook and returns a GREEN / YELLOW / RED classification, flagged clauses with the rule each one violates, and proposed redlines grounded in the playbook's fallback language.

**Output from this tool requires attorney review. It is not legal advice.** Use test data only. Never run real client or confidential NDAs through it.

`playbook/rubric.yaml` is a demonstration playbook. It does not reflect the positions of any client or employer. All sample documents are synthetic, with fictional parties.

Licensed under the MIT License (see `LICENSE`).

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Playbook rubric (`playbook/rubric.yaml`) | Done |
| 2 | Ingestion: parse, split into clauses, normalize | Done |
| 3 | Retrieval (RAG) over playbook fallbacks | Done |
| 4 | Reasoning: Claude structured output + Pydantic validation | Done; run live on the sample (see Live results) |
| 5 | Agent loop: retrieve, draft redline, self-check | Done; run live on the sample (see Live results) |
| 6 | Eval harness | Done; first pass 14/14 on an easy set (see Stage 6) |
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

## Stage 4: reasoning

`nda_triage/classify.py` makes one Claude call (`claude-sonnet-5-5`) per rubric rule, plus one posture cross-check call. Each call sees the whole NDA and a single rule.

- **One call per rule, not per clause.** Several playbook positions are about what's *missing*: no termination right, no trade-secret carve-out, missing exclusions. A call that sees one clause at a time can't notice an absence. A call that reads the whole NDA for one rule can. The cost is sending the NDA ten times. Prompt caching handles that: the system prompt and the document form an identical prefix on every call. The first call runs alone to write the cache, and the nine rule calls then run in parallel and read it at about a tenth of the input price. Token use and estimated cost are reported on every run.
- **The model finds facts; the rubric decides colors.** The model isn't told the posture. It reports which of the rule's patterns the NDA matches, citing clause ids and verbatim quotes. Python then looks up each pattern's color for the stated posture, applies modifiers, and takes the worst color. The risk call is deterministic, auditable, and testable without the API.
- **Structured output, validated by Pydantic.** The response schema lists every pattern id in the rubric, so the model can only name patterns that exist, and the SDK validates the response against the Pydantic model. All rule calls share one schema so they share one cache entry (see Live results); Python rejects any id that belongs to a different rule and fails closed. Forcing a particular tool call isn't supported on this model, so structured outputs are the supported way to force JSON.
- **Verification.** Every cited clause id must exist, and every quote must appear in the cited clause. A finding that fails either check is marked `quote_verified: false`, and its rule can't be GREEN.
- **Fail closed.** If a rule can't be resolved (nothing matched and the rubric has no position for that, a citation didn't verify, or the model declined), the rule is YELLOW with a `needs_review` note. It is never silently GREEN.
- **Refusal fallback.** Requests opt into the API's server-side fallback (`fallbacks: "default"`). If the model declines a request, the API retries it on a fallback model. Every model that served a call is listed in the output's usage.
- **Who "us" is.** Patterns like "one-way, protects us" and "we indemnify them" need to know which party is the client, so the CLI takes `--party`.

The tests in `tests/test_classify.py` use a fake API client. They check everything except the model's judgment: prompt shape, the cacheable prefix, schemas, scoring, modifiers, verification, and failure handling. Stage 6 measures the model's judgment.

Run it (requires `ANTHROPIC_API_KEY` in the environment; the sample is synthetic):

```bash
.venv/Scripts/python -m nda_triage.classify samples/sample_nda.txt --posture mutual --party "Acme Analytics, Inc."
```

## Stage 5: the redline agent

`nda_triage/agent.py` runs a bounded loop for every YELLOW or RED finding:

1. **Retrieve** the fallback. Exact rubric lookup by rule, pattern, and posture comes first. Vector search, restricted to the rule, is used only if the rubric has no exact match. Vector search's unfiltered top 3 is recorded for every redline either way, so the evals can measure how often it would have picked the right fallback on its own.
2. **Draft** the redline with Claude, grounded in the fallback. The drafter may adapt only names, defined terms, and cross-references. For a strike fallback, it removes only the offending language, which matters when a non-solicit is buried inside an otherwise harmless clause.
3. **Check** the draft two ways:
   - **Re-review.** Apply the redline to a copy of the NDA and run the same Stage 4 classifier and rubric on it. The issue must no longer be flagged. The agent's fix is judged by the same reviewer that raised the flag, not by asking the drafter whether it did a good job.
   - **Fidelity.** A separate call confirms the draft adds nothing beyond the original clause and the fallback.
4. **Revise once** if either check fails, feeding the problems back. If the second draft also fails, the redline is returned marked `needs_attorney` with the reasons. It doesn't loop indefinitely.

Findings whose quotes couldn't be verified are not redlined, because drafting against an unverified citation would compound an error. Those rules are already marked `needs_review`.

**Why a bounded workflow rather than an open-ended agent:** the steps are fixed and capped at two attempts, so cost is predictable and every run can be audited step by step. For legal work that's worth more than letting the model decide its own next step.

The tests in `tests/test_agent.py` use a scripted fake client whose reviewer answers based on the document it's shown. A draft that actually removes the non-solicit passes re-review, and one that only rewords it is caught and revised.

Run it (requires `ANTHROPIC_API_KEY`):

```bash
.venv/Scripts/python -m nda_triage.agent samples/sample_nda.txt --posture mutual --party "Acme Analytics, Inc."
```

## Live results

The sample NDA was run end to end against the real API (`claude-sonnet-5-5`, mutual posture, client "Acme Analytics, Inc."). The full output is in `docs/sample_run.json`. This is one synthetic document, not an evaluation; Stage 6 is where accuracy gets measured.

**What it found.** The NDA was RED overall. Eight rules were GREEN and one was RED: the employee non-solicit hidden in section 8.3 ("Cooperation"), under the miscellaneous provisions. Every quote verified against the source text, and the posture cross-check agreed the NDA is mutual.

The model read the clause more precisely than the test fixture I'd written for it. The fixture called it a *narrow* non-solicit. The model reported two patterns instead: *broad*, because the clause has no carve-out for general job advertisements, and *long*, because "during the term and for twelve (12) months thereafter" can exceed 12 months when combined with the two-year term in section 4. Both readings are correct under the rubric.

**The redline.** The playbook fallback for a non-solicit is "strike." The agent removed only the restriction and kept the harmless covenant, and the draft passed both checks on the first attempt:

> **Before:** 8.3 Cooperation. The parties will cooperate in good faith, and during the term and for twelve (12) months thereafter neither party will solicit for employment any employee of the other party with whom it had contact under this Agreement.
>
> **After:** 8.3 Cooperation. The parties will cooperate in good faith.

Re-reviewing the revised NDA returned GREEN for the non-solicit rule, and the fidelity check found nothing added. Retrieval came from exact rubric lookup. Unfiltered vector search, recorded alongside it, again ranked the *term* fallback first for this clause, as Stage 3 predicted. This is why the agent doesn't rely on unfiltered similarity.

**Two bugs the first live run exposed, both fixed:**

| | First run | After fixes |
|---|---|---|
| Calls | 16 | 13 |
| Cache writes / reads (tokens) | 29,301 / 2,272 | 5,986 / 28,104 |
| Estimated cost | $0.11 | $0.05 |
| Wall time | 42 s | 16 s |
| Redlines | 2 (identical) | 1 |

1. **Caching barely worked.** Each rule call had its own response schema, and the schema is part of the cached prompt prefix, so the nine calls created nine separate cache entries instead of sharing one. The fix was a single schema shared by every rule call, with the per-rule check moved into Python. The offline tests couldn't catch this; only real token counts could. A test now asserts that all rule calls send the same schema.
2. **Duplicate redlines.** The model correctly flagged the clause under two patterns, and the agent drafted the same fix twice. Redlines are now deduplicated per rule, fallback, and clause. The re-review still checks every flag on that clause.

Verdicts were identical across both runs.

## Stage 6: evals

`evals/` holds 14 synthetic NDAs, the attorney's expected verdicts (`labels.json`), and a runner (`run_evals.py`). The set has 4 GREEN controls, 2 YELLOW, and 8 RED across all three postures. Each NDA tests something specific:

- **Traps the model has to read past:** a no-hire buried in Miscellaneous under "Personnel"; a non-compete phrased as a "use restriction" on developing similar products; an NDA titled "mutual" that protects only one side; an 18-month standstill in an M&A agreement.
- **Issues created by something missing:** no fixed term and no termination right; a missing independent-development exclusion.
- **Over-escalation traps:** clauses that sound alarming but are GREEN for the stated posture (broad residuals for a recipient; perpetual survival, a feedback clause, and an indemnity running *to* us for a discloser).
- **Paired cases:** a discloser's fixed survival period is RED without a trade-secret carve-out (case 03) and YELLOW with one (case 14).
- **Two Word files**, built by `evals/build_docx.py`: one with heading styles, one with Word's automatic numbering.

The labels are the attorney's, not the tool's. The runner refuses to score against `labels.json` until its status is `confirmed`. Case 14 was left unanswered in the review form, so it's still listed as unconfirmed.

The runner reports overall and per-rule accuracy, **false GREENs** (the dangerous error: the attorney expected YELLOW or RED and the tool said GREEN), under-calls, over-escalations (the cost side, which wastes counsel's time), posture-check accuracy, unverified quotes, and, with `--redlines`, redline pass rate and retrieval agreement. Every miss is written to `disagreements.jsonl` with the model's findings and reasons.

```bash
.venv/Scripts/python evals/run_evals.py              # classification only
.venv/Scripts/python evals/run_evals.py --redlines   # also draft and check redlines
```

### First full run: 14/14, and why that is not the headline

The first classification-only pass (140 calls, about $0.50) got every verdict right: 14 of 14 overall and 126 of 126 rule verdicts, with zero false GREENs, zero over-escalations, a correct posture check on all 14 (including the warning on the fake mutual NDA), and no unverified quotes. Full results are in `evals/RESULTS.md`. The colors were also right for the right reasons: the patterns and clauses the model cited match what each document was built to test. Caching held across the run (306K tokens read from cache vs. 62K written).

**A perfect score means this eval set is too easy to tell good from great.** Honest reasons:

- **The same author wrote the rubric and the test NDAs.** Each document isolates one or two issues in clean, unambiguous language. Real NDAs are longer, messier, and drafted by someone trying to win.
- **One run per document.** Model output varies from run to run, and a single pass doesn't measure that.
- **No hard cases yet:** issues split across defined terms and cross-references, clauses that override each other ("notwithstanding Section 4..."), several problems interacting in one section, or 10-page documents.
- **Redlines were not part of this pass.**

The eval is in place and it works. The next step is making it harder (see the roadmap below).
