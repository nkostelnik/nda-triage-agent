# NDA Triage Agent

Checks an inbound NDA against an attorney-authored playbook and returns a GREEN / YELLOW / RED classification, flagged clauses with the rule each one violates, and proposed redlines grounded in the playbook's fallback language.

**Output from this tool requires attorney review. It is not legal advice.** Use test data only. Never run real client or confidential NDAs through it.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Playbook rubric (`playbook/rubric.yaml`) | Done |
| 2 | Ingestion: parse, split into clauses, normalize | Not started |
| 3 | Retrieval (RAG) over playbook fallbacks | Not started |
| 4 | Reasoning: Claude tool use + Pydantic validation | Not started |
| 5 | Agent loop: retrieve, draft redline, self-check | Not started |
| 6 | Eval harness | Not started |
| 7 | CLI | Not started |

## Stage 1: the rubric

`playbook/rubric.yaml` holds 9 rules: mutuality, term, survival, non-solicit, non-compete, IP assignment, residuals, indemnification, and the definition of confidential information. Every position came from a structured interview with the playbook owner. The tool did not invent any of them.

Design choices:

- **Posture-dependent criteria.** Each rule sorts a clause into a named fact pattern, and each pattern has a color for each of three postures: discloser, recipient, and mutual. The same clause can be fine on one side of a deal and a problem on the other. For example, a residuals clause is GREEN for a recipient and RED for a discloser.
- **Posture comes from the user, with a cross-check.** The lawyer states the posture with `--posture`. The agent also reads the document's structure and warns if the two conflict.
- **Modifiers.** A few positions are adjustments rather than patterns. For example, a missing termination-on-notice right bumps the term color up one level.
- **Fallbacks are addressable.** Each fallback has an `id` and lists the patterns it fixes (`applies_to`). This lets the eval harness check whether retrieval returned the right fallback, not just a plausible-looking one.
- **Overall verdict: the worst rule wins.**
- **Provenance notes.** Where an interview answer was skipped or corrected, the rule's `notes` field records it. Open items to revisit after evals: the discloser-side term position (YELLOW at every length) and the "labeled mutual but asymmetric" pattern, which has no colors yet.
