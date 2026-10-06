"""Stage 2: ingestion. A file goes in; a list of traceable clauses comes out.

    read  ->  normalize  ->  split  ->  Document

1. Read. Turn a .txt or .docx file into a list of paragraphs. Word stores
   automatic section numbers ("3.2") outside the paragraph text, so for .docx
   we rebuild the visible numbers from the document's numbering definitions.
2. Normalize. Clean up Unicode so the same words always look the same:
   curly quotes, non-breaking spaces, long dashes, stray whitespace.
3. Split. Start a new clause at each numbered section. NDAs are drafted in
   sections, and the rubric's rules map onto sections, so we split on the
   document's own structure instead of fixed-size chunks that could cut a
   clause in half. Lettered subparts like "(a)" stay with their parent.

Run it directly to see the clauses:  python -m nda_triage.ingest path/to/nda.docx
"""

import argparse
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from nda_triage.models import Clause, Document

SUPPORTED = {".txt", ".docx"}


@dataclass
class Para:
    text: str
    is_heading_style: bool = False  # .docx paragraph styled "Heading 1", "Title", etc.


# --------------------------------------------------------------------------
# 1. Read
# --------------------------------------------------------------------------

def read_paragraphs(path: Path) -> list[Para]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_txt(path)
    if suffix == ".docx":
        return _read_docx(path)
    raise ValueError(f"Unsupported file type {suffix!r}. Supported: {sorted(SUPPORTED)} (PDF is planned).")


def _read_txt(path: Path) -> list[Para]:
    """Blank lines separate paragraphs. Hard-wrapped lines inside a paragraph
    are rejoined, except that a line starting with a section number always
    starts a new paragraph (some files put every section on its own line with
    no blank line between them)."""
    paras: list[list[str]] = []
    starting_new = True
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            starting_new = True
            continue
        if starting_new or _section_number(normalize(line)):
            paras.append([line])
        else:
            paras[-1].append(line)
        starting_new = False
    return [Para(" ".join(lines)) for lines in paras]


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _read_docx(path: Path) -> list[Para]:
    """Walk the document body in order, paragraphs and tables alike. Headers,
    footers, footnotes, and comments are not read."""
    doc = docx.Document(str(path))
    numbering = _NumberingRebuilder(doc)
    paras = []
    for child in doc.element.body.iterchildren():
        if child.tag == W + "p":
            p = Paragraph(child, doc)
            label = numbering.label_for(p)
            text = f"{label} {p.text}" if label else p.text
            style = p.style.name if p.style is not None else ""
            paras.append(Para(text, is_heading_style=style.startswith(("Heading", "Title"))))
        elif child.tag == W + "tbl":
            for row in Table(child, doc).rows:
                cells = []
                for cell in row.cells:  # merged cells repeat; keep one copy
                    if not cells or cell.text != cells[-1]:
                        cells.append(cell.text)
                paras.append(Para(" | ".join(cells)))
    return paras


class _NumberingRebuilder:
    """Rebuild the numbers Word displays for auto-numbered paragraphs.

    A numbered paragraph stores only a list id and an indent level; the format
    ("%1.%2", decimal vs. letters) lives in numbering.xml. We read those
    formats, keep a running counter per list, and render the label. This
    covers the way NDAs are normally numbered. It is an approximation of
    Word's full rules (it does not model list restarts via overrides, or two
    lists that share one numbering definition and continue each other).
    """

    def __init__(self, doc):
        self.formats = self._read_formats(doc)  # num_id -> {level: (fmt, text, start)}
        self.counters: dict[str, list[int]] = {}

    @staticmethod
    def _read_formats(doc) -> dict:
        try:
            root = doc.part.numbering_part.element
        except (KeyError, NotImplementedError):
            return {}

        def val(el, tag, default):
            found = el.find(W + tag)
            return found.get(W + "val") if found is not None else default

        abstract = {}
        for an in root.findall(W + "abstractNum"):
            abstract[an.get(W + "abstractNumId")] = {
                int(lvl.get(W + "ilvl")): (
                    val(lvl, "numFmt", "decimal"),
                    val(lvl, "lvlText", ""),
                    int(val(lvl, "start", "1")),
                )
                for lvl in an.findall(W + "lvl")
            }
        return {
            num.get(W + "numId"): abstract.get(num.find(W + "abstractNumId").get(W + "val"), {})
            for num in root.findall(W + "num")
        }

    @staticmethod
    def _num_props(p: Paragraph):
        """(list id, level) if the paragraph is auto-numbered, either directly
        or through its style (e.g. a "Heading 1" style linked to a list)."""
        num_pr = p._p.pPr.numPr if p._p.pPr is not None else None
        style = p.style
        while num_pr is None and style is not None:
            style_ppr = style.element.pPr
            num_pr = style_ppr.numPr if style_ppr is not None else None
            style = style.base_style
        if num_pr is None or num_pr.numId is None or num_pr.numId.val == 0:
            return None
        level = num_pr.ilvl.val if num_pr.ilvl is not None else 0
        return str(num_pr.numId.val), level

    def label_for(self, p: Paragraph) -> str:
        props = self._num_props(p)
        if props is None:
            return ""
        num_id, level = props
        levels = self.formats.get(num_id, {})
        counts = self.counters.setdefault(num_id, [0] * 9)
        start = levels.get(level, ("decimal", "", 1))[2]
        counts[level] = counts[level] + 1 if counts[level] else start
        for deeper in range(level + 1, 9):
            counts[deeper] = 0

        fmt, template, _ = levels.get(level, ("decimal", f"%{level + 1}.", 1))
        if fmt in ("bullet", "none"):
            return ""

        def render(match):
            lvl = int(match.group(1)) - 1
            lvl_fmt, _, lvl_start = levels.get(lvl, ("decimal", "", 1))
            return _format_number(counts[lvl] or lvl_start, lvl_fmt)

        return re.sub(r"%([1-9])", render, template)


def _format_number(n: int, fmt: str) -> str:
    if fmt == "lowerLetter":
        return chr(ord("a") + (n - 1) % 26)
    if fmt == "upperLetter":
        return chr(ord("A") + (n - 1) % 26)
    if fmt in ("lowerRoman", "upperRoman"):
        roman = _to_roman(n)
        return roman.lower() if fmt == "lowerRoman" else roman
    return str(n)


def _to_roman(n: int) -> str:
    out = ""
    for value, numeral in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                           (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= value:
            out, n = out + numeral, n - value
    return out


# --------------------------------------------------------------------------
# 2. Normalize
# --------------------------------------------------------------------------

_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',  # curly quotes
    "–": "-", "—": "-", "−": "-",                  # long dashes, minus sign
    "­": "",                                                 # soft hyphen
}


def normalize(text: str) -> str:
    """NFKC folds look-alike characters (non-breaking space, ligatures like
    'ﬁ') into their plain forms; then we straighten quotes and dashes and
    collapse runs of whitespace."""
    text = unicodedata.normalize("NFKC", text)
    for fancy, plain in _REPLACEMENTS.items():
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# 3. Split
# --------------------------------------------------------------------------

# "3." "3)" "3.2" "3.2." followed by text. A bare "30 days..." does not count:
# a single number needs a trailing "." or ")" to be a section number.
_NUMBERED = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2})*)(?P<punct>[.)])?(?:\s+|$)")
# "Section 3", "Article IV"
_LABELED = re.compile(r"^(?:section|article)\s+(?P<num>\d{1,2}(?:\.\d{1,2})*|[IVXLC]+)\b[.:)]?\s*", re.I)
_SIGNATURE = re.compile(r"^in witness whereof\b", re.I)
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}


def _section_number(text: str) -> tuple[str, str] | None:
    """If the paragraph opens a numbered section, return (number, rest of text)."""
    m = _LABELED.match(text)
    if m:
        return m.group("num"), text[m.end():]
    m = _NUMBERED.match(text)
    if m and ("." in m.group("num") or m.group("punct")):
        return m.group("num"), text[m.end():]
    return None


def _looks_like_heading(text: str) -> bool:
    """Short, no closing punctuation, and in Title Case or ALL CAPS."""
    words = text.split()
    if not 0 < len(words) <= 8 or text[-1] in ".;:,":
        return False
    if text.isupper():
        return True
    return words[0][0].isupper() and all(w[0].isupper() or w.lower() in _SMALL_WORDS for w in words)


def _lead_in_heading(rest: str) -> str | None:
    """Pull the heading out of '3. Term. This Agreement...' or '3. TERM'."""
    if _looks_like_heading(rest.rstrip(".")):
        return rest.rstrip(".")
    candidate = rest.split(". ", 1)[0]
    if candidate != rest and _looks_like_heading(candidate):
        return candidate
    return None


@dataclass
class _Draft:
    id: str
    number: str | None
    heading: str | None
    paras: list[str]


def split_clauses(paras: list[Para]) -> list[_Draft]:
    """Three strategies, used in order of reliability:

    1. Numbered sections, if the document has any. Text before the first
       section becomes 'preamble'; 'IN WITNESS WHEREOF' starts 'signature'.
    2. Headings (Word heading styles or short Title Case / ALL CAPS lines),
       for unnumbered documents.
    3. One clause per paragraph, as a last resort.
    """
    texts = [p.text for p in paras]
    if any(_section_number(t) for t in texts):
        drafts = [_Draft("preamble", None, None, [])]
        for t in texts:
            numbered = _section_number(t)
            if numbered:
                number, rest = numbered
                drafts.append(_Draft(f"sec-{number.lower()}", number, _lead_in_heading(rest), [t]))
            elif _SIGNATURE.match(t):
                drafts.append(_Draft("signature", None, None, [t]))
            else:
                drafts[-1].paras.append(t)
        drafts = [d for d in drafts if d.paras]
    else:
        drafts = []
        for p in paras:
            if p.is_heading_style or _looks_like_heading(p.text) or not drafts:
                heading = p.text if (p.is_heading_style or _looks_like_heading(p.text)) else None
                drafts.append(_Draft(f"para-{len(drafts) + 1}", None, heading, [p.text]))
            else:
                drafts[-1].paras.append(p.text)
        if len(drafts) < 2:
            drafts = [_Draft(f"para-{i}", None, None, [t]) for i, t in enumerate(texts, 1)]

    _make_ids_unique(drafts)
    return drafts


def _make_ids_unique(drafts: list[_Draft]) -> None:
    """A document can reuse numbers (e.g. an exhibit that restarts at 1)."""
    seen: dict[str, int] = {}
    for d in drafts:
        if d.id in seen:
            seen[d.id] += 1
            d.id = f"{d.id}-{seen[d.id]}"
        else:
            seen[d.id] = 1


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------

def parse_nda(path: str | Path) -> Document:
    path = Path(path)
    raw = path.read_bytes()
    paras = [Para(normalize(p.text), p.is_heading_style) for p in read_paragraphs(path)]
    paras = [p for p in paras if p.text]
    parts = [(d.id, d.number, d.heading, "\n".join(d.paras)) for d in split_clauses(paras)]
    return assemble(path.name, hashlib.sha256(raw).hexdigest(), parts)


def assemble(source_name: str, sha256: str, parts: list[tuple[str, str | None, str | None, str]]) -> Document:
    """Lay (id, number, heading, text) clauses end to end in one string and
    record each one's offsets, so a flag on a clause can be highlighted in the
    full document. Also used to build a revised copy of a document."""
    clauses, offset = [], 0
    for clause_id, number, heading, text in parts:
        clauses.append(Clause(id=clause_id, number=number, heading=heading,
                              text=text, start=offset, end=offset + len(text)))
        offset += len(text) + 2  # the "\n\n" separator below
    return Document(
        source_name=source_name,
        sha256=sha256,
        text="\n\n".join(p[3] for p in parts),
        clauses=clauses,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split an NDA into clauses.")
    parser.add_argument("path")
    parser.add_argument("--json", action="store_true", help="print the full Document as JSON")
    args = parser.parse_args()

    doc = parse_nda(args.path)
    if args.json:
        print(doc.model_dump_json(indent=2))
        return
    print(f"{doc.source_name}  ({len(doc.clauses)} clauses, sha256 {doc.sha256[:12]}...)\n")
    for c in doc.clauses:
        preview = c.text.replace("\n", " ")
        preview = preview[:70] + "..." if len(preview) > 70 else preview
        print(f"{c.id:<12} {(c.heading or ''):<28.28} {preview}")


if __name__ == "__main__":
    main()
