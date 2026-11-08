"""Turn a 10-K (inline-XBRL HTML) into clean text split by Item."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "Reserved",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# first word(s) expected in each Item's title; tells real headings from running page headers
_TITLE_KEYS = {
    "1": "business", "1A": "risk", "1B": "unresolved", "1C": "cybersecurity", "2": "properties",
    "3": "legal", "4": "mine", "5": "market", "6": "reserved|selected", "7": "management",
    "7A": "quantitative", "8": "financial", "9": "changes", "9A": "controls", "9B": "other",
    "9C": "disclosure|foreign", "10": "directors", "11": "executive", "12": "security",
    "13": "certain", "14": "principal", "15": "exhibit", "16": "form|10-k|summary",
}

# "Item 1A." / "ITEM 7A:" / "Item 1A —" at the start of a line. Two-digit items come first in the
# alternation so "Item 10" is not read as "Item 1"; [ \t] (not \s) keeps the match on one line.
_HEADING = re.compile(
    r"^[ \t]*item[ \t]*(1[0-6]|1[abc]|[1-6]|7a|7|8|9[abc]|9)[ \t]*[\.\:\-—–]?[ \t]*(.{0,120})$",
    re.I | re.M,
)
_TOC_ROW = re.compile(r"\|\s*\d{1,3}\s*$")  # "Item 1A. | Risk Factors | 5"
_BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"]


@dataclass
class Section:
    item: str
    title: str
    text: str


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    for t in soup.find_all(lambda tag: tag.name and tag.name.lower() == "ix:header"):
        t.decompose()  # hidden XBRL header block
    for t in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        t.decompose()
    # tables -> pipe-delimited rows so numbers stay aligned with their labels
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in ("$", ")", "%")]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n" + "\n".join(rows) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    # newline only after block elements; inline spans (small-caps "B" + "USINESS") must stay joined
    for blk in soup.find_all(_BLOCK_TAGS):
        blk.append("\n")
    text = soup.get_text("")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # drop bare page numbers / running footers like "Apple Inc. | 2026 Form 10-K | 12"
    text = re.sub(r"^\s*(\d{1,3}|.{0,60}\|\s*\d{4} Form 10-K\s*\|\s*\d{1,3})\s*$", "", text, flags=re.M)
    return text.strip()


def _title_level(text: str, m: re.Match) -> int:
    """2 = title on the heading line, 1 = title on the next non-blank line, 0 = bare heading."""
    key = _TITLE_KEYS[m.group(1).upper()]
    same_line = m.group(2).strip()
    if re.match(rf"\W*({key})", same_line, re.I):
        return 2
    if not same_line:
        nxt = text[m.end():m.end() + 200].lstrip("\n ").split("\n", 1)[0]
        if re.match(rf"\W*({key})", nxt, re.I):
            return 1
    return 0


def split_items(text: str) -> list[Section]:
    """Split a 10-K's text into Items.

    1. Drop table-of-contents rows.
    2. Per Item keep only its best-titled occurrences; this drops bare running page headers
       ("PART II / Item 7") whenever a real "ITEM 7. MANAGEMENT'S ..." heading exists.
    3. Per Item keep the occurrence that starts the longest section (beats leftover cross-refs).
    """
    matches = [m for m in _HEADING.finditer(text) if not _TOC_ROW.search(m.group(0))]
    if not matches:
        return [Section("full", "Full Document", text)]
    level = {m.start(): _title_level(text, m) for m in matches}
    top: dict[str, int] = {}
    for m in matches:
        item = m.group(1).upper()
        top[item] = max(top.get(item, 0), level[m.start()])
    matches = [m for m in matches if level[m.start()] == top[m.group(1).upper()]]

    best: dict[str, tuple[int, int]] = {}  # item -> (start, length)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item = m.group(1).upper()
        length = end - m.start()
        if item not in best or length > best[item][1]:
            best[item] = (m.start(), length)

    starts = sorted((start, item) for item, (start, _) in best.items())
    sections = []
    for i, (start, item) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[start:end].strip()
        if len(body) < 200:  # "Item 6. [Reserved]" etc.
            continue
        sections.append(Section(item, ITEM_TITLES.get(item, f"Item {item}"), body))
    return sections


def parse_10k(html: str | bytes) -> list[Section]:
    return split_items(html_to_text(html))
