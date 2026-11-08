"""LLM (DSPy) extraction of graph entities from 10-K sections."""
from __future__ import annotations

import re
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from ..config import get_settings
from ..db.graph import company_key
from .parse import Section

RiskCategory = Literal[
    "macroeconomic", "competition", "supply_chain", "regulatory_legal", "geopolitical", "technology",
    "cybersecurity", "financial", "operational", "talent", "reputation", "environmental", "other",
]


class Executive(BaseModel):
    name: str
    title: str


class Risk(BaseModel):
    title: str = Field(description="short risk name, at most 10 words, no company name")
    category: RiskCategory


class ExtractBusinessFacts(dspy.Signature):
    """Extract structured facts from the Business section of a company's annual report (Form 10-K).
    Only include facts explicitly stated in the text. Use short canonical names (e.g. 'iPhone', 'Azure')."""

    company: str = dspy.InputField()
    text: str = dspy.InputField(desc="excerpt of Item 1 (Business) and executive officer information")
    segments: list[str] = dspy.OutputField(desc="reportable business segments as named in the filing")
    products: list[str] = dspy.OutputField(desc="major products, services or platforms; at most 15")
    competitors: list[str] = dspy.OutputField(desc="competitor companies explicitly named; at most 15; [] if none named")
    executives: list[Executive] = dspy.OutputField(desc="executive officers with their titles; [] if not listed")


class ExtractRisks(dspy.Signature):
    """List the distinct risk factors described in this excerpt of a 10-K Risk Factors section (Item 1A).
    Return one entry per risk factor heading or distinct risk, phrased generically so the same risk can be
    matched across companies (e.g. 'Dependence on third-party manufacturing in Asia')."""

    company: str = dspy.InputField()
    text: str = dspy.InputField()
    risks: list[Risk] = dspy.OutputField(desc="at most 12 risks")


class GraphExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.business = dspy.Predict(ExtractBusinessFacts)
        self.risks = dspy.Predict(ExtractRisks)

    def forward(self, company: str, sections: list[Section], max_risk_windows: int = 3) -> dict:
        s = get_settings()
        by_item = {sec.item: sec.text for sec in sections}
        out = {"segments": [], "products": [], "competitors": [], "executives": [], "risks": []}

        business = by_item.get("1", "")
        if business:
            # head of Item 1 (overview, segments, products) + the Competition and Executive Officers
            # subsections, which often sit deep inside long Business sections
            text = "\n...\n".join(filter(None, [
                business[: int(s.extract_max_chars * 0.5)],
                _block(business, r"\n\s*competition\s*\n", 5000),
                _executive_block(by_item),
            ]))
            r = self.business(company=company, text=text)
            out["segments"] = _clean(r.segments)
            out["products"] = _clean(r.products)[:15]
            out["competitors"] = [{"name": c, "key": company_key(c)} for c in _clean(r.competitors)[:15]]
            out["executives"] = [e.model_dump() for e in r.executives if e.name.strip()]

        risk_text = by_item.get("1A", "")
        seen = set()
        for window in _windows(risk_text, s.extract_max_chars)[:max_risk_windows]:
            for rk in self.risks(company=company, text=window).risks:
                key = rk.title.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out["risks"].append(rk.model_dump())
        return out


def _executive_block(by_item: dict[str, str]) -> str:
    for item in ("1", "10"):
        text = by_item.get(item, "")
        m = re.search(r"(information about )?(our )?executive officers", text, re.I)
        if m:
            return text[m.start(): m.start() + 6000]
    return ""


def _block(text: str, heading: str, size: int) -> str:
    m = re.search(heading, text, re.I)
    return text[m.start(): m.start() + size] if m else ""


def _windows(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] if text else []


def _clean(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items or []:
        x = x.strip().strip(".")
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out
