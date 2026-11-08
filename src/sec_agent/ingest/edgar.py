"""Minimal SEC EDGAR client: ticker->CIK, 10-K filing list, document download, XBRL company facts.

SEC fair-access rules: max 10 requests/second and a descriptive User-Agent with contact info.
"""
from __future__ import annotations

import threading
import time
from datetime import date, datetime
from functools import lru_cache

import httpx

from ..config import get_settings

_lock = threading.Lock()
_last = 0.0
_MIN_INTERVAL = 0.15  # ~6-7 req/s, safely under the 10 req/s limit


def _get(url: str) -> httpx.Response:
    global _last
    with _lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
    r = httpx.get(
        url,
        headers={"User-Agent": get_settings().sec_user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=60,
        follow_redirects=True,
    )
    r.raise_for_status()
    return r


@lru_cache
def ticker_map() -> dict[str, dict]:
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    return {
        v["ticker"].upper(): {"cik": str(v["cik_str"]).zfill(10), "ticker": v["ticker"].upper(), "name": v["title"]}
        for v in data.values()
    }


def resolve_company(ticker: str) -> dict:
    try:
        return ticker_map()[ticker.upper()]
    except KeyError:
        raise ValueError(f"Unknown ticker: {ticker}") from None


def list_10k_filings(cik: str, limit: int = 3, include_amendments: bool = False) -> list[dict]:
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    recent = sub["filings"]["recent"]
    forms = {"10-K", "10-K/A"} if include_amendments else {"10-K"}
    out = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        acc = recent["accessionNumber"][i]
        report = recent["reportDate"][i] or recent["filingDate"][i]
        out.append({
            "accession": acc,
            "cik": cik,
            "form": form,
            "period_of_report": _d(report),
            "filed_at": _d(recent["filingDate"][i]),
            "fiscal_year": _d(report).year,
            "url": (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{acc.replace('-', '')}/{recent['primaryDocument'][i]}"),
        })
        if len(out) >= limit:
            break
    return out


def download(url: str) -> bytes:
    """Raw bytes: let BeautifulSoup detect the charset from the document's own meta tag."""
    return _get(url).content


def company_facts(cik: str) -> dict:
    return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json()


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ------------------------------------------------------------------ XBRL normalization
# friendly metric -> candidate us-gaap concepts (first available per period wins)
METRICS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "research_and_development": ["ResearchAndDevelopmentExpense",
                                 "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "share_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}


def normalize_facts(ticker: str, facts: dict) -> list[dict]:
    """Annual 10-K values per metric and fiscal period end. Latest filing wins for restated values."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    best: dict[tuple[str, date], dict] = {}
    for metric, concepts in METRICS.items():
        for priority, concept in enumerate(concepts):
            for unit, points in gaap.get(concept, {}).get("units", {}).items():
                for p in points:
                    if p.get("form") not in ("10-K", "10-K/A") or p.get("fp") != "FY":
                        continue
                    end = _d(p["end"])
                    start = _d(p["start"]) if p.get("start") else None
                    if start and not 330 <= (end - start).days <= 380:
                        continue  # skip quarterly / multi-year durations reported in the 10-K
                    row = {
                        "ticker": ticker, "concept": concept, "metric": metric, "unit": unit,
                        "value": p["val"], "fiscal_year": end.year, "period_start": start,
                        "period_end": end, "accession": p["accn"], "filed_at": _d(p["filed"]),
                        "_priority": priority,
                    }
                    key = (metric, end)
                    cur = best.get(key)
                    if cur is None or (priority, -row["filed_at"].toordinal()) < (
                            cur["_priority"], -cur["filed_at"].toordinal()):
                        best[key] = row
    rows = list(best.values())
    for r in rows:
        r.pop("_priority")
    return rows
