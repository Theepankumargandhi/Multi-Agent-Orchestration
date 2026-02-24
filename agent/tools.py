import math
import numexpr
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults, ArxivQueryRun
from duckduckgo_search import DDGS

web_search = DuckDuckGoSearchResults()

# Kinda busted since it doesn't return links
arxiv_search = ArxivQueryRun()

_QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "for", "of", "in", "on", "at", "by", "with",
    "about", "from", "into", "is", "are", "be", "was", "were", "as", "that", "this",
    "latest", "recent", "current", "news", "update", "updates", "headline", "headlines",
    "today", "now", "week", "weekly", "month", "monthly", "year", "daily", "past", "last",
    "only", "include", "including", "sources", "source", "days", "day",
}


def _timelimit_for_days(recency_days: int | None) -> str | None:
    if recency_days is None or recency_days <= 0:
        return None
    if recency_days <= 1:
        return "d"
    if recency_days <= 7:
        return "w"
    if recency_days <= 31:
        return "m"
    return "y"


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None

    iso_candidate = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_candidate)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d %b %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue

    month_with_day = re.search(
        r"\b("
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")\s+(\d{1,2})(?:,\s*(\d{4}))?\b",
        text,
        flags=re.IGNORECASE,
    )
    if month_with_day:
        month_text = month_with_day.group(1).title()
        day = int(month_with_day.group(2))
        year = int(month_with_day.group(3)) if month_with_day.group(3) else datetime.now(timezone.utc).year
        for fmt in ("%b %d %Y", "%B %d %Y"):
            try:
                return datetime.strptime(f"{month_text} {day} {year}", fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue

    match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _result_datetime(item: dict[str, Any]) -> datetime | None:
    for key in ("date", "datetime", "published", "published_date", "publishedAt", "time"):
        if key in item:
            dt = _parse_datetime(item.get(key))
            if dt:
                return dt

    # Some providers only include publish date inside snippets/title text.
    for key in ("snippet", "body", "title"):
        if key in item:
            dt = _parse_datetime(item.get(key))
            if dt:
                return dt
    return None


def _extract_query_terms(query: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
    tokens: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        if token in _QUERY_STOPWORDS:
            continue
        if len(token) == 1 and token not in {"x", "y"}:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _contains_token(text: str, token: str) -> bool:
    if token == "ai":
        return bool(re.search(r"\b(ai|a\.i\.|artificial intelligence)\b", text))
    return bool(re.search(rf"\b{re.escape(token)}\b", text))


def _relevance_score(item: dict[str, Any], query_terms: list[str]) -> int:
    haystack = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("body") or ""),
            str(item.get("snippet") or ""),
        ]
    ).lower()
    if not haystack:
        return 0

    hits = 0
    for term in query_terms:
        if _contains_token(haystack, term):
            hits += 1
    return hits


def _filter_by_relevance(
    items: list[dict[str, Any]],
    relevance_query: str,
    strict: bool,
) -> list[dict[str, Any]]:
    query_terms = _extract_query_terms(relevance_query)
    if not query_terms:
        return items

    min_hits = 1
    if strict and len(query_terms) >= 2:
        min_hits = 2

    def score_with_threshold(required_hits: int) -> list[tuple[int, dict[str, Any]]]:
        scored_local: list[tuple[int, dict[str, Any]]] = []
        for item in items:
            score = _relevance_score(item, query_terms)
            if score >= required_hits:
                scored_local.append((score, item))
        return scored_local

    scored = score_with_threshold(min_hits)
    if not scored and strict and min_hits > 1:
        scored = score_with_threshold(1)

    if not scored:
        return []

    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in scored]


def _sort_by_freshness(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    floor = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return sorted(items, key=lambda item: _result_datetime(item) or floor, reverse=True)


def _filter_by_recency(
    items: list[dict[str, Any]],
    recency_days: int | None,
    require_dated: bool,
) -> list[dict[str, Any]]:
    if recency_days is None or recency_days <= 0:
        return items

    cutoff = datetime.now(timezone.utc) - timedelta(days=recency_days)
    filtered: list[dict[str, Any]] = []
    for item in items:
        dt = _result_datetime(item)
        if dt is None:
            if not require_dated:
                filtered.append(item)
            continue
        if dt >= cutoff:
            filtered.append(item)
    return filtered


def _normalize_results(items: Iterable[dict[str, Any]], max_results: int = 5) -> str:
    lines = []
    for i, item in enumerate(items):
        if i >= max_results:
            break
        title = (item.get("title") or "").strip()
        href = (item.get("href") or item.get("url") or "").strip()
        snippet = (item.get("body") or item.get("snippet") or "").strip()
        published_at = _result_datetime(item)
        if not (title or href or snippet):
            continue
        if published_at:
            published_line = published_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
            lines.append(f"- {title}\n  Link: {href}\n  Date: {published_line}\n  Snippet: {snippet}")
        else:
            lines.append(f"- {title}\n  Link: {href}\n  Snippet: {snippet}")
    return "\n".join(lines)


def perform_web_search(
    query: str,
    max_results: int = 5,
    recency_days: int | None = None,
    relevance_query: str | None = None,
) -> str:
    """
    Robust web retrieval with retries/fallbacks for news-heavy queries.

    Order:
    1) DuckDuckGo news
    2) DuckDuckGo text
    3) LangChain DuckDuckGoSearchResults wrapper
    """
    q = (query or "").strip()
    if not q:
        return "Web retrieval failed: empty query."

    errors: list[str] = []
    timelimit = _timelimit_for_days(recency_days)
    topic_query = (relevance_query or q).strip()

    # Attempt 1: news search (best for 'latest/news' style prompts).
    try:
        with DDGS(timeout=20) as ddgs:
            kwargs: dict[str, Any] = {"max_results": max_results}
            if timelimit:
                kwargs["timelimit"] = timelimit
            try:
                news = list(ddgs.news(q, **kwargs))
            except TypeError:
                kwargs.pop("timelimit", None)
                news = list(ddgs.news(q, **kwargs))
        filtered_news = _filter_by_recency(news, recency_days, require_dated=True)
        filtered_news = _filter_by_relevance(
            filtered_news,
            relevance_query=topic_query,
            strict=bool(recency_days),
        )
        filtered_news = _sort_by_freshness(filtered_news)
        formatted = _normalize_results(filtered_news, max_results=max_results)
        if formatted:
            return formatted
        if recency_days:
            errors.append(f"ddg.news: no dated results within last {recency_days} days")
        if topic_query:
            errors.append("ddg.news: no topical results matching query terms")
    except Exception as e:
        errors.append(f"ddg.news: {e}")

    # Attempt 2: text search fallback.
    try:
        with DDGS(timeout=20) as ddgs:
            kwargs = {"max_results": max_results}
            if timelimit:
                kwargs["timelimit"] = timelimit
            try:
                text_results = list(ddgs.text(q, **kwargs))
            except TypeError:
                kwargs.pop("timelimit", None)
                text_results = list(ddgs.text(q, **kwargs))
        filtered_text = _filter_by_recency(text_results, recency_days, require_dated=True)
        filtered_text = _filter_by_relevance(
            filtered_text,
            relevance_query=topic_query,
            strict=bool(recency_days),
        )
        filtered_text = _sort_by_freshness(filtered_text)
        formatted = _normalize_results(filtered_text, max_results=max_results)
        if formatted:
            return formatted
        if recency_days:
            errors.append(f"ddg.text: no results within last {recency_days} days")
        if topic_query:
            errors.append("ddg.text: no topical results matching query terms")
    except Exception as e:
        errors.append(f"ddg.text: {e}")

    # If strict recency produced nothing, relax to "freshest available relevant".
    if recency_days:
        try:
            with DDGS(timeout=20) as ddgs:
                news_relaxed = list(ddgs.news(q, max_results=max(max_results * 3, 15)))
            relaxed_news = _filter_by_relevance(news_relaxed, relevance_query=topic_query, strict=False)
            relaxed_news = _sort_by_freshness(relaxed_news)
            formatted = _normalize_results(relaxed_news, max_results=max_results)
            if formatted:
                return formatted
            errors.append("ddg.news: no relevant results after relaxed recency fallback")
        except Exception as e:
            errors.append(f"ddg.news.relaxed: {e}")

        try:
            with DDGS(timeout=20) as ddgs:
                text_relaxed = list(ddgs.text(q, max_results=max(max_results * 3, 15)))
            relaxed_text = _filter_by_relevance(text_relaxed, relevance_query=topic_query, strict=False)
            relaxed_text = _sort_by_freshness(relaxed_text)
            formatted = _normalize_results(relaxed_text, max_results=max_results)
            if formatted:
                return formatted
            errors.append("ddg.text: no relevant results after relaxed recency fallback")
        except Exception as e:
            errors.append(f"ddg.text.relaxed: {e}")

    # Attempt 3: original wrapper fallback.
    try:
        raw = web_search.invoke(q)
        raw_text = str(raw).strip() if raw else ""
        if raw_text:
            # Only trust this fallback if it contains at least one URL.
            if re.search(r"https?://", raw_text):
                return raw_text
            errors.append("langchain.ddg: returned content without URLs")
    except Exception as e:
        errors.append(f"langchain.ddg: {e}")

    return "Web retrieval failed: " + " | ".join(errors[:3])

@tool
def calculator(expression: str) -> str:
    """Calculates a math expression using numexpr.
    
    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )
