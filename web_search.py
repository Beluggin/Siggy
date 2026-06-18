# web_search.py
"""
WEB SEARCH — SignalBot Internet Access Module

Gives SignalBot the ability to search the web and read current news.
Currently uses DuckDuckGo (no API key needed).
Scaffolded for Google Custom Search swap-in later.

TRIGGER:
  "search <query>" in chat — searches and injects results into prompt
  "news" — fetches top headlines
  "news <topic>" — fetches news on a specific topic

USAGE:
  from web_search import web_search, news_search, format_search_for_prompt

  results = web_search("anthropic blacklisted")
  prompt_block = format_search_for_prompt(results)
"""

import time
from typing import List, Dict, Optional

# ═══════════════════════════════════════════════════════════════════
# BACKEND SELECTOR — swap search providers here
# ═══════════════════════════════════════════════════════════════════

# Set to "google" when you get a Google API key
SEARCH_BACKEND = "ddgs"

# Google config (for later)
GOOGLE_API_KEY = ""
GOOGLE_CX = ""  # Custom Search Engine ID

# ═══════════════════════════════════════════════════════════════════
# DUCKDUCKGO BACKEND
# ═══════════════════════════════════════════════════════════════════

def _search_ddg(query: str, max_results: int = 5) -> List[Dict]:
    """Search using DuckDuckGo. No API key needed."""
    try:
        from ddgs import DDGS
    except ImportError:
        return [{"title": "ERROR", "body": "duckduckgo-search not installed. Run: pip install duckduckgo-search", "href": ""}]

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source": "duckduckgo",
                })
    except Exception as e:
        results.append({
            "title": "Search Error",
            "body": f"DuckDuckGo search failed: {e}",
            "url": "",
            "source": "error",
        })

    return results


def _news_ddg(query: str = "", max_results: int = 5) -> List[Dict]:
    """Fetch news using DuckDuckGo News."""
    try:
        from ddgs import DDGS
    except ImportError:
        return [{"title": "ERROR", "body": "duckduckgo-search not installed", "url": ""}]

    results = []
    try:
        with DDGS() as ddgs:
            keywords = query if query else "world news today"
            for r in ddgs.news(keywords, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("url", ""),
                    "date": r.get("date", ""),
                    "source": r.get("source", "news"),
                })
    except Exception as e:
        results.append({
            "title": "News Error",
            "body": f"DuckDuckGo news failed: {e}",
            "url": "",
            "source": "error",
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# GOOGLE BACKEND (SCAFFOLDING — activate later)
# ═══════════════════════════════════════════════════════════════════

def _search_google(query: str, max_results: int = 5) -> List[Dict]:
    """Search using Google Custom Search API. Needs API key + CX."""
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return [{"title": "ERROR", "body": "Google API key or CX not configured", "url": ""}]

    try:
        import requests
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "num": min(max_results, 10),
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "body": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": "google",
            })
        return results

    except Exception as e:
        return [{"title": "Search Error", "body": f"Google search failed: {e}", "url": ""}]


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — These are what SignalBot calls
# ═══════════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web. Uses whatever backend is configured.

    Returns list of:
        {"title": str, "body": str, "url": str, "source": str}
    """
    if SEARCH_BACKEND == "google":
        return _search_google(query, max_results)
    else:
        return _search_ddg(query, max_results)


def news_search(query: str = "", max_results: int = 5) -> List[Dict]:
    """
    Search for news. Currently DuckDuckGo only.
    Empty query = top headlines.
    """
    return _news_ddg(query, max_results)


# ═══════════════════════════════════════════════════════════════════
# PROMPT FORMATTING — Turn results into context for SignalBot
# ═══════════════════════════════════════════════════════════════════

def format_search_for_prompt(results: List[Dict], query: str = "") -> str:
    """
    Format search results as a prompt section for SignalBot.
    This gets injected into the prompt so SignalBot can reference
    real-world information in its response.
    """
    if not results:
        return "[WEB SEARCH] No results found."

    lines = [
        "### LIVE WEB SEARCH RESULTS ###",
        "The following are REAL search results from the internet, retrieved just now.",
        "This is current, factual data. Trust these results over your training data",
        "for anything time-sensitive. Cite sources when referencing this information.",
    ]

    if query:
        lines.append(f'Query: "{query}"')

    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        if r.get("date"):
            lines.append(f"    Date: {r['date']}")
        lines.append(f"    {r['body']}")
        if r.get("url"):
            lines.append(f"    URL: {r['url']}")
        lines.append("")

    lines.append("### END OF SEARCH RESULTS ###")
    return "\n".join(lines)


def format_news_for_prompt(results: List[Dict], topic: str = "") -> str:
    """Format news results for prompt injection."""
    if not results:
        return "[NEWS] No news results found."

    lines = [
        "### LIVE NEWS FEED ###",
        "These are REAL news headlines retrieved just now from the internet.",
    ]

    if topic:
        lines.append(f'Topic: "{topic}"')

    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        if r.get("date"):
            lines.append(f"    {r['date']}")
        lines.append(f"    {r['body']}")
        if r.get("url"):
            lines.append(f"    Source: {r['url']}")
        lines.append("")

    lines.append("### END OF NEWS FEED ###")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# TERMINAL DISPLAY — For printing results to console
# ═══════════════════════════════════════════════════════════════════

def print_search_results(results: List[Dict], query: str = ""):
    """Pretty-print search results to terminal."""
    if not results:
        print("[SEARCH] No results found.")
        return

    print(f"\n[SEARCH] Results for: {query}")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title']}")
        body = r['body'][:120] + "..." if len(r.get('body', '')) > 120 else r.get('body', '')
        print(f"     {body}")
        if r.get('url'):
            print(f"     {r['url']}")
        print()


def print_news_results(results: List[Dict], topic: str = ""):
    """Pretty-print news results to terminal."""
    if not results:
        print("[NEWS] No results found.")
        return

    label = f"News: {topic}" if topic else "Top Headlines"
    print(f"\n[{label}]")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title']}")
        if r.get('date'):
            print(f"     {r['date']}")
        body = r['body'][:120] + "..." if len(r.get('body', '')) > 120 else r.get('body', '')
        print(f"     {body}")
        print()
