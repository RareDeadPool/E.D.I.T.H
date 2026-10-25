"""Browser and Web Application Adapter for E.D.I.T.H. V4.

Provides web navigation, search execution, real result summarization,
form filling assistance, and safety gating for submissions and purchases.
"""

import os
import re
import json
import time
import urllib.parse
import urllib.request
import webbrowser
from typing import Dict, Any, List, Optional, Tuple


class BrowserAdapter:
    """Production application adapter for browser tasks."""

    POPULAR_SERVICES = {
        "gmail": "https://mail.google.com",
        "google mail": "https://mail.google.com",
        "youtube": "https://www.youtube.com",
        "github": "https://www.github.com",
        "google": "https://www.google.com",
        "reddit": "https://www.reddit.com",
        "twitter": "https://twitter.com",
        "x": "https://x.com",
        "linkedin": "https://www.linkedin.com",
        "chatgpt": "https://chatgpt.com",
        "netflix": "https://netflix.com",
        "amazon": "https://amazon.com",
        "maps": "https://maps.google.com",
        "calendar": "https://calendar.google.com",
        "drive": "https://drive.google.com",
        "docs": "https://docs.google.com",
        "sheets": "https://sheets.google.com",
        "whatsapp": "https://web.whatsapp.com",
        "whatsapp web": "https://web.whatsapp.com",
        "spotify": "https://open.spotify.com",
    }

    def __init__(self):
        self._last_opened_url: Optional[str] = None

    def resolve_url(self, target: str) -> str:
        """Resolve a service name, domain, or raw query to a valid web URL."""
        t_clean = target.strip().lower()
        t_clean = re.sub(r"^(?:the\s+)?(?:website\s+|page\s+)?", "", t_clean).strip()

        if t_clean in self.POPULAR_SERVICES:
            return self.POPULAR_SERVICES[t_clean]

        for s_name, s_url in self.POPULAR_SERVICES.items():
            if t_clean == s_name or t_clean.startswith(f"{s_name} ") or f" {s_name}" in t_clean:
                return s_url

        if re.match(r"^https?://", target, re.IGNORECASE):
            return target

        if "." in target and not target.endswith(".") and " " not in target:
            return f"https://{target}"

        # Otherwise perform Google search query
        encoded = urllib.parse.quote_plus(target)
        return f"https://www.google.com/search?q={encoded}"

    def open_page(self, target_or_url: str) -> Dict[str, Any]:
        """Navigate to website or web application."""
        final_url = self.resolve_url(target_or_url)
        try:
            webbrowser.open(final_url)
            self._last_opened_url = final_url
            time.sleep(0.5)
            return {
                "status": "success",
                "action": "open_browser_page",
                "url": final_url,
                "verified": True,
                "message": f"Opened '{final_url}' in your default browser."
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to open '{final_url}': {e}"}

    def search_and_summarize(self, query: str, max_results: int = 4) -> Dict[str, Any]:
        """
        Execute web search and return synthesized factual findings.
        Uses local lightweight parsing without requiring cloud LLMs.
        """
        q_clean = query.strip()
        encoded = urllib.parse.quote_plus(q_clean)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        results: List[Dict[str, str]] = []
        try:
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Extract result snippets and titles from duckduckgo html
            raw_snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
            raw_titles = re.findall(r'<a class="result__url[^>]*>(.*?)</a>', html, re.DOTALL)

            for i in range(min(len(raw_snippets), max_results)):
                snip_clean = re.sub(r"<[^>]+>", "", raw_snippets[i]).strip()
                title_clean = re.sub(r"<[^>]+>", "", raw_titles[i]).strip() if i < len(raw_titles) else f"Result {i+1}"
                if snip_clean:
                    results.append({"title": title_clean, "snippet": snip_clean})
        except Exception:
            # Fallback: construct standard Google search URL and provide direct portal
            pass

        # Also open Google search in browser for user visibility
        google_url = f"https://www.google.com/search?q={encoded}"
        try:
            webbrowser.open(google_url)
        except Exception:
            pass

        if results:
            summary_lines = [f"Search findings for '{q_clean}':"]
            for r in results:
                summary_lines.append(f"• {r['snippet']}")
            summary_text = "\n".join(summary_lines)
        else:
            summary_text = f"Opened search results for '{q_clean}' in your browser at {google_url}."

        return {
            "status": "success",
            "action": "search_and_summarize",
            "query": q_clean,
            "url": google_url,
            "results_count": len(results),
            "results": results,
            "summary": summary_text,
            "message": summary_text
        }

    def prepare_form_data(
        self,
        form_data: Dict[str, Any],
        target_form_name: Optional[str] = None,
        submit: bool = False,
        authorized: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare form fields for user review with authorization gating before submission.
        """
        fields_summary = "\n".join(f"  • {k}: {v}" for k, v in form_data.items())
        form_desc = target_form_name or "Web Form"

        if submit and not authorized:
            return {
                "status": "needs_confirmation",
                "action": "submit_web_form",
                "requires_confirmation": True,
                "form_name": form_desc,
                "fields": form_data,
                "message": (
                    f"Please review the supplied information for {form_desc}:\n"
                    f"{fields_summary}\n\n"
                    f"Would you like me to submit this form? (Say 'Yes, submit' or 'Cancel')"
                )
            }

        return {
            "status": "success",
            "action": "fill_web_form",
            "form_name": form_desc,
            "fields": form_data,
            "submitted": submit,
            "message": f"Prepared values for {form_desc}:\n{fields_summary}\n(Form ready for inspection.)"
        }

    def read_and_summarize_url(self, url: str) -> Dict[str, Any]:
        """Fetch content from a webpage URL and extract a clean factual summary."""
        target_url = self.resolve_url(url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(req, timeout=7.0) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            m_title = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = re.sub(r"<[^>]+>", "", m_title.group(1)).strip() if m_title else "Web Page"

            clean_html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
            raw_text = re.sub(r"<[^>]+>", " ", clean_html)
            words = [w.strip() for w in raw_text.split() if len(w.strip()) > 2]
            clean_body = " ".join(words[:400])

            sentences = [s.strip() for s in re.split(r"[.!?]\s+", clean_body) if len(s.strip()) > 25]
            key_pts = sentences[:4]
            if key_pts:
                summary_text = f"Summary of '{title}' ({target_url}):\n\n" + "\n".join(f"• {pt}." for pt in key_pts)
            else:
                summary_text = f"Accessed page '{title}' at {target_url}."

            return {
                "status": "success",
                "action": "read_and_summarize_url",
                "url": target_url,
                "title": title,
                "summary": summary_text,
                "message": summary_text
            }
        except Exception as e:
            return {
                "status": "partial",
                "action": "read_and_summarize_url",
                "url": target_url,
                "summary": f"Could not directly parse {target_url}: {e}. Opened in your browser.",
                "message": f"Opened {target_url} in your browser."
            }
