#!/usr/bin/env python3
"""
Automated headless browser paper fetcher using Playwright and your active proxy session.

Solves the challenge of modern SSO / publisher gateways:
- OpenAthens & Elsevier / Shibboleth require client-side JavaScript execution
  to complete SAML token exchanges and handle Cloudflare bot challenges.
- Injects cookies from cookies.txt directly into a real Chromium browser context.
- Handles redirects, detects citation_pdf_url and PDF download streams, and saves
  valid PDFs directly into data/papers/.
- Checkpoints progress into data/papers/.browser_proxy_manifest.json.

Usage:
    python fetch_via_browser.py --cookies cookies.txt
    python fetch_via_browser.py --cookies cookies.txt --limit 20
    python fetch_via_browser.py --cookies cookies.txt --proxy-prefix "https://proxy.library.nyu.edu/login?url="
"""

import argparse
import http.cookiejar
import json
import os
import re
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
MAIN_MANIFEST = PAPERS_DIR / ".manifest.json"
BROWSER_MANIFEST = PAPERS_DIR / ".browser_proxy_manifest.json"
PROXY_LOG = PAPERS_DIR / "browser_proxy_fetch_log.tsv"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
DEFAULT_PROXY = "https://proxy.library.nyu.edu/login?url="


def sanitize_filename(name, fallback):
    if not name:
        name = fallback
    name = re.sub(r"[^\w\s.-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:150] or fallback


def build_filename(pmid=None, title=None):
    id_prefix = f"PMID{pmid}" if pmid else "paper"
    name_part = sanitize_filename(title, "") if title else ""
    return f"{id_prefix}_{name_part}.pdf" if name_part else f"{id_prefix}.pdf"


def already_downloaded(pmid: str) -> bool:
    if not PAPERS_DIR.exists():
        return False
    matches = list(PAPERS_DIR.glob(f"PMID{pmid}_*.pdf")) + list(PAPERS_DIR.glob(f"PMID{pmid}.pdf"))
    return any(m.stat().st_size > 1000 for m in matches)


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_json(path: Path, data: dict):
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(path)
    except Exception:
        # Fallback to direct write if atomic replace encounters a lock
        try:
            path.write_text(json.dumps(data, indent=1, sort_keys=True))
        except Exception:
            pass


CRITICAL_COOKIE_NAMES = {
    "__cf_bm": "Cloudflare Bot Clearance",
    "_pk_ses": "NYU Proxy Session",
    "JSESSIONID": "Publisher/IdP Session",
    "oaloginorg": "OpenAthens Login Org",
    "sd_session_id": "ScienceDirect Session",
    "__Host-shib_idp_session": "NYU Shibboleth Session"
}


def get_min_time_to_expiration(cookies_path: Path) -> float | None:
    """Returns minimum seconds until expiration among all active critical cookies, or None if none have expiries."""
    jar = http.cookiejar.MozillaCookieJar(str(cookies_path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception:
        return None

    now = time.time()
    min_remaining = None
    for c in jar:
        if c.expires:
            for crit in CRITICAL_COOKIE_NAMES:
                if crit in c.name:
                    rem = c.expires - now
                    if min_remaining is None or rem < min_remaining:
                        min_remaining = rem
    return min_remaining


def check_cookie_expiration(cookies_path: Path) -> list[str]:
    """Returns a list of descriptions for any critical cookies that have expired."""
    jar = http.cookiejar.MozillaCookieJar(str(cookies_path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception:
        return []

    now = time.time()
    expired = []
    for c in jar:
        if c.expires and c.expires < now:
            for crit, desc in CRITICAL_COOKIE_NAMES.items():
                if crit in c.name:
                    exp_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c.expires))
                    expired.append(f"{c.domain} -> {c.name} ({desc}) expired at {exp_time}")
    return expired


def validate_and_load_cookies(cookies_path: Path):
    if not cookies_path.exists():
        sys.exit(f"Error: Cookies file not found: {cookies_path}")

    jar = http.cookiejar.MozillaCookieJar(str(cookies_path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as e:
        sys.exit(f"Error parsing cookies file: {e}")

    expired = check_cookie_expiration(cookies_path)
    if expired:
        print("\n" + "="*70)
        print("CRITICAL: REQUIRED AUTH/PROXY COOKIES ARE EXPIRED!")
        print("="*70)
        for exp in expired[:5]:
            print(f"  ❌ {exp}")
        print("\nPlease refresh cookies.txt by logging into NYU proxy / publisher")
        print("in your browser and re-exporting cookies.txt before running.")
        print("="*70 + "\n")
        sys.exit(1)

    now = time.time()
    cookies = []
    for c in jar:
        domain = c.domain
        if domain.startswith("."):
            domain = domain[1:]
        
        cookie_dict = {
            "name": c.name,
            "value": c.value,
            "domain": domain,
            "path": c.path,
            "secure": c.secure,
        }
        if c.expires:
            cookie_dict["expires"] = float(c.expires)
            if c.expires < now:
                continue
        cookies.append(cookie_dict)

    return cookies


def fetch_paper_browser(context, proxy_prefix: str, doi: str, dest_path: Path, timeout_sec: int = 30) -> tuple[bool, str]:
    page = context.new_page()
    target = proxy_prefix + f"https://doi.org/{doi}"
    try:
        try:
            page.goto(target, wait_until="load", timeout=timeout_sec * 1000)
        except Exception as e:
            if "Download is starting" not in str(e):
                page.close()
                return False, f"Navigation error: {str(e)[:100]}"

        # Fast abort if proxy rate-limited or forbidden
        if "403 Forbidden" in page.title() or "login?url=" in page.url and "proxy.library.nyu.edu" in page.url:
            page.close()
            return False, "Proxy rejected request (403 Forbidden or Session Expired)"

        pdf_url = None
        for _ in range(25):
            page.wait_for_timeout(400)
            curr_url = page.url.lower()

            if any(gateway in curr_url for gateway in ("institutionlogin", "shibauth", "login.openathens.net", "linkinghub", "action/ssostart")):
                continue

            try:
                html = page.content()
                m = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                if m:
                    pdf_url = m.group(1)
                    break
            except Exception:
                pass

        if not pdf_url:
            url_preview = page.url[:60]
            page.close()
            return False, f"No PDF link or citation_pdf_url found on landing page (ended on {url_preview})"

        # Download via robust in-page JS fetch with credentials (bypasses popup and download hangs)
        import base64
        b64_data = page.evaluate('''async (url) => {
            try {
                const r = await fetch(url, {credentials: 'include'});
                if (!r.ok) return null;
                const buf = await r.arrayBuffer();
                const u8 = new Uint8Array(buf);
                let bstr = '';
                const chunk = 8192;
                for (let i = 0; i < u8.length; i += chunk) {
                    bstr += String.fromCharCode.apply(null, u8.subarray(i, i + chunk));
                }
                return btoa(bstr);
            } catch (e) {
                return null;
            }
        }''', pdf_url)

        page.close()

        if b64_data:
            raw_pdf = base64.b64decode(b64_data)
            if len(raw_pdf) > 1000 and raw_pdf[:4] == b"%PDF":
                dest_path.write_bytes(raw_pdf)
                return True, f"Success ({len(raw_pdf)} bytes)"
            return False, f"Downloaded content not a valid PDF (size={len(raw_pdf)})"

        return False, "In-page PDF fetch failed"

    except Exception as e:
        try:
            page.close()
        except Exception:
            pass
        return False, f"Error: {str(e)[:100]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cookies", default="cookies.txt", help="Path to cookies.txt")
    parser.add_argument("--proxy-prefix", default=DEFAULT_PROXY, help="Institutional proxy URL prefix")
    parser.add_argument("--status", default="no_oa", help="Target status bucket(s) in manifest, comma-separated (e.g. 'no_oa', 'pmc_manual', or 'no_oa,pmc_manual')")
    parser.add_argument("--limit", type=int, default=None, help="Limit papers to attempt")
    parser.add_argument("--delay", type=float, default=0.35, help="Delay in seconds between requests (default: 0.35s)")
    parser.add_argument("--merge-into-main", action="store_true", help="Merge downloaded records into .manifest.json")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle pending papers")
    parser.add_argument("--retry-failed", action="store_true", help="Include papers that already failed in browser proxy attempts")
    args = parser.parse_args()

    main_manifest = load_json(MAIN_MANIFEST)
    browser_manifest = load_json(BROWSER_MANIFEST)

    if args.merge_into_main:
        merged = 0
        for pmid, info in browser_manifest.items():
            if info.get("status") == "downloaded" and pmid in main_manifest:
                main_manifest[pmid]["status"] = "downloaded"
                main_manifest[pmid]["file"] = info.get("file")
                main_manifest[pmid]["updated"] = info.get("updated")
                merged += 1
        save_json(MAIN_MANIFEST, main_manifest)
        print(f"Merged {merged} records into {MAIN_MANIFEST}")
        return

    cookies_path = Path(args.cookies)
    cookies = validate_and_load_cookies(cookies_path)

    # Allowed statuses (comma-separated or 'all')
    target_statuses = set(s.strip() for s in args.status.split(","))

    # Identify pending papers
    pending = []
    failed_papers = []
    for pmid, entry in main_manifest.items():
        if "all" not in target_statuses and entry.get("status") not in target_statuses:
            continue
        doi = entry.get("doi")
        if not doi:
            continue
        if already_downloaded(pmid):
            continue
        if browser_manifest.get(pmid, {}).get("status") == "downloaded":
            continue
        
        item = {
            "pmid": pmid,
            "doi": doi,
            "title": entry.get("title")
        }

        # If it failed previously in browser manifest, don't retry immediately unless requested
        if browser_manifest.get(pmid, {}).get("status") == "failed":
            failed_papers.append(item)
        else:
            pending.append(item)

    # Identify publisher source family for round-robin rotation
    def get_publisher_source(doi: str) -> str:
        doi_lower = doi.lower()
        if "10.1038/" in doi_lower or "nature." in doi_lower:
            return "Nature"
        elif "10.1074/" in doi_lower or "jbc." in doi_lower:
            return "JBC"
        elif "10.1126/" in doi_lower or "science." in doi_lower:
            return "Science"
        elif "10.1093/" in doi_lower:
            return "Oxford"
        elif "10.1002/" in doi_lower:
            return "Wiley"
        elif "10.1099/" in doi_lower:
            return "Microbiology"
        elif any(p in doi_lower for p in ("10.1016/", "10.1006/")):
            return "ScienceDirect"
        return "Other"

    import random
    from collections import defaultdict

    # Group into Combined (High+Medium) vs Low (ScienceDirect/challenging)
    combined_good = [x for x in pending if get_publisher_source(x["doi"]) != "ScienceDirect"]
    challenging = [x for x in pending if get_publisher_source(x["doi"]) == "ScienceDirect"]

    if not args.no_shuffle:
        # Group by publisher family within combined_good
        by_source = defaultdict(list)
        for item in combined_good:
            src = get_publisher_source(item["doi"])
            by_source[src].append(item)

        # Shuffle each publisher bucket individually
        for src in by_source:
            random.shuffle(by_source[src])

        # Round-robin rotation across distinct publishers:
        # Takes one from Nature, then one from JBC, then one from Science, etc.
        interleaved = []
        sources = list(by_source.keys())
        random.shuffle(sources)
        while any(by_source[s] for s in sources):
            for s in sources:
                if by_source[s]:
                    interleaved.append(by_source[s].pop(0))

        random.shuffle(challenging)
        pending = interleaved + challenging
        print(f"Prioritized queue: {len(interleaved)} round-robin rotated across sources ({', '.join(sources)}), {len(challenging)} challenging.")

    # If user explicitly wants to retry failed papers, append them at the very end
    if args.retry_failed:
        if not args.no_shuffle:
            random.shuffle(failed_papers)
        pending.extend(failed_papers)

    print(f"Found {len(pending)} un-attempted papers in status='{args.status}' pending browser download ({len(failed_papers)} previous failures skipped).")
    if args.limit:
        pending = pending[:args.limit]
        print(f"Selected {len(pending)} papers for this batch.")

    if not pending:
        print("Nothing to do.")
        return

    success_count = 0
    fail_count = 0
    consecutive_proxy_errors = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        context.add_cookies(cookies)

        try:
            for idx, item in enumerate(pending, 1):
                # Check cookie expiration before each paper
                expired = check_cookie_expiration(cookies_path)
                if expired:
                    print("\n" + "="*70)
                    print(f"ABORTING RUN AT PAPER [{idx}/{len(pending)}]: COOKIES EXPIRED MID-RUN!")
                    for exp in expired[:3]:
                        print(f"  ❌ {exp}")
                    print("Manifest saved. Please refresh cookies.txt before resuming.")
                    print("="*70 + "\n")
                    break

                pmid = item["pmid"]
                doi = item["doi"]
                title = item["title"]
                filename = build_filename(pmid=pmid, title=title)
                dest_path = PAPERS_DIR / filename

                # Compute remaining time to cookie expiration formatted as [T-XX min]
                rem_sec = get_min_time_to_expiration(cookies_path)
                if rem_sec is not None and rem_sec > 0:
                    rem_min = int(rem_sec // 60)
                    t_str = f"[T-{rem_min:02d} min]"
                elif rem_sec is not None:
                    t_str = "[T-00 min]"
                else:
                    t_str = "[T-?? min]"

                print(f"[{idx}/{len(pending)}] {t_str} PMID:{pmid} (DOI:{doi})...", end=" ", flush=True)
                ok, msg = fetch_paper_browser(context, args.proxy_prefix, doi, dest_path)
                
                status_str = "downloaded" if ok else "failed"
                print("SUCCESS" if ok else f"FAILED: {msg}", flush=True)

                if ok:
                    success_count += 1
                    consecutive_proxy_errors = 0
                    browser_manifest[pmid] = {
                        "status": "downloaded",
                        "doi": doi,
                        "file": filename,
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                else:
                    fail_count += 1
                    browser_manifest[pmid] = {
                        "status": "failed",
                        "doi": doi,
                        "error": msg,
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                    if "Proxy rejected request" in msg or "403 Forbidden" in msg:
                        consecutive_proxy_errors += 1
                    else:
                        consecutive_proxy_errors = 0

                # Append log
                with open(PROXY_LOG, "a") as log_f:
                    log_f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{pmid}\t{doi}\t{status_str}\t{msg}\n")

                if idx % 5 == 0:
                    save_json(BROWSER_MANIFEST, browser_manifest)

                # Early-abort if 3 consecutive requests get rejected by proxy
                if consecutive_proxy_errors >= 3:
                    print("\n" + "="*70)
                    print("ABORTING RUN: 3 consecutive requests were rejected by the proxy (403 Forbidden or Session Expired).")
                    print("Please refresh your browser session and export fresh cookies.txt.")
                    print("="*70)
                    break

                if args.delay > 0:
                    time.sleep(args.delay)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving manifest...")
            save_json(BROWSER_MANIFEST, browser_manifest)
            # Auto-merge downloaded items into fresh main manifest
            try:
                cur_main = load_json(MAIN_MANIFEST)
                for pmid, info in browser_manifest.items():
                    if info.get("status") == "downloaded" and pmid in cur_main:
                        cur_main[pmid]["status"] = "downloaded"
                        cur_main[pmid]["file"] = info.get("file")
                        cur_main[pmid]["updated"] = info.get("updated")
                save_json(MAIN_MANIFEST, cur_main)
            except Exception as e:
                print(f"Warning: could not merge into main manifest: {e}")
            import os
            os._exit(0)
        finally:
            save_json(BROWSER_MANIFEST, browser_manifest)
            # Auto-merge downloaded items into fresh main manifest
            try:
                cur_main = load_json(MAIN_MANIFEST)
                for pmid, info in browser_manifest.items():
                    if info.get("status") == "downloaded" and pmid in cur_main:
                        cur_main[pmid]["status"] = "downloaded"
                        cur_main[pmid]["file"] = info.get("file")
                        cur_main[pmid]["updated"] = info.get("updated")
                save_json(MAIN_MANIFEST, cur_main)
            except Exception as e:
                print(f"Warning: could not merge into main manifest: {e}")
            try:
                browser.close()
            except Exception:
                pass

    print(f"\nFinished run. Downloaded: {success_count}, Failed: {fail_count}")
    print(f"Automatically synced all downloaded records into {MAIN_MANIFEST}")


if __name__ == "__main__":
    main()
