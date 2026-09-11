#!/usr/bin/env python3
"""Fetch paywalled papers (the "no_oa" bucket in data/papers/.manifest.json)
through an institutional proxy/SSO service, using a cookies.txt exported
from a browser session where you've already logged in.

How it works per paper:
  1. Resolve a DOI for the PMID (manifest's own doi field, else OpenAlex).
  2. GET the DOI resolver link through the proxy prefix, using your cookies.
  3. If that response is already a PDF, save it.
  4. Otherwise parse the landing-page HTML for a `citation_pdf_url` meta tag
     (the standard tag publishers embed for Google Scholar indexing --
     present on nearly all major journal platforms) and fetch that, also
     through the proxy.

Getting cookies.txt: NYU migrated off classic EZProxy to OpenAthens, which
works differently -- rather than one proxy domain rewriting every URL (so
one domain's cookies covered everything), OpenAthens does a SAML handshake
through go.openathens.net that lands you authenticated directly on each
*publisher's own* domain. A cookie export scoped to a single domain likely
won't be enough. So: log into a few representative resources through NYU's
catalog first (so you pick up cookies for go.openathens.net *and* a couple
publisher domains -- e.g. sciencedirect.com, springer.com), then export
ALL cookies (not just current-tab-domain) with a browser extension like
"Get cookies.txt LOCALLY", and copy the file here (e.g.
`scp cookies.txt <this-host>:~/`). If a fetch fails with "still on a proxy
login page", that specific publisher domain's cookie is likely missing or
expired -- visit that one article manually once, re-export, and retry.

Resumable: a PMID already having a PDF on disk (data/papers/PMID<id>*.pdf)
is skipped. Successes update data/papers/.manifest.json to status
"downloaded" like the rest of the pipeline; every attempt (success or not)
is appended to data/papers/proxy_fetch_log.tsv for follow-up.

Usage:
    python fetch_via_proxy.py --cookies ~/cookies.txt
    python fetch_via_proxy.py --cookies ~/cookies.txt --limit 20   # test on a few first
    # override if you're using a different institution's proxy:
    python fetch_via_proxy.py --cookies ~/cookies.txt --proxy-prefix "https://proxy.library.rutgers.edu/login?url="
"""

import argparse
import http.cookiejar
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from paper_sources import USER_AGENT, build_filename, resolve_doi_openalex  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
FETCH_MANIFEST = PAPERS_DIR / ".manifest.json"
PROXY_LOG = PAPERS_DIR / "proxy_fetch_log.tsv"

CITATION_PDF_RE = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def load_manifest() -> dict:
    if not FETCH_MANIFEST.exists():
        sys.exit(f"missing {FETCH_MANIFEST}")
    return json.loads(FETCH_MANIFEST.read_text())


def save_manifest(manifest: dict):
    tmp = FETCH_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    tmp.replace(FETCH_MANIFEST)


def already_have_pdf(pmid: str) -> bool:
    return any(PAPERS_DIR.glob(f"PMID{pmid}_*.pdf")) or (PAPERS_DIR / f"PMID{pmid}.pdf").exists()


def proxied_get(opener, proxy_prefix: str, url: str, timeout: int = 60) -> tuple[bytes, str]:
    """GET a URL through the proxy prefix; returns (body, content_type)."""
    proxied_url = proxy_prefix + url
    req = urllib.request.Request(proxied_url, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("Content-Type", "")


def fetch_one(opener, proxy_prefix: str, pmid: str, doi: str, dest_path: Path) -> tuple[bool, str]:
    target = f"https://doi.org/{doi}"
    try:
        body, content_type = proxied_get(opener, proxy_prefix, target)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return False, f"landing page fetch failed: {e}"

    if body.startswith(b"%PDF") or "pdf" in content_type.lower():
        dest_path.write_bytes(body)
        return True, "ok (direct PDF)"

    html = body.decode("utf-8", errors="ignore")
    m = CITATION_PDF_RE.search(html)
    if not m:
        if "proxy" in html.lower() and "login" in html.lower():
            return False, "still on a proxy login page -- cookies likely expired, re-export them"
        return False, "no citation_pdf_url meta tag found on landing page"

    pdf_url = m.group(1)
    try:
        body, content_type = proxied_get(opener, proxy_prefix, pdf_url)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return False, f"citation_pdf_url fetch failed: {e}"

    if not (body.startswith(b"%PDF") or "pdf" in content_type.lower()):
        return False, "citation_pdf_url did not return a PDF (paywall/login wall past this point?)"

    dest_path.write_bytes(body)
    return True, "ok (via citation_pdf_url)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cookies", required=True, type=Path, help="cookies.txt exported from your logged-in proxy session")
    parser.add_argument("--proxy-prefix", default="https://go.openathens.net/redirector/nyu.edu?url=",
                         help="Proxy/SSO redirector URL prefix (default: NYU's OpenAthens redirector). "
                              "Override if using a different institution's proxy, or a classic EZProxy "
                              'link of the form "https://proxy.<school>.edu/login?url=".')
    parser.add_argument("--status", default="no_oa", help="which fetch-manifest status bucket to target (default no_oa)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between papers (politeness)")
    parser.add_argument("--limit", type=int, default=None, help="only attempt the first N pending papers (for testing)")
    args = parser.parse_args()

    if not args.cookies.exists():
        sys.exit(f"cookies file not found: {args.cookies}")

    jar = http.cookiejar.MozillaCookieJar(str(args.cookies))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as e:
        sys.exit(f"couldn't parse cookies file (expected Netscape cookies.txt format): {e}")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    manifest = load_manifest()
    pending = [pmid for pmid, v in manifest.items()
               if v.get("status") == args.status and not already_have_pdf(pmid)]
    if args.limit:
        pending = pending[:args.limit]
    log(f"{len(pending)} paper(s) in status={args.status!r} still need a PDF")

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    log_is_new = not PROXY_LOG.exists()
    with open(PROXY_LOG, "a") as logf:
        if log_is_new:
            logf.write("timestamp\tpmid\tdoi\tresult\tdetail\n")

        ok = failed = 0
        for i, pmid in enumerate(pending, 1):
            entry = manifest[pmid]
            doi = entry.get("doi") or resolve_doi_openalex(pmid)
            timestamp = datetime.now().isoformat(timespec="seconds")

            if not doi:
                logf.write(f"{timestamp}\t{pmid}\t\tfailed\tno DOI resolvable\n")
                failed += 1
            else:
                filename = build_filename(pmid=pmid, doi=doi, title=entry.get("title"))
                dest_path = PAPERS_DIR / filename
                success, detail = fetch_one(opener, args.proxy_prefix, pmid, doi, dest_path)
                logf.write(f"{timestamp}\t{pmid}\t{doi}\t{'ok' if success else 'failed'}\t{detail}\n")
                if success:
                    ok += 1
                    manifest[pmid] = {**entry, "status": "downloaded", "file": filename,
                                       "doi": doi, "updated": timestamp}
                else:
                    failed += 1
                    log(f"  FAILED {pmid}: {detail}")

            logf.flush()
            if i % 10 == 0 or i == len(pending):
                log(f"  progress: {i}/{len(pending)} ({ok} ok, {failed} failed)")
                save_manifest(manifest)  # checkpoint periodically, not just at the end
            time.sleep(args.delay)

    save_manifest(manifest)
    log(f"Done. {ok} downloaded, {failed} failed (see {PROXY_LOG} for details, re-run to retry failures).")


if __name__ == "__main__":
    main()
