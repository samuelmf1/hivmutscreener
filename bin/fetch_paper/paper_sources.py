"""
Shared, dependency-free logic for resolving paper identifiers (PMID/DOI/
PMCID/URL) and locating legitimate open-access PDF copies. Used by both
fetch_paper.py (single paper, interactive) and batch_fetch_papers.py (bulk).

Only ever returns PDFs from sources that explicitly flag content as open
access (Europe PMC's "OA" availabilityCode, Unpaywall's OA locations). No
paywall bypassing or anti-bot circumvention.
"""

import json
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlparse

USER_AGENT = "Mozilla/5.0 (compatible; PaperFetcher/1.0)"
IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={ids}&format=json"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&resultType=core"
EUROPEPMC_GET_PDF_URL = "https://europepmc.org/api/getPdf?pmcid={pmcid}"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&format=json"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}?email={email}"
CROSSREF_URL = "https://api.crossref.org/works/{doi}"
# Fallback DOI resolver for when NCBI's own ID Converter has no mapping --
# empirically, noticeably more complete: every PMID sampled where IDCONV
# came back with doi=None, OpenAlex still had a DOI for. Matters a lot here
# since every source below except CORE's pubmedId branch is DOI-gated.
OPENALEX_URL = "https://api.openalex.org/works/pmid:{pmid}"

# Additional, longer-tail OA sources -- only worth querying if the cheap/
# authoritative ones above (PMC Open Data, Europe PMC, Unpaywall) came up
# empty, see the short-circuit in find_oa_pdf_candidates().
# id goes directly in the path as e.g. "DOI:10.1234/x" or "PMID:12345", no
# extra wrapping -- https://api.semanticscholar.org/api-docs/graph
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/{external_id}?fields=openAccessPdf"
# v2 was retired; v3 works unauthenticated but capped at 10 req/min (headers
# confirm this), a free key at core.ac.uk/services/api raises that a lot.
# Trailing slash before the query string is required -- CORE 301s without it.
CORE_API_URL = "https://api.core.ac.uk/v3/search/works/?q={query}"
CORE_API_KEY = os.environ.get("CORE_API_KEY")
DOAJ_URL = "https://doaj.org/api/search/articles/{query}"
ARXIV_URL = "https://export.arxiv.org/api/query?search_query=doi:{doi}&max_results=1"

# NCBI's own sanctioned bulk-retrieval channel for PMC content licensed for
# redistribution (the "PMC Article Datasets" on AWS Open Data) -- anonymous,
# no bot-check, distinct from the gated website. Replaced the legacy FTP
# service in August 2026.
PMC_OPENDATA_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
PMC_OPENDATA_LIST_URL = PMC_OPENDATA_BUCKET + "?list-type=2&prefix={prefix}"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

IDCONV_BATCH_SIZE = 200  # NCBI's documented max ids per ID Converter call

# Hosts that serve free content directly (govt./repository sites). Routing
# these through an institutional EZproxy is pointless -- they aren't in its
# licensed-domain list, so the proxy just loops back to a login page instead
# of ever reaching the content.
FREE_CONTENT_HOSTS = (
    "ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "europepmc.org",
    "ebi.ac.uk",
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "doaj.org",
    "plos.org",
    "ncbi.gov",
    "core.ac.uk",
    "osf.io",
    "figshare.com",
    # NOT researchgate.net / academia.edu: those host uploads without
    # verifying the uploader had rights to redistribute, unlike the
    # repositories above -- doesn't fit this module's "explicitly flagged as
    # OA" policy (see module docstring), so don't auto-trust their hostname.
)


def http_get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _get_redirect_location(url, timeout=15):
    """HEAD a URL without following its redirect; return the Location header
    it points to, or None. Only ever reads response headers, never a body --
    used to read PMC's own redirect chain, not to fetch gated content."""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    try:
        opener.open(req, timeout=timeout)
        return None
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers.get("Location")
        return None
    except urllib.error.URLError:
        return None


def resolve_pmc_filename(pmcid):
    """PMC's /pdf/ endpoint 301-redirects (twice) to the article's real PDF
    filename before you ever hit the bot-check page that gates the actual
    file -- so this reads that filename off the redirect headers alone,
    without touching the gated response. Returns None if anything's off."""
    hop1 = _get_redirect_location(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/")
    if not hop1:
        return None
    hop2 = _get_redirect_location(hop1)
    if not hop2:
        return None
    name = hop2.rsplit("/", 1)[-1]
    return name or None


def http_get_json(url, **kw):
    data, _ = http_get(url, **kw)
    return json.loads(data)


def parse_identifier(raw):
    """Return (kind, value) where kind is one of pmid/doi/pmcid/url."""
    raw = raw.strip()

    doi_match = re.search(r"10\.\d{4,9}/[^\s\"'>]+", raw)

    if raw.lower().startswith("http"):
        parsed = urlparse(raw)
        host = parsed.netloc.lower()
        if "doi.org" in host and doi_match:
            return "doi", doi_match.group(0)
        pmc_match = re.search(r"PMC\d+", raw, re.IGNORECASE)
        if pmc_match:
            return "pmcid", pmc_match.group(0).upper()
        pubmed_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", raw)
        if pubmed_match:
            return "pmid", pubmed_match.group(1)
        if doi_match:
            return "doi", doi_match.group(0)
        return "url", raw

    if re.fullmatch(r"PMC\d+", raw, re.IGNORECASE):
        return "pmcid", raw.upper()
    if re.fullmatch(r"\d{1,9}", raw):
        return "pmid", raw
    if doi_match:
        return "doi", doi_match.group(0)

    raise ValueError(f"Could not interpret '{raw}' as a PMID, DOI, PMCID, or URL")


def resolve_ids(kind, value):
    """Use NCBI's ID Converter to fill in pmid/pmcid/doi from whichever we have."""
    result = {"pmid": None, "pmcid": None, "doi": None}
    if kind == "url":
        return result
    result[kind] = value
    try:
        data = http_get_json(IDCONV_URL.format(ids=quote(value)))
        records = data.get("records", [])
        if records:
            rec = records[0]
            result["pmid"] = rec.get("pmid") or result["pmid"]
            result["pmcid"] = rec.get("pmcid") or result["pmcid"]
            result["doi"] = rec.get("doi") or result["doi"]
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
        pass
    return result


def resolve_doi_openalex(pmid):
    """Single-PMID DOI lookup via OpenAlex, for when NCBI's ID Converter
    doesn't have one (see OPENALEX_URL comment)."""
    try:
        data = http_get_json(OPENALEX_URL.format(pmid=pmid))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
        return None
    doi_url = data.get("doi") or ""
    return doi_url.rsplit("doi.org/", 1)[-1] if doi_url else None


def resolve_ids_batch(pmids, openalex_fallback=True):
    """Resolve many PMIDs to {pmid, pmcid, doi} in chunks of IDCONV_BATCH_SIZE.
    Returns {pmid: {"pmid":.., "pmcid":.., "doi":..}}; a PMID NCBI has no
    record for at all is simply absent rather than raising, but one NCBI
    knows about minus a DOI still gets an OpenAlex fallback attempt (one
    request each, so this is the slow part for a PMID-heavy batch -- pass
    openalex_fallback=False to skip it if that's not worth the time)."""
    out = {}
    pmids = list(pmids)
    for i in range(0, len(pmids), IDCONV_BATCH_SIZE):
        chunk = pmids[i:i + IDCONV_BATCH_SIZE]
        try:
            data = http_get_json(IDCONV_URL.format(ids=quote(",".join(chunk))))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
            continue
        for rec in data.get("records", []):
            pmid = rec.get("pmid")
            if not pmid:
                continue
            out[str(pmid)] = {
                "pmid": pmid,
                "pmcid": rec.get("pmcid"),
                "doi": rec.get("doi"),
            }

    if openalex_fallback:
        for pmid in pmids:
            entry = out.get(pmid)
            if entry and entry.get("doi"):
                continue
            doi = resolve_doi_openalex(pmid)
            if not doi:
                continue
            if entry:
                entry["doi"] = doi
            else:
                out[pmid] = {"pmid": pmid, "pmcid": None, "doi": doi}

    return out


def get_title(pmid=None, doi=None):
    try:
        if pmid:
            data = http_get_json(ESUMMARY_URL.format(pmid=pmid))
            return data["result"][str(pmid)]["title"]
        if doi:
            data = http_get_json(CROSSREF_URL.format(doi=quote(doi)))
            return data["message"]["title"][0]
    except Exception:
        pass
    return None


def find_pmc_opendata_pdfs(pmcid):
    """NCBI's PMC Article Datasets on AWS Open Data (s3://pmc-oa-opendata):
    an anonymous, sanctioned bulk-retrieval bucket for PMC content licensed
    for redistribution -- separate from the bot-gated website, so nothing
    here needs to touch that gate at all. Lists the bucket for this PMCID's
    version prefix(es) and returns any article PDF object URLs found."""
    if not pmcid:
        return []
    try:
        data, _ = http_get(PMC_OPENDATA_LIST_URL.format(prefix=quote(f"{pmcid}.")))
    except (urllib.error.URLError, urllib.error.HTTPError):
        return []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []
    candidates = []
    key_pattern = re.compile(rf"^{re.escape(pmcid)}\.\d+/{re.escape(pmcid)}\.\d+\.pdf$")
    for contents in root.findall("s3:Contents", S3_NS):
        key_el = contents.find("s3:Key", S3_NS)
        if key_el is not None and key_el.text and key_pattern.match(key_el.text):
            candidates.append(PMC_OPENDATA_BUCKET + key_el.text)
    return candidates


def find_europepmc_pdfs(pmid=None, pmcid=None, doi=None):
    """Europe PMC's REST API flags open-access full text explicitly
    (availabilityCode "OA"), so this only ever returns legally free copies."""
    if pmcid:
        query = f"PMCID:{pmcid}"
    elif pmid:
        query = f"EXT_ID:{pmid} AND SRC:MED"
    elif doi:
        query = f'DOI:"{doi}"'
    else:
        return []
    try:
        data = http_get_json(EUROPEPMC_SEARCH_URL.format(query=quote(query)))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
        return []
    results = data.get("resultList", {}).get("result", [])
    if not results:
        return []
    urls = results[0].get("fullTextUrlList", {}).get("fullTextUrl", [])
    return [
        u["url"] for u in urls
        if u.get("availabilityCode") == "OA" and u.get("documentStyle") == "pdf"
    ]


def find_unpaywall_pdfs(doi, email):
    """Return (candidate_pdf_urls, landing_url). Tries every OA location
    Unpaywall knows about, not just best_oa_location, since the "best" one
    doesn't always expose a direct PDF link."""
    try:
        data = http_get_json(UNPAYWALL_URL.format(doi=quote(doi), email=quote(email)))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
        return [], None
    locations = []
    if data.get("best_oa_location"):
        locations.append(data["best_oa_location"])
    locations.extend(data.get("oa_locations", []))
    candidates = []
    for loc in locations:
        for key in ("url_for_pdf", "url"):
            url = loc.get(key)
            if url and url not in candidates:
                candidates.append(url)
    landing = data.get("doi_url") or f"https://doi.org/{doi}"
    return candidates, landing


def find_semantic_scholar_pdfs(doi):
    """Search Semantic Scholar for an OA PDF by DOI. One request: the
    external-id lookup already returns full paper fields directly, no need
    for a second call by internal paperId."""
    if not doi:
        return []
    try:
        url = SEMANTIC_SCHOLAR_URL.format(external_id=quote(f"DOI:{doi}", safe=":/"))
        data = http_get_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
        return []
    oa_pdf = data.get("openAccessPdf")
    url_link = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None
    return [url_link] if url_link else []


def _core_search(query):
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"} if CORE_API_KEY else {}
    try:
        data = http_get_json(CORE_API_URL.format(query=quote(query)), headers=headers)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
        return []
    candidates = []
    for work in data.get("results", []):
        if work.get("downloadUrl"):
            candidates.append(work["downloadUrl"])
        candidates.extend(work.get("sourceFulltextUrls") or [])
    return candidates


def find_core_ac_pdfs(doi=None, pmid=None):
    """Search CORE (aggregates repository-hosted copies) for a PDF by DOI or
    PMID. Works unauthenticated but rate-limited to 10 req/min; set
    CORE_API_KEY (free signup at core.ac.uk/services/api) for real bulk use."""
    candidates = []
    if doi:
        candidates.extend(_core_search(f"doi:{doi}"))
    if pmid:
        candidates.extend(c for c in _core_search(f"pubmedId:{pmid}") if c not in candidates)
    return candidates


def find_doaj_pdfs(doi):
    """Search DOAJ (Directory of Open Access Journals -- fully-OA journals
    only, so any fulltext link found here is safe to treat as OA)."""
    if not doi:
        return []
    try:
        url = DOAJ_URL.format(query=quote(f"doi:{doi}", safe=""))
        data = http_get_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
        return []
    urls = []
    for article in data.get("results", []):
        for link in article.get("bibjson", {}).get("link", []):
            if link.get("type") == "fulltext" and link.get("url"):
                urls.append(link["url"])
    return urls


def find_arxiv_pdfs(doi):
    """Search arXiv for papers by DOI and get PDF URLs."""
    if not doi:
        return []
    try:
        url = ARXIV_URL.format(doi=quote(doi))
        data, _ = http_get(url)
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        pdfs = []
        for entry in entries:
            for link in entry.findall("atom:link", ns):
                rel = link.get("rel")
                href = link.get("href")
                if rel == "related" and "pdf" in href.lower():
                    pdfs.append(href)
                if rel == "alternate" and href.endswith(".pdf"):
                    pdfs.append(href)
        return pdfs
    except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError):
        return []


def download_pdf(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()
    if not body.startswith(b"%PDF") and "pdf" not in content_type.lower():
        return False
    with open(dest_path, "wb") as f:
        f.write(body)
    return True


def sanitize_filename(name, fallback):
    if not name:
        name = fallback
    name = re.sub(r"[^\w\s.-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:150] or fallback


def build_filename(pmid=None, pmcid=None, doi=None, title=None):
    if pmid:
        id_prefix = f"PMID{pmid}"
    elif pmcid:
        id_prefix = pmcid
    elif doi:
        id_prefix = f"DOI{doi.replace('/', '_')}"
    else:
        id_prefix = "paper"
    name_part = sanitize_filename(title, "") if title else ""
    return f"{id_prefix}_{name_part}.pdf" if name_part else f"{id_prefix}.pdf"


def is_free_content_host(url):
    host = urlparse(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return any(host == h or host.endswith("." + h) for h in FREE_CONTENT_HOSTS)


def find_oa_pdf_candidates(pmid=None, pmcid=None, doi=None, email=None):
    """Try every automatic OA source in order. Returns (candidates, landing_url)."""
    candidates = []
    landing_url = None

    # 1. PMC Open Data & Europe PMC PDF endpoint (highest priority for PMC content)
    if pmcid:
        opendata_candidates = find_pmc_opendata_pdfs(pmcid)
        candidates.extend(c for c in opendata_candidates if c not in candidates)
        # Direct Europe PMC rendered PDF endpoint (official open mirror for PMC content)
        epmc_direct = EUROPEPMC_GET_PDF_URL.format(pmcid=pmcid)
        if epmc_direct not in candidates:
            candidates.append(epmc_direct)
        landing_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

    # 2. Europe PMC (OA papers only)
    if pmid or pmcid or doi:
        epmc_candidates = find_europepmc_pdfs(pmid=pmid, pmcid=pmcid, doi=doi)
        candidates.extend(c for c in epmc_candidates if c not in candidates)
        if pmcid:
            landing_url = landing_url or f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

    # 3. Unpaywall (legitimate OA from repositories)
    if doi and email:
        unpaywall_candidates, unpaywall_landing = find_unpaywall_pdfs(doi, email)
        candidates.extend(c for c in unpaywall_candidates if c not in candidates)
        landing_url = landing_url or unpaywall_landing

    # 4-7: longer-tail sources, several of them tightly rate-limited
    # (Semantic Scholar, CORE unauthenticated) -- only worth spending that
    # budget on papers the sources above didn't already resolve.
    if not candidates and doi:
        candidates.extend(c for c in find_semantic_scholar_pdfs(doi) if c not in candidates)
    if not candidates and (doi or pmid):
        candidates.extend(c for c in find_core_ac_pdfs(doi=doi, pmid=pmid) if c not in candidates)
    if not candidates and doi:
        candidates.extend(c for c in find_doaj_pdfs(doi) if c not in candidates)
    if not candidates and doi:
        candidates.extend(c for c in find_arxiv_pdfs(doi) if c not in candidates)

    if not landing_url:
        landing_url = f"https://doi.org/{doi}" if doi else None

    return candidates, landing_url
