"""
Shared, dependency-free logic for resolving paper identifiers (PMID/DOI/
PMCID/URL) and locating legitimate open-access PDF copies. Used by both
fetch_paper.py (single paper, interactive) and batch_fetch_papers.py (bulk).

Only ever returns PDFs from sources that explicitly flag content as open
access (Europe PMC's "OA" availabilityCode, Unpaywall's OA locations). No
paywall bypassing or anti-bot circumvention.
"""

import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlparse

USER_AGENT = "Mozilla/5.0 (compatible; PaperFetcher/1.0)"
IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={ids}&format=json"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&resultType=core"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&format=json"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}?email={email}"
CROSSREF_URL = "https://api.crossref.org/works/{doi}"

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


def resolve_ids_batch(pmids):
    """Resolve many PMIDs to {pmid, pmcid, doi} in chunks of IDCONV_BATCH_SIZE.
    Returns {pmid: {"pmid":.., "pmcid":.., "doi":..}}; missing/unresolved
    PMIDs are simply absent from the result rather than raising."""
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

    if pmcid:
        opendata_candidates = find_pmc_opendata_pdfs(pmcid)
        candidates.extend(c for c in opendata_candidates if c not in candidates)
        landing_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

    if pmid or pmcid or doi:
        epmc_candidates = find_europepmc_pdfs(pmid=pmid, pmcid=pmcid, doi=doi)
        candidates.extend(c for c in epmc_candidates if c not in candidates)
        if pmcid:
            landing_url = landing_url or f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

    if doi and email:
        unpaywall_candidates, unpaywall_landing = find_unpaywall_pdfs(doi, email)
        candidates.extend(c for c in unpaywall_candidates if c not in candidates)
        landing_url = landing_url or unpaywall_landing

    if not landing_url:
        landing_url = f"https://doi.org/{doi}" if doi else None

    return candidates, landing_url
