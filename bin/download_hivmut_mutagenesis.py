#!/usr/bin/env python3
"""
Download per-residue mutagenesis/polymorphism annotation tables from the HIV
Mutation Browser (hivmut.org) for all nine HIV-1 proteins and combine them
into one table.

Data source (per residue, per protein):
    https://hivmut.org/php/download_excel.php?residue={N}&uniprot={ACC}&filetype=csv

That endpoint returns a small tab-delimited body (despite filetype=csv) with
one row per (mutation, paper): adjust_mut, pubmed_id, title, authors, journal.
Not every residue has data; residues with nothing mapped return an empty or
"Error ..." body and are simply skipped.

Usage:
    python download_hivmut_mutagenesis.py --out hivmut_mutagenesis_combined.tsv

    # Just re-run one gene (e.g. after fixing a bug), reusing the rest of the cache:
    python download_hivmut_mutagenesis.py --out hivmut_mutagenesis_combined.tsv --only env

Notes:
    - Every raw HTTP response is cached under --cache-dir, one file per
      (uniprot, residue). A killed/interrupted run resumes for free: rerun the
      same command and it only fetches what's missing.
    - hivmut.org is a small academic server; keep --delay at a courteous value
      (default 0.6s). A full run across all nine proteins is ~3,100 requests,
      so budget ~30-45 minutes.
"""
import argparse
import csv
import html
import os
import random
import sys
import time

import requests

# (gene tab name on hivmut.org, UniProt accession, first residue, last residue)
# Ranges are as numbered on the protein's own "Sequence" tab (this is what the
# `residue` query parameter indexes), not HXB2 genome coordinates.
PROTEINS = [
    ("gag", "P04591", 1, 500),
    ("gag-pol", "P04585", 433, 1435),
    ("env", "P04578", 1, 856),
    ("tat", "P04608", 1, 86),
    ("nef", "P04601", 1, 206),
    ("rev", "P04618", 1, 116),
    ("vif", "P69723", 1, 192),
    ("vpr", "P69726", 1, 78),
    ("vpu", "P05919", 1, 81),
]

BASE_URL = "https://hivmut.org/php/download_excel.php"
OUT_FIELDS = ["protein", "uniprot", "residue", "adjust_mut", "pubmed_id", "title", "authors", "journal"]


def decode(resp):
    """hivmut declares Content-Type: text/plain;charset=UTF-8 but the body is
    actually latin-1/cp1252 (accented author names like Tisne,C decode wrong
    as UTF-8). Try UTF-8 first, fall back rather than trusting the header."""
    raw = resp.content
    for codec in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(codec)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def fetch(session, uniprot, residue, timeout, retries, delay):
    params = {"residue": residue, "uniprot": uniprot, "filetype": "csv"}
    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(BASE_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                time.sleep(delay + random.uniform(0, 0.2))
                return decode(resp)
            last_err = "HTTP %d" % resp.status_code
        except requests.RequestException as exc:
            last_err = "%s: %s" % (type(exc).__name__, exc)
        time.sleep(delay * (2 ** attempt) + random.uniform(0, 1.0))
    raise RuntimeError("failed after %d attempts (uniprot=%s residue=%d): %s"
                        % (retries, uniprot, residue, last_err))


def parse_rows(text):
    """Parse the per-residue export body. Returns a list of dicts with keys
    adjust_mut/pubmed_id/title/authors/journal, or [] if this residue has no
    mapped data (empty body, or a server error body)."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    if lines[0].lower().startswith("adjust_mut"):
        lines = lines[1:]
    elif lines[0].lower().startswith("error"):
        return []
    rows = []
    for line in lines:
        f = line.rstrip("\r").split("\t")
        if len(f) < 2:
            continue
        mut = html.unescape(f[0]).strip()
        pmid = html.unescape(f[1]).strip()
        if not mut or not pmid.isdigit():
            continue
        if len(f) >= 5:
            # Anchor from both ends: titles can themselves contain stray tabs,
            # so whatever's left in the middle is the title.
            journal = html.unescape(f[-1]).strip()
            authors = html.unescape(f[-2]).strip()
            title = html.unescape("\t".join(f[2:-2])).strip()
        else:
            # Short rows (2-4 fields) mean hivmut has no metadata for this
            # paper. Roughly half of all rows are like this -- keep them with
            # blank metadata rather than dropping or misassigning fields.
            title = authors = journal = ""
        rows.append({"adjust_mut": mut, "pubmed_id": pmid, "title": title,
                     "authors": authors, "journal": journal})
    return rows


def cache_path(cache_dir, uniprot, residue):
    return os.path.join(cache_dir, "%s_%d.txt" % (uniprot, residue))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="hivmut_mutagenesis_combined.tsv")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--delay", type=float, default=0.6, help="seconds between requests")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--contact", default="sfriedman@nygenome.org",
                     help="contact email sent in the User-Agent, as a courtesy to the host")
    ap.add_argument("--only", nargs="*", metavar="GENE",
                     help="restrict to these gene names (default: all nine)")
    args = ap.parse_args()

    proteins = PROTEINS
    if args.only:
        wanted = {g.lower() for g in args.only}
        proteins = [p for p in PROTEINS if p[0].lower() in wanted]
        if not proteins:
            sys.exit("no match for --only %r (choices: %s)"
                      % (args.only, ", ".join(p[0] for p in PROTEINS)))

    os.makedirs(args.cache_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "hivmut-mutagenesis-download/1.0 (research script; contact: %s) python-requests"
                      % args.contact,
        "Accept-Language": "en",
        "Connection": "close",
    })

    n_requests = n_cache_hits = n_rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS, delimiter="\t")
        writer.writeheader()

        for name, uniprot, start, end in proteins:
            gene_rows = gene_residues_with_data = 0
            total = end - start + 1
            for i, residue in enumerate(range(start, end + 1), 1):
                cpath = cache_path(args.cache_dir, uniprot, residue)
                if os.path.exists(cpath):
                    with open(cpath, encoding="utf-8") as cf:
                        text = cf.read()
                    n_cache_hits += 1
                else:
                    text = fetch(session, uniprot, residue, args.timeout, args.retries, args.delay)
                    with open(cpath, "w", encoding="utf-8") as cf:
                        cf.write(text)
                    n_requests += 1

                rows = parse_rows(text)
                if rows:
                    gene_residues_with_data += 1
                for row in rows:
                    writer.writerow({"protein": name, "uniprot": uniprot, "residue": residue, **row})
                    gene_rows += 1
                    n_rows += 1

                if i % 100 == 0 or i == total:
                    print("  [%s] %d/%d residues checked, %d with data, %d rows so far"
                          % (name, i, total, gene_residues_with_data, gene_rows), file=sys.stderr)

            print("[%s] done: %d/%d residues had data, %d rows"
                  % (name, gene_residues_with_data, total, gene_rows), file=sys.stderr)

    print("\nAll done. %d HTTP requests made (%d served from cache), %d total rows written to %s"
          % (n_requests, n_cache_hits, n_rows, args.out), file=sys.stderr)


if __name__ == "__main__":
    main()
