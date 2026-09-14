#!/usr/bin/env python3
"""Build data/manifest.tsv: one row per unique PMID referenced in the hivmut
mutagenesis data, joining the paper-fetch status with pipeline progress.

Sources:
  data/papers/.manifest.json   fetch status per PMID (downloaded / pmc_manual
                                / no_oa / error), written by
                                bin/fetch_paper/batch_fetch_papers.py
  data/papers/*.pdf            which PDFs actually exist on disk
  data/extracted/<stem>/       extraction + per-model answer + critic flag
                                files, written by bin/batch_process.py and
                                bin/critic_flag.py

critic_flag.py runs two independent critic passes over every paper's .llm
answer(s) (one pass with Qwen as critic, one with gpt-oss as critic). Within
each pass, for each paper, we count how many of its available .llm files
(0-2: one per extraction model) that critic judged as showing real evidence
of a mutation beating wildtype -- and, on a YES, whether that mutation's
advantage is antiretroviral DRUG_RESISTANCE or an INTRINSIC (non-resistance)
property. qwen_critic_score / gptoss_critic_score below are the two per-pass
"any evidence" counts, final_score is their average; the *_drugres_score
columns are the same but restricted to drug-resistance mutations, and
final_intrinsic_score = final_score - final_drugres_score is what's left --
the "pure"/intrinsic gain-of-function signal this screen mainly cares about,
since resistance-only mutations are lower priority here. Everything is
blank wherever a critic pass hasn't judged any of that paper's files yet.

Output columns:
  pmid, title, fetch_status, pmcid, doi, pdf_file, extracted,
  qwen_answered, gptoss_answered,
  qwen_critic_score, gptoss_critic_score, final_score, flagged,
  qwen_critic_drugres_score, gptoss_critic_drugres_score, final_drugres_score,
  final_intrinsic_score, flagged_intrinsic,
  mutations_intrinsic, mutations_drug_resistance

Usage:
    python build_manifest.py [-o data/manifest.tsv]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
EXTRACTED_ROOT = PROJECT_ROOT / "data" / "extracted"
FETCH_MANIFEST = PAPERS_DIR / ".manifest.json"
HIVMUT_TSV = PROJECT_ROOT / "data" / "hivmut" / "hivmut_mutagenesis_combined.tsv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "manifest.tsv"

PMID_RE = re.compile(r"^PMID(\d+)(?:_.*)?$")
SOURCES = ["qwen", "gptoss"]
CRITICS = ["qwen", "gptoss"]
EVIDENCE_RE = re.compile(r"EVIDENCE:\s*(YES|NO)", re.IGNORECASE)
MUTATION_RE = re.compile(r"^MUTATION:\s*(.+)$", re.MULTILINE)
MUTATION_TYPE_RE = re.compile(r"MUTATION_TYPE:\s*(DRUG_RESISTANCE|INTRINSIC|NONE)", re.IGNORECASE)


def load_fetch_manifest() -> dict:
    if not FETCH_MANIFEST.exists():
        sys.exit(f"missing {FETCH_MANIFEST} -- extract it from the papers zip first")
    return json.loads(FETCH_MANIFEST.read_text())


def load_tsv_titles() -> dict:
    """Fallback titles for any PMID not in the fetch manifest."""
    titles = {}
    if not HIVMUT_TSV.exists():
        return titles
    with open(HIVMUT_TSV, newline="", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pmid = (row.get("pmid") or "").strip()
            if pmid and pmid.isdigit():
                titles.setdefault(pmid, (row.get("title") or "").strip() or None)
    return titles


def pdf_stems_by_pmid() -> dict:
    """PMID -> pdf stem, from what's actually on disk in data/papers/."""
    stems = {}
    for pdf in PAPERS_DIR.glob("*.pdf"):
        m = PMID_RE.match(pdf.stem)
        if m:
            stems[m.group(1)] = pdf.stem
    return stems


def read_flag(path: Path):
    """Parsed (evidence: bool, mutation_type: str, mutations: list[str]) for one
    (source, critic) flag file, or None if it doesn't exist / has no verdict yet."""
    if not path.exists():
        return None
    text = path.read_text()
    evidence_match = EVIDENCE_RE.search(text)
    if not evidence_match:
        return None
    evidence = evidence_match.group(1).upper() == "YES"

    mutation_type_match = MUTATION_TYPE_RE.search(text)
    mutation_type = mutation_type_match.group(1).upper() if mutation_type_match else "NONE"

    mutations = []
    if evidence:
        mutation_match = MUTATION_RE.search(text)
        raw = mutation_match.group(1).strip() if mutation_match else ""
        if raw and raw.lower() != "none":
            mutations = [m.strip() for m in raw.split(",") if m.strip()]

    return {"evidence": evidence, "mutation_type": mutation_type, "mutations": mutations}


def critic_summary(out_dir: Path, stem: str, critic: str) -> dict:
    """This critic pass's verdict for one paper: how many of its available .llm
    files showed evidence overall vs. specifically drug-resistance evidence (each
    0-2, or "" if the pass hasn't judged any of this paper's files yet), plus the
    mutation names it found in each bucket."""
    flags = [read_flag(out_dir / f"{stem}.{source}.{critic}.flag") for source in SOURCES]
    judged = [f for f in flags if f is not None]
    mutations_intrinsic, mutations_drugres = [], []
    for f in judged:
        if f["mutation_type"] == "INTRINSIC":
            mutations_intrinsic += f["mutations"]
        elif f["mutation_type"] == "DRUG_RESISTANCE":
            mutations_drugres += f["mutations"]
    if not judged:
        return {"score": "", "drugres_score": "", "mutations_intrinsic": [], "mutations_drugres": []}
    return {
        "score": sum(f["evidence"] for f in judged),
        "drugres_score": sum(f["mutation_type"] == "DRUG_RESISTANCE" for f in judged),
        "mutations_intrinsic": mutations_intrinsic,
        "mutations_drugres": mutations_drugres,
    }


def dedup(names: list[str]) -> str:
    seen = []
    for n in names:
        n = " ".join(n.split())  # collapse any stray whitespace/newlines for a clean TSV cell
        if n and n not in seen:
            seen.append(n)
    return "; ".join(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    fetch_manifest = load_fetch_manifest()
    fallback_titles = load_tsv_titles()
    stems_by_pmid = pdf_stems_by_pmid()

    all_pmids = set(fetch_manifest) | set(fallback_titles)

    rows = []
    counts = {
        "fetch_status": {}, "extracted": 0, "qwen_answered": 0,
        "gptoss_answered": 0, "flagged": 0, "flagged_intrinsic": 0,
    }
    for pmid in sorted(all_pmids, key=int):
        entry = fetch_manifest.get(pmid, {})
        title = entry.get("title") or fallback_titles.get(pmid) or ""
        fetch_status = entry.get("status", "not_attempted")
        counts["fetch_status"][fetch_status] = counts["fetch_status"].get(fetch_status, 0) + 1

        stem = stems_by_pmid.get(pmid)
        pdf_file = f"{stem}.pdf" if stem else ""

        extracted = qwen_answered = gptoss_answered = False
        critic = {c: {"score": "", "drugres_score": "", "mutations_intrinsic": [], "mutations_drugres": []}
                  for c in CRITICS}
        if stem:
            out_dir = EXTRACTED_ROOT / stem
            extracted = (out_dir / f"{stem}.txt").exists()
            qwen_answered = (out_dir / f"{stem}.qwen.llm").exists()
            gptoss_answered = (out_dir / f"{stem}.gptoss.llm").exists()
            critic = {c: critic_summary(out_dir, stem, c) for c in CRITICS}

        # Average the two critic passes into one final score (both overall and
        # drug-resistance-only); flagged if, on average, at least one .llm file
        # per pass showed real evidence. final_intrinsic_score is what's left
        # after subtracting out drug-resistance hits -- the signal this screen
        # actually cares about most.
        scores = [critic[c]["score"] for c in CRITICS if isinstance(critic[c]["score"], int)]
        drugres_scores = [critic[c]["drugres_score"] for c in CRITICS if isinstance(critic[c]["drugres_score"], int)]
        final_score = round(sum(scores) / len(scores), 2) if scores else ""
        final_drugres_score = round(sum(drugres_scores) / len(drugres_scores), 2) if drugres_scores else ""
        final_intrinsic_score = (
            round(final_score - final_drugres_score, 2)
            if isinstance(final_score, float) and isinstance(final_drugres_score, float) else ""
        )
        flagged = 1 if (isinstance(final_score, float) and final_score >= 1) else 0
        flagged_intrinsic = 1 if (isinstance(final_intrinsic_score, float) and final_intrinsic_score >= 1) else 0

        mutations_intrinsic = dedup(critic["qwen"]["mutations_intrinsic"] + critic["gptoss"]["mutations_intrinsic"])
        mutations_drugres = dedup(critic["qwen"]["mutations_drugres"] + critic["gptoss"]["mutations_drugres"])

        if extracted:
            counts["extracted"] += 1
        if qwen_answered:
            counts["qwen_answered"] += 1
        if gptoss_answered:
            counts["gptoss_answered"] += 1
        if flagged:
            counts["flagged"] += 1
        if flagged_intrinsic:
            counts["flagged_intrinsic"] += 1

        rows.append({
            "pmid": pmid,
            "title": title,
            "fetch_status": fetch_status,
            "pmcid": entry.get("pmcid") or "",
            "doi": entry.get("doi") or "",
            "pdf_file": pdf_file,
            "extracted": extracted,
            "qwen_answered": qwen_answered,
            "gptoss_answered": gptoss_answered,
            "qwen_critic_score": critic["qwen"]["score"],
            "gptoss_critic_score": critic["gptoss"]["score"],
            "final_score": final_score,
            "flagged": flagged,
            "qwen_critic_drugres_score": critic["qwen"]["drugres_score"],
            "gptoss_critic_drugres_score": critic["gptoss"]["drugres_score"],
            "final_drugres_score": final_drugres_score,
            "final_intrinsic_score": final_intrinsic_score,
            "flagged_intrinsic": flagged_intrinsic,
            "mutations_intrinsic": mutations_intrinsic,
            "mutations_drug_resistance": mutations_drugres,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} papers -> {args.out}", file=sys.stderr)
    print(f"fetch status: {counts['fetch_status']}", file=sys.stderr)
    print(f"extracted: {counts['extracted']}, qwen answered: {counts['qwen_answered']}, "
          f"gptoss answered: {counts['gptoss_answered']}", file=sys.stderr)
    print(f"flagged (final_score >= 1, any evidence): {counts['flagged']}", file=sys.stderr)
    print(f"flagged_intrinsic (final_intrinsic_score >= 1, excludes drug-resistance-only hits): "
          f"{counts['flagged_intrinsic']}", file=sys.stderr)


if __name__ == "__main__":
    main()
