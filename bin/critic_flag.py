#!/usr/bin/env python3
"""Critic pass: for each paper, ask a local LLM to look at that paper's
extraction answer(s) (the .llm files -- up to two, one per extraction model)
and count how many of them present real evidence that some mutation/variant
performs better than wildtype (>100%) on the assay the paper actually
studied.

Run twice, once per critic model, as two independent passes:

    python critic_flag.py --model qwen      # pass 1
    python critic_flag.py --model gptoss    # pass 2

(or just `python critic_flag.py` with no --model, which runs both passes
back-to-back). Each pass judges *every* .llm file with that one critic, so
in the end each source .llm file has been judged once by Qwen-as-critic and
once by gpt-oss-as-critic -- two independent opinions per file, which
build_manifest.py turns into a 0-2 "how many files showed evidence" count
per critic pass, and then averages the two passes into a final score. Using
two different models as critics (instead of trusting either one's own
self-report) is the point: an extraction model rubber-stamping its own
answer is a much weaker check than a second model re-reading it skeptically,
and disagreement between the two critics is itself signal.

The question is binary and deliberately strict, not the vague "does this
sound like a yes" scoring the first version of this script did: it requires
a *specific* mutation and a *direct, quantitative, same-assay* comparison to
wildtype. Evolutionary/statistical proxies (Ka/Ks, dN/dS, conservation
scores, phylogenetic selection signatures), in-silico-only predictions,
non-mutation variants (chemical analogs/reagents beating wildtype protein,
not a mutation of it), and vague qualitative language ("may enhance", "trend
towards") do not count, even when the extraction answer itself claims "yes"
-- the critic is there to catch exactly that kind of overreach. See
PROMPT_TEMPLATE.

On a YES, the critic also names the mutation(s) and classifies whether the
advantage is antiretroviral DRUG_RESISTANCE (a resistance/susceptibility
measurement against a specific drug class -- NRTI/NNRTI/PI/InSTI/entry/
maturation inhibitor) or an INTRINSIC property (replication, infectivity,
binding, enzymatic activity, stability, measured without drug selection).
Drug-resistance hits are lower priority for this screen than intrinsic
gain-of-function mutations, so build_manifest.py reports them separately.

Input:  data/extracted/<stem>/<stem>.qwen.llm, <stem>.gptoss.llm
        (written by batch_process.py)
Output: data/extracted/<stem>/<stem>.<source>.<critic>.flag, one per
        (source .llm file, critic model) pair, holding:
            EVIDENCE: <YES|NO>
            MUTATION: <comma-separated name(s), or "none">
            MUTATION_TYPE: <DRUG_RESISTANCE|INTRINSIC|NONE>
            REASON: <explanation>

Resumable: a (source, critic) pair that already has a .flag file is skipped.
Use --clear-cache to delete existing flags for the sources/critics about to
run first (e.g. after changing the prompt, like this rewrite) so the whole
corpus gets re-judged instead of silently keeping stale verdicts.

Usage:
    python critic_flag.py                  # both passes: qwen critic, then gptoss critic
    python critic_flag.py --model qwen     # only the qwen-critic pass
    python critic_flag.py --model gptoss   # only the gpt-oss-critic pass
    python critic_flag.py --source qwen    # only judge Qwen's own extraction answers
    python critic_flag.py --clear-cache    # wipe existing flags for the selected sources/critics first
"""

import argparse
import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from batch_process import API_KEY_PATH, DEFAULT_QUESTION, EXTRACTED_ROOT, MODELS, switch_to  # noqa: E402

CRITIC_MODELS = {m["suffix"]: m for m in MODELS}
CRITIC_ORDER = ["qwen", "gptoss"]  # pass 1, then pass 2
SOURCES = ["qwen", "gptoss"]

# Tuned per critic model for this classification task specifically (independent of
# the reasoning_effort batch_process.py uses for the much harder extraction task).
# Qwen3.5 is a hybrid thinking model: asking it to reason at all ("low"+) makes it
# spend the *entire* token budget on chain-of-thought and return empty content
# (finish_reason="length", 0 answer tokens) -- "none" gets a clean, correctly
# formatted, correct answer in ~80 tokens. gpt-oss's Harmony format rejects
# reasoning_effort="none" outright (400 error), but "low" answers cleanly and fast.
CRITIC_REASONING_EFFORT = {"qwen": "none", "gptoss": "low"}

PROMPT_TEMPLATE = (
    "You are a skeptical, analytically rigorous scientific reviewer. A colleague "
    "was asked this question about a paper:\n\n"
    "\"{question}\"\n\n"
    "Here is the colleague's answer (an LLM's own extraction from the paper, which "
    "may itself be sloppy, over-eager, or conflate the wrong metric -- judge the "
    "underlying evidence it cites, not just its stated conclusion):\n"
    "<answer>\n{answer}\n</answer>\n\n"
    "Decide whether the answer actually establishes, for at least one *specific, "
    "named* mutation/variant, *direct experimental evidence* that it exceeds "
    "wildtype's performance (>100%) on the same assay/metric the paper used to "
    "characterize wildtype -- e.g. viral fitness or replication capacity, "
    "infectivity, entry/receptor binding or tropism/targeting (co-receptor use, "
    "cell-type or tissue targeting, attachment), virion/particle assembly or "
    "formation efficiency (budding, capsid/particle formation, release, "
    "maturation), enzymatic activity, general binding affinity, drug resistance "
    "(fold-change/EC50/IC50), protein expression or stability, or an equivalent "
    "directly-measured functional readout, reported as a same-experiment "
    "head-to-head comparison against a wildtype control -- not a hedge, a trend, "
    "a non-significant difference, or values within error of 100%.\n\n"
    "A bare percentage/fold-change number is NOT required -- a paper's own "
    "specific, unhedged qualitative or categorical call counts too, as long as "
    "it's a direct same-experiment comparison to wildtype: e.g. a systematic "
    "phenotype scale where the mutant is explicitly scored/categorized above "
    "wildtype (such as a table ranking mutants as \"Enhanced\" vs. wildtype-like "
    "vs. reduced vs. failed), or an unambiguous specific statement like "
    "\"produced significantly longer/more particles than wildtype\" or \"only the "
    "mutant, not wildtype, retained function\". The bar is specificity and "
    "directness, not decimal precision.\n\n"
    "Answer NO (even if the colleague's answer says \"yes\") when the cited evidence "
    "is actually one of these common false positives:\n"
    "  - An evolutionary/statistical proxy for selection, not a measured activity: "
    "Ka/Ks or dN/dS ratios, phylogenetic branch-selection signatures, conservation "
    "scores, or similar. A Ka/Ks > 1 means positive selection, not \">100% activity\".\n"
    "  - A purely in-silico/computational prediction (docking score, structure "
    "prediction, MD simulation) with no wet-lab measurement.\n"
    "  - Genuinely vague or hedged language (\"may improve fitness\", \"could "
    "enhance binding\", \"some mutants trended higher\", \"appeared somewhat "
    "greater\") that doesn't commit to a specific mutation actually beating "
    "wildtype in that experiment -- as opposed to a qualitative claim that IS "
    "specific and committed (see above), which counts as YES.\n"
    "  - A result borrowed from a different paper/prior study being discussed, "
    "rather than this paper's own data.\n"
    "  - No wildtype baseline was actually measured in the same assay.\n"
    "  - The thing that outperforms is NOT a mutation/variant of the protein, gene, "
    "or organism under study -- e.g. a synthetic chemical analog, a differently-"
    "formulated drug/reagent, a different delivery vehicle, or an unrelated "
    "molecule outcompeting the wildtype protein (such as a modified heparin "
    "out-binding natural heparin). Only mutations/variants of the biological "
    "sequence itself count; if every actual protein/viral mutant tested performs "
    "at or below wildtype, that is a NO even if some other compound in the paper "
    "outperforms something else.\n\n"
    "If EVIDENCE is YES, also classify what *kind* of advantage it is -- this "
    "distinction matters a lot downstream, so get it right:\n"
    "  - MUTATION_TYPE: DRUG_RESISTANCE -- the >100% comparison IS itself a "
    "resistance/susceptibility measurement to an antiretroviral drug (an NRTI, "
    "NNRTI, protease inhibitor, integrase strand-transfer inhibitor, entry/fusion "
    "inhibitor, maturation inhibitor, etc.) -- e.g. a fold-resistance value, an "
    "EC50/IC50 shift under drug, or replication/infectivity measured in the "
    "presence of the drug. The mutation's advantage IS that it evades a drug.\n"
    "  - MUTATION_TYPE: INTRINSIC -- the >100% advantage is a general/inherent "
    "property measured without drug selection (replication capacity, infectivity, "
    "enzymatic activity, binding affinity, protein stability/expression, etc. in "
    "the absence of the relevant drug), even if the same paper separately also "
    "discusses drug resistance elsewhere.\n"
    "If EVIDENCE is NO, set MUTATION_TYPE: NONE.\n\n"
    "Read carefully and do not miss a real, well-supported hit -- if the answer "
    "cites a specific mutation with a specific number or unambiguous statement "
    "clearly above 100% of wildtype in a real assay, that is a YES even if the "
    "colleague's own prose is clumsy. But do not give the benefit of the doubt to "
    "borderline, hedged, or proxy-metric claims -- this is a precision filter, and "
    "an over-eager colleague's confident tone is not evidence.\n\n"
    "Reply strictly in this exact format:\n"
    "EVIDENCE: <YES|NO>\n"
    "MUTATION: <comma-separated list of the specific mutation/variant name(s), e.g. "
    "\"M184V, K65R\" -- names only, no commentary -- or \"none\" if EVIDENCE is NO>\n"
    "MUTATION_TYPE: <DRUG_RESISTANCE|INTRINSIC|NONE>\n"
    "REASON: <one or two sentence justification citing the actual evidence type>"
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def flag_path_for(out_dir: Path, stem: str, source: str, critic: str) -> Path:
    return out_dir / f"{stem}.{source}.{critic}.flag"


def pending_tasks(sources: list[str], critic: str) -> list[tuple[str, str, Path, Path]]:
    """[(stem, source_suffix, answer_path, flag_path), ...] still needing a verdict
    from this critic."""
    tasks = []
    for out_dir in sorted(EXTRACTED_ROOT.iterdir()):
        if not out_dir.is_dir():
            continue
        stem = out_dir.name
        for source in sources:
            answer_path = out_dir / f"{stem}.{source}.llm"
            flag_path = flag_path_for(out_dir, stem, source, critic)
            if answer_path.exists() and not flag_path.exists():
                tasks.append((stem, source, answer_path, flag_path))
    return tasks


def clear_cache(sources: list[str], critics: list[str]) -> int:
    removed = 0
    for out_dir in sorted(EXTRACTED_ROOT.iterdir()):
        if not out_dir.is_dir():
            continue
        stem = out_dir.name
        for source in sources:
            for critic in critics:
                flag_path = flag_path_for(out_dir, stem, source, critic)
                if flag_path.exists():
                    flag_path.unlink()
                    removed += 1
    return removed


EVIDENCE_RE = re.compile(r"EVIDENCE:\s*(YES|NO)", re.IGNORECASE)
MUTATION_TYPE_RE = re.compile(r"MUTATION_TYPE:\s*(DRUG_RESISTANCE|INTRINSIC|NONE)", re.IGNORECASE)


async def classify_one(client: AsyncOpenAI, critic: dict, question: str, reasoning_effort: str,
                        stem: str, source: str, answer_path: Path, flag_path: Path,
                        sem: asyncio.Semaphore) -> tuple[str, str, bool, str]:
    answer_text = answer_path.read_text()
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer_text)

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=critic["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                extra_body={"reasoning_effort": reasoning_effort},
                timeout=critic["timeout"],
            )
        except Exception as e:
            return stem, source, False, f"request failed: {e}"

    reply = (resp.choices[0].message.content or "").strip()

    match = EVIDENCE_RE.search(reply)
    if not match:
        # Don't guess on a genuinely unparseable reply -- leave no .flag file so
        # this pair gets retried on the next run instead of silently miscounted.
        return stem, source, False, f"unparseable reply (no EVIDENCE: line): {reply[:200]!r}"
    evidence = match.group(1).upper()

    type_match = MUTATION_TYPE_RE.search(reply)
    mutation_type = type_match.group(1).upper() if type_match else None
    if evidence == "YES" and mutation_type is None:
        # A YES without a parseable DRUG_RESISTANCE/INTRINSIC call is missing
        # information we need downstream -- retry rather than guess.
        return stem, source, False, f"YES but no parseable MUTATION_TYPE: line: {reply[:200]!r}"
    mutation_type = mutation_type or "NONE"

    # `reply` already contains EVIDENCE/MUTATION/MUTATION_TYPE/REASON in this exact
    # format (that's what we asked for) -- just prefix it with a timestamp header
    # rather than re-emitting the parsed fields a second time above it.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flag_path.write_text(f"[{timestamp}][critic:{critic['model']}]\n{reply}\n")
    return stem, source, True, f"{evidence}/{mutation_type}"


async def run_async(critic: dict, question: str, reasoning_effort: str, tasks: list, concurrency: int):
    api_key = API_KEY_PATH.read_text().strip()
    client = AsyncOpenAI(base_url=f"http://localhost:{critic['port']}/v1", api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    coros = [classify_one(client, critic, question, reasoning_effort, stem, source, ap, fp, sem)
             for stem, source, ap, fp in tasks]
    ok, failed, verdicts, done = 0, [], {}, 0
    for coro in asyncio.as_completed(coros):
        stem, source, success, verdict_or_msg = await coro
        done += 1
        if success:
            ok += 1
            verdicts[verdict_or_msg] = verdicts.get(verdict_or_msg, 0) + 1
        else:
            failed.append((stem, source, verdict_or_msg))
            log(f"  FAILED {stem} [{source}]: {verdict_or_msg}")
        if done % 25 == 0 or done == len(tasks):
            sorted_verdicts = dict(sorted(verdicts.items()))
            log(f"  progress: {done}/{len(tasks)} ({ok} ok, {len(failed)} failed) {sorted_verdicts}")
    return failed


def run_pass(critic_suffix: str, sources: list[str], question: str, reasoning_effort: str | None,
             concurrency: int, timeout: int, limit: int | None) -> list:
    tasks = pending_tasks(sources, critic_suffix)
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        log(f"[{critic_suffix} critic] nothing to do -- every judged .llm file already has a .flag")
        return []
    log(f"[{critic_suffix} critic] {len(tasks)} answer(s) need a verdict (sources={sources})")

    critic = dict(CRITIC_MODELS[critic_suffix])
    critic["timeout"] = timeout
    reasoning_effort = reasoning_effort or CRITIC_REASONING_EFFORT[critic_suffix]

    log(f"[{critic_suffix} critic] switching to {critic['model']}...")
    switch_to(critic["service"], critic["port"])
    log(f"[{critic_suffix} critic] {critic['model']} is up. Classifying {len(tasks)} answer(s) "
        f"(concurrency={concurrency}, reasoning_effort={reasoning_effort})...")
    failed = asyncio.run(run_async(critic, question, reasoning_effort, tasks, concurrency))
    if failed:
        log(f"[{critic_suffix} critic] {len(failed)} pair(s) failed -- re-run to retry them (resumable)")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(CRITIC_MODELS), default=None,
                         help="Which local model to use as the critic. Default: run both passes "
                              "(qwen critic, then gpt-oss critic).")
    parser.add_argument("--source", choices=SOURCES + ["both"], default="both",
                         help="Judge Qwen's extraction answers, gpt-oss's, or both (default both)")
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION,
                         help="The original question that was asked (must match what the .llm answers responded to)")
    parser.add_argument("--concurrency", type=int, default=25,
                         help="Classification concurrency")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--reasoning-effort", default=None,
                         help="reasoning_effort passed to the critic model. Default: tuned per critic "
                              f"({CRITIC_REASONING_EFFORT}) -- only override this if you know what you're doing, "
                              "Qwen3.5 in particular returns empty output above 'none' (see CRITIC_REASONING_EFFORT).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N pending pairs per pass (for testing)")
    parser.add_argument("--clear-cache", action="store_true",
                         help="Delete existing .flag files for the selected sources/critic(s) before running, "
                              "so every file is re-judged instead of skipped as already-done")
    args = parser.parse_args()

    sources = SOURCES if args.source == "both" else [args.source]
    critics = [args.model] if args.model else list(CRITIC_ORDER)

    if args.clear_cache:
        removed = clear_cache(sources, critics)
        log(f"cleared {removed} cached flag file(s) for sources={sources} critics={critics}")

    default_service = next(m for m in MODELS if m["service"] == "qwen35-vllm.service")
    try:
        all_failed = []
        for critic_suffix in critics:
            all_failed += run_pass(critic_suffix, sources, args.question, args.reasoning_effort,
                                    args.concurrency, args.timeout, args.limit)
    finally:
        log(f"Restoring default model: {default_service['service']}")
        try:
            switch_to(default_service["service"], default_service["port"])
        except Exception as e:
            log(f"warning: failed to confirm default model restored: {e}")

    if all_failed:
        log(f"{len(all_failed)} pair(s) failed across all passes -- re-run this script to retry them (resumable)")
    log("Done.")


if __name__ == "__main__":
    main()
