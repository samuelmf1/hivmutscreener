#!/usr/bin/env python3
"""Critic pass: ask a local LLM to read each already-generated .llm answer
and decide whether it actually concludes "yes, some variant beats wildtype",
writing a YES/NO/UNCLEAR verdict next to it.

Why a second LLM call instead of just grepping the answer for "yes": the raw
answers hedge ("essentially equal to wildtype", "within error of 100%",
"technically below 100% though the error bar overlaps") in ways that are
easy for keyword matching to get wrong in either direction. A short
classification call handles that far more reliably.

Input:  data/extracted/<stem>/<stem>.qwen.llm, <stem>.gptoss.llm
        (written by batch_process.py)
Output: data/extracted/<stem>/<stem>.qwen.flag, <stem>.gptoss.flag
        each holding "YES", "NO", or "UNCLEAR" on the first line, optionally
        followed by a one-sentence reason.

Resumable: a stem/source pair already having a .flag file is skipped, so
re-running after an interruption (or after batch_process.py produces more
.llm files) only does the missing work.

Run this only when neither vLLM service is mid-use by another job (it
switches the active model via systemd, same as batch_process.py).

Usage:
    python critic_flag.py                  # critic-model gpt-oss, both sources
    python critic_flag.py --model qwen     # use Qwen as the critic instead
    python critic_flag.py --source qwen    # only classify Qwen's own answers
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from batch_process import API_KEY_PATH, DEFAULT_QUESTION, EXTRACTED_ROOT, MODELS, switch_to  # noqa: E402

CRITIC_MODELS = {m["suffix"]: m for m in MODELS}
DEFAULT_CRITIC = "gptoss"  # text-only and fast; no need for vision here
SOURCES = ["qwen", "gptoss"]

PROMPT_TEMPLATE = (
    "You are fact-checking a colleague's answer to a yes/no research question "
    "about a scientific paper.\n\n"
    "Question asked: \"{question}\"\n\n"
    "Colleague's answer:\n<answer>\n{answer}\n</answer>\n\n"
    "Based only on the answer above, did the colleague conclude YES -- that "
    "the paper reports at least one variant/mutant that clearly outperforms "
    "wildtype (i.e., is reported above 100% on whatever metric is being "
    "studied)? Treat a hedge like \"essentially equal to wildtype\" or "
    "\"within error of 100%\" as NO. If the answer is too garbled or "
    "inconclusive to tell, say UNCLEAR instead.\n\n"
    "Reply with exactly one word on the first line -- YES, NO, or UNCLEAR -- "
    "optionally followed by a one-sentence reason on the next line."
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def pending_tasks(sources: list[str]) -> list[tuple[str, str, Path, Path]]:
    """[(stem, source_suffix, answer_path, flag_path), ...] still needing a verdict."""
    tasks = []
    for out_dir in sorted(EXTRACTED_ROOT.iterdir()):
        if not out_dir.is_dir():
            continue
        stem = out_dir.name
        for source in sources:
            answer_path = out_dir / f"{stem}.{source}.llm"
            flag_path = out_dir / f"{stem}.{source}.flag"
            if answer_path.exists() and not flag_path.exists():
                tasks.append((stem, source, answer_path, flag_path))
    return tasks


async def classify_one(client: AsyncOpenAI, critic: dict, question: str,
                        stem: str, source: str, answer_path: Path, flag_path: Path,
                        sem: asyncio.Semaphore) -> tuple[str, str, bool, str]:
    answer_text = answer_path.read_text()
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer_text)

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=critic["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                extra_body={"reasoning_effort": "low"},
                timeout=critic["timeout"],
            )
        except Exception as e:
            return stem, source, False, f"request failed: {e}"

    reply = (resp.choices[0].message.content or "").strip()
    first_line = reply.splitlines()[0].strip().upper() if reply else ""
    verdict = next((v for v in ("YES", "NO", "UNCLEAR") if first_line.startswith(v)), "UNCLEAR")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flag_path.write_text(f"[{timestamp}][critic:{critic['model']}]\n{verdict}\n{reply}\n")
    return stem, source, True, verdict


async def run_async(critic: dict, question: str, tasks: list, concurrency: int):
    api_key = API_KEY_PATH.read_text().strip()
    client = AsyncOpenAI(base_url=f"http://localhost:{critic['port']}/v1", api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    coros = [classify_one(client, critic, question, stem, source, ap, fp, sem)
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
            log(f"  progress: {done}/{len(tasks)} ({ok} ok, {len(failed)} failed) {verdicts}")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(CRITIC_MODELS), default=DEFAULT_CRITIC,
                         help=f"Which local model to use as the critic (default {DEFAULT_CRITIC})")
    parser.add_argument("--source", choices=SOURCES + ["both"], default="both",
                         help="Classify Qwen's answers, gpt-oss's, or both (default both)")
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION,
                         help="The original question that was asked (must match what the .llm answers responded to)")
    parser.add_argument("--concurrency", type=int, default=60,
                         help="Classification is a short text-only call, so this can run much higher than the main answer pass")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending pairs (for testing)")
    args = parser.parse_args()

    sources = SOURCES if args.source == "both" else [args.source]
    tasks = pending_tasks(sources)
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        log("nothing to do -- every .llm file already has a .flag")
        return
    log(f"{len(tasks)} answer(s) need a critic verdict (sources={sources})")

    critic = dict(CRITIC_MODELS[args.model])
    critic["timeout"] = args.timeout

    try:
        log(f"Switching to {critic['model']} (critic)...")
        switch_to(critic["service"], critic["port"])
        log(f"{critic['model']} is up. Classifying {len(tasks)} answer(s) (concurrency={args.concurrency})...")
        failed = asyncio.run(run_async(critic, args.question, tasks, args.concurrency))
    finally:
        default = next(m for m in MODELS if m["service"] == "qwen35-vllm.service")
        log(f"Restoring default model: {default['service']}")
        try:
            switch_to(default["service"], default["port"])
        except Exception as e:
            log(f"warning: failed to confirm default model restored: {e}")

    if failed:
        log(f"{len(failed)} pair(s) failed -- re-run this script to retry them (resumable)")
    log("Done.")


if __name__ == "__main__":
    main()
