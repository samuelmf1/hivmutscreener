#!/usr/bin/env python3
"""Batch-process every PDF in data/papers/ through extraction and both
LLMs, at scale (designed for thousands of papers).

Runs in three phases so each model/tool is only loaded/started once:
  1. Extraction  - Docling text+image extraction, parallelized across CPU
                   workers (runs entirely on CPU here, no GPU contention).
  2. Qwen pass   - starts qwen35-vllm.service once, asks all papers missing
                   a Qwen answer, with many requests in flight concurrently.
  3. gpt-oss pass- switches to gptoss-vllm.service once, same idea.

Every step is resumable: a paper already having its output file is skipped,
so re-running after an interruption (or after dropping more PDFs into data/papers/)
only does the missing work.

Output layout matches ask_paper.py:
  data/extracted/<stem>/images/
  data/extracted/<stem>/<stem>.txt
  data/extracted/<stem>/<stem>.qwen.llm     (Qwen's answer)
  data/extracted/<stem>/<stem>.gptoss.llm   (gpt-oss's answer)
"""

import argparse
import asyncio
import base64
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PDFS_DIR = PROJECT_ROOT / "data" / "papers"
EXTRACTED_ROOT = PROJECT_ROOT / "data" / "extracted"
API_KEY_PATH = Path("~/.config/qwen35/api_key.txt").expanduser()
PDFEXTRACT_PY = "/home/sfriedman/.conda/envs/pdfextract/bin/python3"
EXTRACT_SCRIPT = str(SCRIPT_DIR / "extract_pdf.py")
DEFAULT_SERVICE = "qwen35-vllm.service"
DEFAULT_QUESTION = (
    "Do any variants in this paper perform better than wildtype? "
    "(> 100 percent on whatever variable is being studied)"
)

# concurrency: NOT max_num_seqs - vLLM can only actually run as many requests
# concurrently as fit in its KV cache, which for Qwen (full paper text + table
# images per request) turned out to be ~3-6 in practice even though
# max_num_seqs=200 and this was set to 150. The other ~145 in-flight requests
# just sat in the server's queue burning wall-clock until the client-side
# timeout below killed them - that's what caused a 92% failure rate on a real
# run (see git history). Keep this close to the real steady-state "Running"
# count from the vLLM engine log (`journalctl --user -u <service>`), not
# max_num_seqs. gpt-oss (text-only, no images) has a much smaller per-request
# KV footprint and really does sustain ~70-80 concurrent, so it can stay high.
# timeout: per-request client timeout; must comfortably exceed the time a
# request can spend queued behind `concurrency` others ahead of it, not just
# the time to actually generate an answer.
# reasoning_effort: gpt-oss ignores Qwen's enable_thinking template kwarg, so use the
# universal top-level reasoning_effort param for both. gpt-oss needs "high" to reliably
# parse this paper's tables correctly (lower efforts sometimes missed entries), which in
# turn needs a bigger max_tokens budget than its default reasoning uses.
MODELS = [
    {"service": "qwen35-vllm.service", "port": 8000, "model": "Qwen3.5-27B-FP8",
     "vision": True, "suffix": "qwen", "concurrency": 10, "timeout": 1800,
     "max_tokens": 4096, "reasoning_effort": "none"},
    {"service": "gptoss-vllm.service", "port": 8001, "model": "gpt-oss-20b",
     "vision": False, "suffix": "gptoss", "concurrency": 100, "timeout": 900,
     "max_tokens": 8192, "reasoning_effort": "high"},
]


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


# ---------- Phase 1: extraction ----------

def extract_one(pdf_path: Path, timeout: int) -> tuple[str, bool, str]:
    stem = pdf_path.stem
    out_dir = EXTRACTED_ROOT / stem
    text_path = out_dir / f"{stem}.txt"
    if text_path.exists():
        return stem, True, "already extracted"

    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [PDFEXTRACT_PY, EXTRACT_SCRIPT, str(pdf_path), "--image-dir", str(image_dir)],
            capture_output=True, text=True, check=True, timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        return stem, False, f"docling failed: {e.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return stem, False, f"docling timed out after {timeout}s"

    text = result.stdout
    if not text.strip():
        return stem, False, "no text extracted"
    text_path.write_text(text)
    return stem, True, "ok"


def run_extraction(pdfs: list[Path], workers: int, timeout: int):
    log(f"Phase 1: extracting {len(pdfs)} PDF(s) with {workers} parallel workers "
        f"(timeout={timeout}s)")
    ok, failed = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, p, timeout): p for p in pdfs}
        for i, fut in enumerate(as_completed(futures), 1):
            stem, success, msg = fut.result()
            if success:
                ok += 1
            else:
                failed.append((stem, msg))
                log(f"  FAILED {stem}: {msg}")
            if i % 25 == 0 or i == len(pdfs):
                log(f"  extraction progress: {i}/{len(pdfs)} ({ok} ok, {len(failed)} failed)")
    log(f"Phase 1 done: {ok} ok, {len(failed)} failed")
    return failed


# ---------- Phase 2/3: LLM passes ----------

def switch_to(service: str, port: int, timeout: float = 300.0):
    subprocess.run(["systemctl", "--user", "start", service], check=True)
    api_key = API_KEY_PATH.read_text().strip()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            OpenAI(base_url=f"http://localhost:{port}/v1", api_key=api_key).models.list()
            return
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"{service} did not become healthy within {timeout}s")


def table_images(stem: str) -> list[Path]:
    return sorted((EXTRACTED_ROOT / stem / "images").glob("table-*.png"))


async def ask_one(client: AsyncOpenAI, entry: dict, stem: str, question: str, sem: asyncio.Semaphore):
    out_path = EXTRACTED_ROOT / stem / f"{stem}.{entry['suffix']}.llm"
    if out_path.exists():
        return stem, True, "already answered"

    text_path = EXTRACTED_ROOT / stem / f"{stem}.txt"
    paper_text = text_path.read_text()
    imgs = table_images(stem) if entry["vision"] else []

    # Context guard: skip paper if estimated tokens exceed model context limit (65536)
    # Approx: 1 token ~= 3.5 chars of technical text; table images ~= 1000 tokens each.
    est_tokens = len(paper_text) / 3.5 + len(imgs) * 1000 + entry["max_tokens"] + 500
    if est_tokens > 64_000:
        return stem, False, f"skipped: prompt exceeds context limit (~{int(est_tokens)} tokens)"

    prompt = (
        f"Here is the full text of a scientific paper:\n\n<paper>\n{paper_text}\n</paper>\n\n"
    )
    if imgs:
        prompt += (
            f"The following {len(imgs)} image(s) are the paper's data tables, extracted "
            f"directly from the PDF for cases where the text extraction above may have "
            f"mangled table layout. Use them to verify any numeric claims.\n\n"
        )
    content = [{"type": "text", "text": prompt + f"Question: {question}"}]
    for img_path in imgs:
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=entry["model"],
                messages=[{"role": "user", "content": content}],
                max_tokens=entry["max_tokens"],
                extra_body={"reasoning_effort": entry["reasoning_effort"]},
                timeout=entry["timeout"],
            )
        except Exception as e:
            return stem, False, f"request failed: {e}"

    choice = resp.choices[0]
    answer = choice.message.content
    if not answer:
        reasoning = getattr(choice.message, "reasoning", None)
        answer = (
            f"[No final answer produced before max_tokens; raw reasoning below]\n\n{reasoning}"
            if reasoning else "[Model returned no content]"
        )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(f"[{timestamp}][{entry['model']}]\n{answer}\n")
    return stem, True, "ok"


async def run_llm_pass_async(entry: dict, stems: list[str], question: str):
    api_key = API_KEY_PATH.read_text().strip()
    client = AsyncOpenAI(base_url=f"http://localhost:{entry['port']}/v1", api_key=api_key)
    sem = asyncio.Semaphore(entry["concurrency"])

    pending = [s for s in stems if not (EXTRACTED_ROOT / s / f"{s}.{entry['suffix']}.llm").exists()]
    log(f"  {entry['model']}: {len(pending)}/{len(stems)} paper(s) need answers "
        f"(concurrency={entry['concurrency']})")
    if not pending:
        return []

    tasks = [ask_one(client, entry, s, question, sem) for s in pending]
    ok, failed, done = 0, [], 0
    for coro in asyncio.as_completed(tasks):
        stem, success, msg = await coro
        done += 1
        if success:
            ok += 1
        else:
            failed.append((stem, msg))
            log(f"  FAILED {stem} [{entry['model']}]: {msg}")
        if done % 25 == 0 or done == len(pending):
            log(f"  {entry['model']} progress: {done}/{len(pending)} ({ok} ok, {len(failed)} failed)")
    return failed


def run_llm_pass(entry: dict, stems: list[str], question: str):
    log(f"Switching to {entry['model']}...")
    switch_to(entry["service"], entry["port"])
    log(f"{entry['model']} is up. Asking {len(stems)} paper(s)...")
    return asyncio.run(run_llm_pass_async(entry, stems, question))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", type=Path,
                         help="Specific PDFs to process (default: everything in data/papers/)")
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION)
    parser.add_argument("--extract-workers", type=int, default=12,
                         help="Parallel Docling extraction workers (CPU-bound, default 12)")
    parser.add_argument("--extract-timeout", type=int, default=1800,
                         help="Per-PDF Docling timeout in seconds (default 1800; large "
                              "multi-hundred-page PDFs like conference abstract books need it)")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--only-model", choices=[m["suffix"] for m in MODELS], default=None,
                         help="Run only one model's pass instead of both")
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {PDFS_DIR}")
    log(f"{len(pdfs)} PDF(s) to process")

    if not args.skip_extraction:
        run_extraction(pdfs, args.extract_workers, args.extract_timeout)

    stems = [p.stem for p in pdfs if (EXTRACTED_ROOT / p.stem / f"{p.stem}.txt").exists()]
    log(f"{len(stems)} paper(s) have extracted text and are ready for LLM passes")

    models_to_run = [m for m in MODELS if args.only_model in (None, m["suffix"])]

    all_failed = {}
    try:
        for entry in models_to_run:
            failed = run_llm_pass(entry, stems, args.question)
            if failed:
                all_failed[entry["model"]] = failed
    finally:
        log(f"Restoring default model: {DEFAULT_SERVICE}")
        default = next(m for m in MODELS if m["service"] == DEFAULT_SERVICE)
        try:
            switch_to(default["service"], default["port"])
        except Exception as e:
            log(f"warning: failed to confirm default model restored: {e}")

    if all_failed:
        log("Some papers failed:")
        for model, fails in all_failed.items():
            for stem, msg in fails:
                log(f"  [{model}] {stem}: {msg}")
    log("Done.")


if __name__ == "__main__":
    main()
