#!/usr/bin/env python3
"""Extract text (and table images) from a PDF via Docling and ask multiple
local LLMs a question about it.

Input PDFs live in data/papers/ and are read in place (not copied).
Output goes to data/extracted/<pdf-stem>/, containing:
  - images/                   extracted figures and tables
  - <pdf-stem>.txt             the extracted paper text
  - <pdf-stem>.qwen.llm        Qwen's answer, headed by [datetime][model]
  - <pdf-stem>.gptoss.llm      gpt-oss's answer, headed by [datetime][model]

Only one vLLM-served model runs on the GPU at a time; this script switches
between them via systemd (Conflicts= ensures starting one stops the other) and
restores the default model (Qwen) when done.
"""

import argparse
import base64
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXTRACTED_ROOT = PROJECT_ROOT / "data" / "extracted"
API_KEY_PATH = Path("~/.config/qwen35/api_key.txt").expanduser()
PDFEXTRACT_PY = "/home/sfriedman/.conda/envs/pdfextract/bin/python3"
EXTRACT_SCRIPT = str(SCRIPT_DIR / "extract_pdf.py")
DEFAULT_SERVICE = "qwen35-vllm.service"

MODELS = [
    {"service": "qwen35-vllm.service", "port": 8000, "model": "Qwen3.5-27B-FP8", "vision": True, "suffix": "qwen"},
    {"service": "gptoss-vllm.service", "port": 8001, "model": "gpt-oss-20b", "vision": False, "suffix": "gptoss"},
]


def pdf_to_text(pdf_path: Path, image_dir: Path) -> str:
    result = subprocess.run(
        [PDFEXTRACT_PY, EXTRACT_SCRIPT, str(pdf_path), "--image-dir", str(image_dir)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def table_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("table-*.png"))


def switch_to(service: str, port: int, timeout: float = 180.0):
    subprocess.run(["systemctl", "--user", "start", service], check=True)
    api_key = API_KEY_PATH.read_text().strip()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key=api_key)
            client.models.list()
            return
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"{service} did not become healthy within {timeout}s")


def ask(port: int, model: str, paper_text: str, question: str, images: list[Path] | None = None) -> str:
    api_key = API_KEY_PATH.read_text().strip()
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key=api_key)

    text = (
        f"Here is the full text of a scientific paper:\n\n"
        f"<paper>\n{paper_text}\n</paper>\n\n"
    )
    if images:
        text += (
            f"The following {len(images)} image(s) are the paper's data tables, "
            f"extracted directly from the PDF for cases where the text extraction "
            f"above may have mangled table layout. Use them to verify any numeric "
            f"claims.\n\n"
        )
    content = [{"type": "text", "text": text + f"Question: {question}"}]
    for img_path in images or []:
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    messages = [{"role": "user", "content": content}]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    content = choice.message.content
    if not content:
        reasoning = getattr(choice.message, "reasoning", None)
        content = (
            f"[No final answer produced before max_tokens; raw reasoning below]\n\n{reasoning}"
            if reasoning else "[Model returned no content]"
        )
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "-q", "--question",
        default="Do any variants in this paper perform better than wildtype? (> 100 percent on whatever variable is being studied)",
        help="Question to ask about the paper",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"File not found: {args.pdf}")

    stem = args.pdf.stem
    out_dir = EXTRACTED_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"

    print(f"[extracting text from {args.pdf.name} via docling]", file=sys.stderr)
    paper_text = pdf_to_text(args.pdf, image_dir)
    if not paper_text.strip():
        sys.exit("No text extracted (scanned/image-only PDF?).")
    print(f"[extracted {len(paper_text)} chars]", file=sys.stderr)

    text_path = out_dir / f"{stem}.txt"
    text_path.write_text(paper_text)
    print(f"[wrote {text_path}]", file=sys.stderr)

    imgs = table_images(image_dir)
    print(f"[found {len(imgs)} extracted table image(s)]", file=sys.stderr)

    try:
        for entry in MODELS:
            print(f"[switching to {entry['model']}]", file=sys.stderr)
            switch_to(entry["service"], entry["port"])
            print(f"[asking {entry['model']}]", file=sys.stderr)
            model_imgs = imgs if entry["vision"] else []
            answer = ask(entry["port"], entry["model"], paper_text, args.question, model_imgs)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            llm_path = out_dir / f"{stem}.{entry['suffix']}.llm"
            llm_path.write_text(f"[{timestamp}][{entry['model']}]\n{answer}\n")
            print(f"[wrote {llm_path}]", file=sys.stderr)
    finally:
        print(f"[restoring default model: {DEFAULT_SERVICE}]", file=sys.stderr)
        default = next(m for m in MODELS if m["service"] == DEFAULT_SERVICE)
        try:
            switch_to(default["service"], default["port"])
        except Exception as e:
            print(f"[warning: failed to confirm default model restored: {e}]", file=sys.stderr)


if __name__ == "__main__":
    main()
