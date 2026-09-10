#!/usr/bin/env python3
"""Convert a PDF to Markdown using Docling, extracting figures and tables as
images. Run in the `pdfextract` conda env.

Prints the Markdown (with references to the saved images) to stdout.
Images are saved to <pdf-stem>_images/figure-N.png and table-N.png next to
the PDF (or in --image-dir if given), and their paths are also printed to
stderr as "IMAGE: <path>" lines.
"""
import argparse
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--image-dir", type=Path, default=None)
    args = parser.parse_args()

    image_dir = args.image_dir or args.pdf.with_name(args.pdf.stem + "_images")
    image_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(str(args.pdf))
    doc = result.document

    fig_i, tbl_i = 0, 0
    for item, _level in doc.iterate_items():
        if isinstance(item, PictureItem) and item.image is not None:
            fig_i += 1
            path = image_dir / f"figure-{fig_i}.png"
            item.image.pil_image.save(path)
            print(f"IMAGE: {path}", file=sys.stderr)
        elif isinstance(item, TableItem) and item.image is not None:
            tbl_i += 1
            path = image_dir / f"table-{tbl_i}.png"
            item.image.pil_image.save(path)
            print(f"IMAGE: {path}", file=sys.stderr)

    print(f"[saved {fig_i} figure(s), {tbl_i} table image(s) to {image_dir}]", file=sys.stderr)
    sys.stdout.write(doc.export_to_markdown())


if __name__ == "__main__":
    main()
