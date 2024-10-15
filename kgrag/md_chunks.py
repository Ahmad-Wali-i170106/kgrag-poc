import json
import re
import tempfile
from itertools import chain
from pathlib import Path
from typing import Iterable

import numpy as np
import pymupdf4llm
from loguru import logger

MIN_CHUNK_SIZE = 400


def flatten(o: Iterable):
    for item in o:
        if isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def resolve_data_path(
    data_path: list[str | Path] | str | Path, file_extension: str | None = None
) -> chain:
    if not isinstance(data_path, list):
        data_path = [data_path]
    paths = []
    for dp in flatten(data_path):
        if isinstance(dp, (str, Path)):
            dp = Path(dp)
            if not dp.exists():
                raise Exception(f"Path {dp} does not exist.")
            if dp.is_dir():
                if file_extension:
                    paths.append(dp.glob(f"*.{file_extension}"))
                else:
                    paths.append(dp.iterdir())
            else:
                if file_extension is None or dp.suffix == f".{file_extension}":
                    paths.append([dp])
    return chain(*paths)


def docs_to_md(docs: list[str | Path] | str | Path) -> list[str]:
    docs_md = []
    for doc in resolve_data_path(data_path=docs):
        doc = Path(doc)
        if not doc.exists():
            md = str(doc)
        elif doc.suffix in [".md", ".txt"]:
            md = doc.read_text()
        elif doc.suffix == ".pdf":
            with tempfile.TemporaryDirectory() as image_folder:
                md = pymupdf4llm.to_markdown(
                    doc=str(doc),
                    write_images=True,
                    image_path=image_folder,
                    table_strategy="lines",
                )
        else:
            md = str(doc)
        docs_md.append(md)
    return docs_md


def chunk_md(
    text: str, separators: list[str] | None = None, min_chunk_size: int = MIN_CHUNK_SIZE
) -> dict[int, dict]:
    separators = separators or [r"#{1,6}\s+.+", r"\*\*.*?\*\*", r"---{3,}"]
    pattern = f'({"|".join(separators)})'
    splits = [chunk.strip() for chunk in re.split(pattern, text) if chunk.strip()]
    chunks = []
    current_chunk = ""
    start = 0
    for split in splits:
        if re.match(pattern, split) and current_chunk:
            end = min(len(text), start + len(current_chunk))
            chunks.append({"start": start, "end": end, "text": current_chunk})
            current_chunk = ""
            start = end
        current_chunk += split
    if current_chunk:
        end = min(len(text), start + len(current_chunk))
        chunks.append({"start": start, "end": end, "text": current_chunk})
    final_chunks: dict[int, dict] = {0: chunks[0]}
    for chunk in chunks[1:]:
        if len(chunk["text"]) > min_chunk_size:
            final_chunks[len(final_chunks)] = chunk
        else:
            final_chunks[len(final_chunks) - 1]["end"] = chunk["end"]
            final_chunks[len(final_chunks) - 1]["text"] += " " + chunk["text"]
    return final_chunks


if __name__ == "__main__":
    md = docs_to_md("/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/kosmos.pdf")[0]
    chunks = chunk_md(text=md)
    lens = [len(chunk["text"]) for chunk in chunks.values()]
    logger.info(f"Total document length: {len(md)}")
    logger.info(f"Num chunks: {len(chunks)}")
    logger.info(f"Max chunk size: {max(lens)}")
    logger.info(f"Min chunk size: {min(lens)}")
    logger.info(f"Average chunk size: {np.mean(lens):.2f}")
    Path("/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/kosmos_chunks.json").write_text(json.dumps(chunks))
