#!/usr/bin/env python3
"""Simple ingest script: reads text/pdf files from a folder, chunks text, embeds with
sentence-transformers, builds a FAISS index and writes metadata.

Usage:
  python ingest_to_faiss.py --input-dir ../sample_docs --output-dir ../index_data

This is intentionally minimal for smoke tests.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import faiss
    USE_FAISS = True
except Exception:
    faiss = None
    USE_FAISS = False
from tqdm import tqdm
import hashlib
import math

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def extract_text_from_pdf(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF not available to read PDF files")
    doc = fitz.open(path.as_posix())
    pages = []
    for p in doc:
        pages.append(p.get_text())
    return "\n".join(pages)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def ingest(input_dir: Path, output_dir: Path, model_name: str = "all-MiniLM-L6-v2"):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(model_name)
        except Exception:
            model = None

    texts = []
    metas = []
    # iterate files
    for p in sorted(input_dir.iterdir()):
        if p.is_dir():
            continue
        if p.suffix.lower() in [".txt"]:
            text = read_text_file(p)
        elif p.suffix.lower() in [".pdf"]:
            text = extract_text_from_pdf(p)
        else:
            print(f"Skipping unsupported file: {p}")
            continue

        chunks = chunk_text(text)
        for idx, c in enumerate(chunks):
            metas.append({
                "doc_id": p.stem,
                "file": str(p.name),
                "chunk_idx": idx,
                "text": c,
            })
            texts.append(c)

    if not texts:
        print("No text chunks found. Exiting.")
        return

    print(f"Embedding {len(texts)} chunks with model {model_name}... (fallback if necessary)")
    if model is not None:
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
    else:
        # Fallback: simple hash-based embedding to get deterministic dense vectors for smoke tests
        dim = 128
        embeddings = []
        for t in texts:
            # create a dim-length vector from sha256 chunks
            h = hashlib.sha256(t.encode("utf-8")[:1024]).digest()
            vec = [((b % 128) - 64) / 64.0 for b in h]
            # if h shorter than dim, repeat
            if len(vec) < dim:
                repeats = math.ceil(dim / len(vec))
                vec = (vec * repeats)[:dim]
            else:
                vec = vec[:dim]
            embeddings.append(vec)
        embeddings = np.array(embeddings, dtype="float32")
    if USE_FAISS:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss_path = output_dir / "faiss_index.index"
        meta_path = output_dir / "metadata.jsonl"
        faiss.write_index(index, faiss_path.as_posix())

        with meta_path.open("w", encoding="utf-8") as fh:
            for m in metas:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"Wrote index to {faiss_path} and metadata to {meta_path}")
    else:
        emb_path = output_dir / "embeddings.npy"
        meta_path = output_dir / "metadata.jsonl"
        np.save(emb_path.as_posix(), embeddings)

        with meta_path.open("w", encoding="utf-8") as fh:
            for m in metas:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"Wrote embeddings to {emb_path} and metadata to {meta_path} (FAISS unavailable)")


def search(index_path: Path, meta_path: Path, query: str, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
    model = None
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(model_name)
        except Exception:
            model = None

    if model is not None:
        q_emb = model.encode([query], convert_to_numpy=True)
    else:
        # simple hash-based embedding for query (must match ingest fallback dim)
        import hashlib, math
        h = hashlib.sha256(query.encode("utf-8")[:1024]).digest()
        vec = [((b % 128) - 64) / 64.0 for b in h]
        dim = 128
        if len(vec) < dim:
            repeats = math.ceil(dim / len(vec))
            vec = (vec * repeats)[:dim]
        q_emb = np.array([vec], dtype="float32")

    metas = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines()]

    results = []
    if USE_FAISS:
        index = faiss.read_index(index_path.as_posix())
        D, I = index.search(q_emb, top_k)
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(metas):
                continue
            m = metas[idx]
            m_copy = m.copy()
            m_copy["score"] = float(score)
            results.append(m_copy)
    else:
        emb_path = index_path.with_name("embeddings.npy")
        embs = np.load(emb_path.as_posix())
        # compute L2 distances
        diffs = embs - q_emb
        dists = np.sum(diffs * diffs, axis=1)
        idxs = np.argsort(dists)[:top_k]
        for idx in idxs:
            m = metas[int(idx)]
            m_copy = m.copy()
            m_copy["score"] = float(dists[int(idx)])
            results.append(m_copy)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    ingest(inp, out, model_name=args.model)
    # quick search smoke test
    qres = search(out / "faiss_index.index", out / "metadata.jsonl", "research assistant", model_name=args.model, top_k=3)
    print("Search results (top 3):")
    for r in qres:
        print(r)


if __name__ == "__main__":
    main()
