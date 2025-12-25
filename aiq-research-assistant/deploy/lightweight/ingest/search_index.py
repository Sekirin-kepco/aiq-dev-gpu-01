import os
import json
from pathlib import Path
from typing import List, Dict, Any
import re

import numpy as np

try:
    import faiss
    USE_FAISS = True
except Exception:
    faiss = None
    USE_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

import hashlib
import math


INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/home/ec2-user/aiq/aiq-research-assistant/deploy/lightweight/index_data"))


# Japanese common keywords for filtering and boosting
KEYWORD_MAP = {
    "地震": ["地震応答", "地震動", "地震解析", "耐震"],
    "補助建屋": ["補助", "建屋", "A/B"],
    "燃料": ["燃料取扱", "FH/B"],
    "制御": ["制御建屋", "C/B"],
    "中間": ["中間建屋", "I/B"],
    "ディーゼル": ["ディーゼル", "DG/B"],
    "応答": ["応答解析", "応答値", "応答加速度"],
    "加速度": ["加速度", "最大加速度", "cm/s"],
    "変位": ["変位", "最大変位", "変形"],
    "解析": ["解析", "応答解析", "動的解析"],
    "耐力": ["耐力", "必要保有水平耐力"],
}


def _extract_keywords(query: str) -> List[str]:
    """Extract Japanese keywords from query for relevance boosting."""
    keywords = []
    query_lower = query.lower()
    
    # Direct keyword matching
    for keyword, variants in KEYWORD_MAP.items():
        for variant in variants:
            if variant in query:
                keywords.append(keyword)
                break
    
    return list(set(keywords))  # Remove duplicates


def _embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(model_name)
            q_emb = model.encode([query], convert_to_numpy=True)
            return q_emb
        except Exception:
            pass
    # Fallback deterministic hash-based vector (must match ingest fallback)
    h = hashlib.sha256(query.encode("utf-8")[:1024]).digest()
    vec = [((b % 128) - 64) / 64.0 for b in h]
    dim = 128
    if len(vec) < dim:
        repeats = math.ceil(dim / len(vec))
        vec = (vec * repeats)[:dim]
    q_emb = np.array([vec], dtype="float32")
    return q_emb


def _boost_scores_by_keywords(results: List[Dict[str, Any]], query_keywords: List[str]) -> List[Dict[str, Any]]:
    """Boost scores for results that match query keywords."""
    if not query_keywords:
        return results
    
    boosted = []
    for result in results:
        boost_factor = 1.0
        snippet = result.get("snippet", "") or result.get("text", "")
        
        # Check if keywords appear in snippet
        for keyword in query_keywords:
            if keyword in snippet:
                boost_factor *= 0.95  # Lower score (better) for keyword matches
        
        result["score"] = result.get("score", 0) * boost_factor
        result["keyword_boost"] = boost_factor
        boosted.append(result)
    
    # Re-sort by boosted score
    boosted.sort(key=lambda x: x["score"])
    return boosted


def _load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    if not meta_path.exists():
        return []
    return [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines()]


def search(query: str, top_k: int = 5, model_name: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """Enhanced search with keyword boosting and multi-strategy retrieval.
    
    Args:
        query: Search query in any language (preferably Japanese)
        top_k: Number of top results to return
        model_name: Embedding model name
    
    Returns:
        List of search results with scores, sorted by relevance
    """
    idx_path = INDEX_DIR / "faiss_index.index"
    meta_path = INDEX_DIR / "metadata.jsonl"
    emb_path = INDEX_DIR / "embeddings.npy"

    # Extract keywords for boosting
    keywords = _extract_keywords(query)
    
    q_emb = _embed_query(query, model_name=model_name)
    metas = _load_metadata(meta_path)

    results: List[Dict[str, Any]] = []
    
    if USE_FAISS and idx_path.exists():
        index = faiss.read_index(idx_path.as_posix())
        # Retrieve more results initially to allow for keyword re-ranking
        initial_k = min(top_k * 2, len(metas))
        D, I = index.search(q_emb, initial_k)
        
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(metas):
                continue
            m = metas[idx].copy()
            m["score"] = float(score)
            results.append(m)
        
        # Apply keyword boosting
        results = _boost_scores_by_keywords(results, keywords)
        results = results[:top_k]
        return results

    # fallback: use embeddings.npy and compute L2 distances
    if emb_path.exists():
        embs = np.load(emb_path.as_posix())
        diffs = embs - q_emb
        dists = np.sum(diffs * diffs, axis=1)
        initial_k = min(top_k * 2, len(metas))
        idxs = np.argsort(dists)[:initial_k]
        
        for idx in idxs:
            m = metas[int(idx)].copy()
            m["score"] = float(dists[int(idx)])
            results.append(m)
        
        # Apply keyword boosting
        results = _boost_scores_by_keywords(results, keywords)
        results = results[:top_k]
    
    return results

