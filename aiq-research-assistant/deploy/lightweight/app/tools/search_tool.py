from typing import List, Dict
import traceback
import os

# Use the same local_search used by the app; try common import paths
try:
    from ingest.search_index import search as _local_search
except Exception:
    try:
        from ..ingest.search_index import search as _local_search
    except Exception:
        try:
            from deploy.lightweight.ingest.search_index import search as _local_search
        except Exception:
            _local_search = None


import time

# Simple in-memory cache for search results (keyed by query + top_k)
# Each entry: { 'ts': float, 'value': List[Dict] }
_CACHE: Dict[str, Dict] = {}
_CACHE_ORDER: List[str] = []
_CACHE_MAX = 128
# Counter to help tests detect whether underlying search was invoked
_call_count = 0
# TTL in seconds (default 1 hour)
_CACHE_TTL = int(os.environ.get('SEARCH_CACHE_TTL', '3600'))


def search(query: str, top_k: int = 5) -> List[Dict]:
    """Wrapper around the local search implementation that returns a
    predictable list of dicts with keys: id, file, page, snippet, text, score.
    """
    if _local_search is None:
        return []
    try:
        # Check cache first
        key = f"{query}||{top_k}"
        if key in _CACHE:
            entry = _CACHE.get(key)
            if entry:
                age = time.time() - float(entry.get('ts', 0))
                if age <= _CACHE_TTL:
                    # return a shallow copy to avoid accidental mutation
                    return [dict(h) for h in entry.get('value', [])]
                else:
                    # expired
                    try:
                        _CACHE.pop(key, None)
                        _CACHE_ORDER.remove(key)
                    except Exception:
                        pass

        # Not cached (or expired): call underlying search
        global _call_count
        _call_count += 1
        raw = _local_search(query, top_k=top_k)
        normalized = []
        for i, h in enumerate(raw or []):
            if isinstance(h, dict):
                normalized.append({
                    'id': h.get('id') or h.get('doc_id') or h.get('uid') or f'hit-{i}',
                    'file': h.get('file') or h.get('source') or h.get('path'),
                    'page': h.get('page') or h.get('p'),
                    'snippet': h.get('snippet') or h.get('text') or h.get('content'),
                    'text': h.get('text') or h.get('snippet') or h.get('content'),
                    'score': h.get('score') or h.get('sim') or h.get('distance')
                })
            else:
                normalized.append({
                    'id': f'hit-{i}',
                    'file': None,
                    'page': None,
                    'snippet': str(h),
                    'text': str(h),
                    'score': None
                })
        # store into cache
        try:
            _store_in_cache(query, top_k, normalized)
        except Exception:
            # non-fatal
            pass
        return normalized
    except Exception:
        traceback.print_exc()
        return []


def _store_in_cache(query: str, top_k: int, normalized: List[Dict]):
    key = f"{query}||{top_k}"
    if key in _CACHE:
        return
    # Evict oldest if necessary
    if len(_CACHE_ORDER) >= _CACHE_MAX:
        old = _CACHE_ORDER.pop(0)
        _CACHE.pop(old, None)
    _CACHE_ORDER.append(key)
    # store a copy with timestamp
    _CACHE[key] = {'ts': time.time(), 'value': [dict(h) for h in normalized]}


def get_cache_stats() -> Dict:
    # compute expired count without removing
    now = time.time()
    expired = 0
    for k, v in list(_CACHE.items()):
        ts = float(v.get('ts', 0))
        if now - ts > _CACHE_TTL:
            expired += 1
    return {
        'entries': len(_CACHE),
        'expired': expired,
        'order_len': len(_CACHE_ORDER),
        'max': _CACHE_MAX,
        'call_count': _call_count,
        'ttl_seconds': _CACHE_TTL
    }


def clear_cache():
    _CACHE.clear()
    _CACHE_ORDER.clear()


def _cache_file_path() -> str | None:
    return os.environ.get('SEARCH_CACHE_FILE')


def save_cache_to_disk():
    """Persist current cache to JSON file if SEARCH_CACHE_FILE is set."""
    fp = _cache_file_path()
    if not fp:
        return False
    try:
        serial = {}
        for k, v in _CACHE.items():
            serial[k] = {'ts': float(v.get('ts', 0)), 'value': v.get('value', [])}
        d = {'meta': {'saved_at': time.time()}, 'cache': serial}
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False)
        return True
    except Exception:
        traceback.print_exc()
        return False


def load_cache_from_disk():
    fp = _cache_file_path()
    if not fp or not os.path.exists(fp):
        return False
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        c = d.get('cache', {})
        for k, v in c.items():
            _CACHE[k] = {'ts': float(v.get('ts', 0)), 'value': v.get('value', [])}
            if k not in _CACHE_ORDER:
                _CACHE_ORDER.append(k)
        return True
    except Exception:
        traceback.print_exc()
        return False


# Attempt to load cache on import if env var provided
try:
    if _cache_file_path():
        load_cache_from_disk()
except Exception:
    pass
