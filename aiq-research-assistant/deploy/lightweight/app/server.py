from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from werkzeug.utils import safe_join
import os
import traceback
import json
from openai import OpenAI
from agent.orchestrator import get_orchestrator

# Try to import local search helper (faiss or numpy fallback)
try:
    # common absolute import if `ingest` is mounted into /app/ingest
    from ingest.search_index import search as local_search
except Exception:
    try:
        from ..ingest.search_index import search as local_search
    except Exception:
        # fallback import path if executed from this directory directly
        try:
            from deploy.lightweight.ingest.search_index import search as local_search
        except Exception:
            local_search = None

app = Flask(__name__, static_folder='static', static_url_path='/static')


# Small helper to generate an answer using the NVIDIA Mistral API
# or a local fallback generator. This keeps generation logic isolated so it can
# be swapped for other LLMs later.
def generate_answer(query: str, context: str | None = None) -> dict:
    """Generate an answer for `query` optionally using `context`.

    Uses NVIDIA Mistral API (OpenAI-compatible) as primary backend.
    Falls back to local generation if API fails or is not configured.

    Returns a dict that will be JSON-serializable and can include fields
    like `answer`, `key_points`, `context_preview`, and `source`.
    """
    llm_backend = os.environ.get('LLM_BACKEND', 'nvidia_mistral')
    
    # Try NVIDIA Mistral API if backend is enabled and API key is set
    if llm_backend == 'nvidia_mistral':
        nvidia_api_key = os.environ.get('NVIDIA_API_KEY')
        if nvidia_api_key:
            try:
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=nvidia_api_key
                )
                
                # Build prompt with context (Japanese language)
                if context:
                    prompt = f"""あなたは有用な研究アシスタントです。提供されたコンテキストに基づいてユーザーのクエリに答えてください。

コンテキスト:
{context}

ユーザーのクエリ: {query}

上記のコンテキストに基づいて、明確かつ簡潔な回答を日本語のみで提供してください。英語の翻訳は不要です。"""
                else:
                    prompt = f"ユーザーのクエリ: {query}\n\n有用な回答を日本語のみで提供してください。英語の翻訳は不要です。"
                
                # Call Mistral API
                completion = client.chat.completions.create(
                    model="mistralai/mistral-7b-instruct-v0.3",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024
                )
                
                answer = completion.choices[0].message.content
                
                # Remove English translations and extra metadata if present
                # Multiple patterns: "Translation: ...", "(Translation: ...", "(Translated response: ...", etc.
                translation_patterns = [
                    'Translation:',
                    '(Translation:',
                    '(Translated response:',
                    'Translated:',
                    '英語:',  # Japanese label for English
                ]
                for pattern in translation_patterns:
                    if pattern in answer:
                        answer = answer.split(pattern)[0].strip()
                        break
                
                # Also strip any trailing parentheses or brackets
                answer = answer.rstrip()
                
                # Extract key points from context if available
                if context:
                    lines = [line.strip() for line in context.split('\n') if line.strip()]
                    key_points = lines[:3] if lines else []
                    context_preview = context[:200] + "..." if len(context) > 200 else context
                    context_length = len(context)
                else:
                    key_points = []
                    context_preview = ""
                    context_length = 0
                
                return {
                    'answer': answer,
                    'key_points': key_points,
                    'source': 'rag+nvidia_mistral' if context else 'nvidia_mistral',
                    'context_length': context_length,
                    'context_preview': context_preview
                }
            except Exception as e:
                # Log error and fall through to local generator
                print(f"Mistral API error: {e}")
                traceback.print_exc()
    
    # Local RAG-aware fallback: produce a structured response using retrieved context.
    # This generates a natural-language summary with key points extracted from
    # the context, suitable for lightweight deployment without external LLMs.
    if context:
        # Split context into lines/chunks for analysis
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        # Extract key points (first 3 non-empty lines / chunks as representative snippets)
        key_points = lines[:3] if lines else []
        
        # Build summary: combine query intent with context hints
        summary_parts = [
            f"Based on the retrieved documents, here is information related to '{query}':"
        ]
        if key_points:
            summary_parts.append("Key findings from the documents:")
            for i, point in enumerate(key_points, 1):
                # Truncate long points for readability
                truncated = point[:100] + "..." if len(point) > 100 else point
                summary_parts.append(f"  {i}. {truncated}")
        
        summary = "\n".join(summary_parts)
        
        return {
            'answer': summary,
            'key_points': key_points[:3],
            'source': 'rag+local',
            'context_length': len(context),
            'context_preview': context[:200] + "..." if len(context) > 200 else context
        }
    
    # If no context, provide a simple response indicating lack of source material
    return {
        'answer': f"Query: '{query}' — No relevant documents found. Please try a different search term.",
        'key_points': [],
        'source': 'fallback'
    }


def normalize_hits(raw_hits):
    """Normalize various hit shapes into a standard dict shape.

    Desired output keys: id, file, page, snippet, text, score, source
    """
    normalized = []
    if not raw_hits:
        return normalized

    for i, h in enumerate(raw_hits):
        nh = {'id': None, 'file': None, 'page': None, 'snippet': None, 'text': None, 'score': None, 'source': None}
        if isinstance(h, dict):
            nh['id'] = h.get('id') or h.get('doc_id') or h.get('uid') or f'hit-{i}'
            nh['file'] = h.get('file') or h.get('source') or h.get('path')
            nh['page'] = h.get('page') or h.get('p')
            nh['snippet'] = h.get('snippet') or h.get('text') or h.get('content')
            nh['text'] = nh['snippet']
            # Try common score keys
            nh['score'] = h.get('score') or h.get('sim') or h.get('distance')
            nh['source'] = h.get('file') or h.get('source') or 'local'
        else:
            nh['id'] = f'hit-{i}'
            nh['text'] = str(h)
            nh['snippet'] = str(h)
            nh['source'] = 'local'
        normalized.append(nh)
    return normalized


def needs_retrieval(query: str) -> bool:
    """Decide whether the incoming `query` requires retrieval from the document
    index. Return True if retrieval should be performed, False if the query is
    likely chit-chat or otherwise doesn't need document context.

    Heuristics used:
    - Treat common Japanese greetings and short social phrases as NO-RETRIEVAL.
    - If the query contains explicit question words or interrogative particles
      (e.g. 何, どこ, いつ, どう, なぜ, -とは, 教えて, 要約 等) => perform retrieval.
    - If query contains a question mark (？ or ?) or ends with 'か' (common in
      Japanese questions), prefer retrieval.
    - Very short queries (<=4 characters) without question words are
      treated as chit-chat and do NOT trigger retrieval.
    """
    if not query or not isinstance(query, str):
        return False

    q = query.strip()
    # Common greeting/social phrases that do not need retrieval
    greetings = [
        "こんにちは", "おはよう", "こんばんは", "どうも", "ありがとう",
        "お疲れ様", "おつかれ", "よろしく", "はじめまして", "元気"
    ]
    for g in greetings:
        if g in q:
            return False

    # If it contains explicit interrogative words or verbs that imply information need
    interrogatives = [
        "何", "なに", "どこ", "いつ", "どう", "なぜ", "理由", "誰", "だれ",
        "どの", "どれ", "教え", "説明", "要約", "まとめ", "とは", "定義", "意味"
    ]
    for w in interrogatives:
        if w in q:
            return True

    # Punctuation cues
    if '？' in q or '?' in q:
        return True

    # Typical Japanese question ending 'か' (with some padding) -> treat as question
    if q.endswith('か') or q.endswith('か？') or q.endswith('か?'):
        return True

    # Short inputs (very short) without question words -> likely chit-chat
    if len(q) <= 4:
        return False

    # Default: perform retrieval for longer inputs
    return True

# Try to import the real aira client. If not present, keep the placeholder behavior.
# For lightweight deployment, always prefer local generation unless explicitly enabled.
AIRA_AVAILABLE = False
aira_client = None
# Disable aira_client initialization to always use improved local generation.
# To re-enable real aira client, set environment variable: ENABLE_AIRA_CLIENT=1
ENABLE_AIRA_CLIENT = os.environ.get('ENABLE_AIRA_CLIENT', '0') == '1'

if ENABLE_AIRA_CLIENT:
    try:
        # The real package name and client class may differ; try common possibilities
        try:
            # preferred import when installed as package
            from aiq_aira.client import AiraClient as _AiraClient
        except Exception:
            try:
                from aiq_aira import AiraClient as _AiraClient
            except Exception:
                _AiraClient = None

        if _AiraClient is not None:
            # Initialize from environment if the package exposes such helper, else construct default
            try:
                if hasattr(_AiraClient, 'from_env'):
                    aira_client = _AiraClient.from_env()
                else:
                    aira_client = _AiraClient()
                AIRA_AVAILABLE = True
            except Exception:
                # keep fallback if initialization fails
                aira_client = None
                AIRA_AVAILABLE = False
    except Exception:
        AIRA_AVAILABLE = False


@app.route('/', methods=['GET'])
def index():
    """Serve the chat UI"""
    return send_file('static/index.html')


@app.route('/docs/<path:filename>', methods=['GET'])
def serve_doc(filename: str):
    """Serve files from the mounted `sample_docs` directory for debugging/testing.

    Note: In production you should protect or proxy access to documents appropriately.
    """
    docs_dir = os.path.join(os.getcwd(), 'sample_docs')
    # Ensure path traversal is not possible
    try:
        safe_path = safe_join(docs_dir, filename)
    except Exception:
        abort(404)
    if not os.path.exists(safe_path):
        abort(404)
    # send_from_directory will set correct headers for common file types
    return send_from_directory(docs_dir, filename, as_attachment=False)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'aira_available': AIRA_AVAILABLE}), 200


@app.route('/internal/debug', methods=['GET'])
def internal_debug():
    info = {'local_search_present': local_search is not None}
    # Try to probe ingest.search_index module paths
    try:
        import importlib
        mod = importlib.import_module('ingest.search_index')
        idx = getattr(mod, 'INDEX_DIR', None)
        info['ingest_index_dir'] = str(idx) if idx is not None else None
    except Exception as e:
        info['ingest_probe_error'] = str(e)
    return jsonify(info), 200


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(silent=True) or {}
    q = data.get('q', '')
    # Decide if this query needs retrieval (avoid searching for greetings/chit-chat)
    do_retrieval = needs_retrieval(q)

    orchestrator = get_orchestrator()

    # If local index is configured, we may want to control top_k before calling orchestrator
    model_backend = os.environ.get('MODEL_BACKEND', '').lower()
    top_k = None
    if do_retrieval and model_backend in ('faiss_local', 'local_index') and local_search is not None:
        query_length = len(q.split())
        if query_length > 10:
            top_k = int(data.get('top_k', 8))
        elif query_length < 3:
            top_k = int(data.get('top_k', 3))
        else:
            top_k = int(data.get('top_k', 5))

    try:
        orchestration = orchestrator.run(q, do_retrieval, top_k=top_k or int(data.get('top_k', 3)))
        context = orchestration.get('context', '')
        gen = generate_answer(q, context=context)
        if not isinstance(gen, dict):
            gen = {'answer': str(gen)}
        gen['query'] = q
        gen['hits'] = orchestration.get('hits', [])
        gen['actions'] = orchestration.get('actions', [])
        gen['iterations'] = orchestration.get('iterations', 1)
        gen['source'] = 'agent_orchestrator'
        gen['retrieval_used'] = bool(do_retrieval)
        gen['top_k_used'] = top_k or int(data.get('top_k', 3)) if do_retrieval else 0
        return jsonify(gen), 200
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'query': q, 'answer': f'[orchestration error] {str(e)}', 'trace': tb}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
