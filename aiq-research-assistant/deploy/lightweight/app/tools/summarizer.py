"""
Summarizer tool: condenses given context into concise key points.
"""
from typing import List, Dict
import traceback


def summarize(context: str, max_points: int = 3, max_chars_per_point: int = 150) -> List[str]:
    """Summarize context into concise key points.
    
    Args:
        context: Raw text to summarize.
        max_points: Maximum number of key points to extract.
        max_chars_per_point: Maximum characters per key point.
    
    Returns:
        List of key point strings (Japanese or mixed language).
    """
    if not context or not isinstance(context, str):
        return []
    
    try:
        # Split context into lines, filter empty lines
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        
        if not lines:
            return []
        
        # Take first max_points lines as key points
        key_points = []
        for line in lines[:max_points]:
            # Truncate to max length if needed
            if len(line) > max_chars_per_point:
                truncated = line[:max_chars_per_point].rsplit(' ', 1)[0] + '...'
            else:
                truncated = line
            key_points.append(truncated)
        
        return key_points
    except Exception:
        traceback.print_exc()
        return []


def summarize_with_llm(context: str, query: str = "", llm_client=None) -> Dict:
    """Summarize context using an LLM (if client provided).
    
    Falls back to simple line-based summarization if LLM not available.
    
    Args:
        context: Text to summarize.
        query: Optional query hint for the LLM.
        llm_client: OpenAI-compatible client (optional).
    
    Returns:
        Dict with 'summary' (str) and 'key_points' (List[str]).
    """
    if llm_client is None:
        # Fallback to simple summarization
        key_points = summarize(context, max_points=3, max_chars_per_point=150)
        return {
            'summary': '\n'.join(key_points),
            'key_points': key_points,
            'source': 'local'
        }
    
    try:
        import json
        prompt = f"""以下のコンテキストを3つの要点に要約してください。各要点は1-2文で簡潔に。
クエリ: {query if query else "なし"}

コンテキスト:
{context}

JSON形式で返してください:
{{"summary": "全体の要約", "key_points": ["要点1", "要点2", "要点3"]}}
"""
        completion = llm_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct-v0.3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512
        )
        
        text = completion.choices[0].message.content.strip()
        # Find JSON in response
        first = text.find('{')
        if first != -1:
            text_json = text[first:]
        else:
            text_json = text
        
        result = json.loads(text_json)
        return {
            'summary': result.get('summary', ''),
            'key_points': result.get('key_points', []),
            'source': 'llm'
        }
    except Exception:
        traceback.print_exc()
        # Fallback to simple summarization
        key_points = summarize(context, max_points=3, max_chars_per_point=150)
        return {
            'summary': '\n'.join(key_points),
            'key_points': key_points,
            'source': 'local_fallback'
        }
