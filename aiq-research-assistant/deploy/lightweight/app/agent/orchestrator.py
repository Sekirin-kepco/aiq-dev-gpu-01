from typing import Any, Dict, List
import os
import json
import traceback
from tools.search_tool import search as tool_search
from tools.summarizer import summarize


class Orchestrator:
    """Orchestrator with iterative LLM-based planning (Agent loop).

    Behavior:
    - When `do_retrieval` is True, iteratively:
      1. Call the LLM to produce a JSON plan (a list of actions).
      2. Execute each action (currently: `search`, `summarize`).
      3. Evaluate if the query is sufficiently answered.
      4. If not satisfied, generate a new plan based on current results.
    - Repeat until satisfied or max iterations reached.
    - Return full action history, aggregated hits, and synthesized context.
    - Maintains an in-memory cache of search results to avoid redundant calls.
    """
    def __init__(self, max_iterations: int = 3):
        # LLM client is created lazily
        self._client = None
        self.max_iterations = max_iterations
        # Simple in-memory cache for search results: key = (query, top_k), value = hits
        self._search_cache = {}

    def _get_client(self):
        if self._client is None:
            nvidia_api_key = os.environ.get('NVIDIA_API_KEY')
            if not nvidia_api_key:
                raise RuntimeError('NVIDIA_API_KEY not configured for planner')
            try:
                from openai import OpenAI
            except Exception as e:
                # Raise a clear error so callers can fallback; do not fail at import time
                raise RuntimeError('openai package not available for planner') from e
            self._client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_api_key)
        return self._client

    def _plan_with_llm(self, query: str, top_k: int, previous_results: str = "") -> List[Dict[str, Any]]:
        """Ask the LLM to produce a JSON plan for which tools to call.

        The model is asked to output a JSON object with a top-level `plan`
        key that is a list of actions. Each action is an object with
        `tool` (string) and `args` (object).
        
        If `previous_results` is provided, include it in the prompt so the LLM
        can see what has been retrieved so far and potentially refine the search.
        Example:
        {
          "plan": [
            {"tool":"search", "args": {"query":"原子炉とは何ですか？","top_k":3}}
          ]
        }
        """
        try:
            client = self._get_client()
            context_note = ""
            if previous_results:
                context_note = f"\n\n前回までの検索結果::\n{previous_results}\n\n上記の結果を踏まえて、さらに必要な情報を得るための計画を立ててください。"
            
            prompt = f"""あなたはツールを呼び出す計画を立てるアシスタントです。
入力クエリに対して、どのツールをどのような引数で順番に呼ぶかをJSONで返してください。

利用可能なツール:
- search(query, top_k): ドキュメント検索を実行します
- summarize(context, max_points): 与えられたコンテキストを要点に要約します

ツール使用のガイドライン:
- 最初は search で関連ドキュメントを探す
- search で多数の結果を得た場合（5ドキュメント以上）、続けて summarize でキーポイントを抽出
- 複雑なクエリの場合、複数の search ステップを組み合わせることもできます
- summarize は検索後のコンテキストサマリに有効です

出力形式（厳密なJSON）:
例1（単純検索）:
{{"plan": [{{"tool": "search", "args": {{"query": "原子炉", "top_k": 3}}}}]}}

例2（検索後に要約）:
{{"plan": [{{"tool": "search", "args": {{"query": "安全評価", "top_k": 5}}}}, {{"tool": "summarize", "args": {{"context": "previous_search_results", "max_points": 3}}}}]}}

Use Japanese for any explanatory text, but the returned value must be VALID JSON ONLY.

Query: {query}
Default top_k: {top_k}{context_note}
"""
            completion = client.chat.completions.create(
                model=os.environ.get('LLM_MODEL', 'mistralai/mistral-7b-instruct-v0.3'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512
            )
            text = completion.choices[0].message.content
            # Try to find JSON in response
            text_stripped = text.strip()
            # If model prepends explanation, try to locate first '{'
            first = text_stripped.find('{')
            if first != -1:
                text_json = text_stripped[first:]
            else:
                text_json = text_stripped
            plan_obj = json.loads(text_json)
            plan = plan_obj.get('plan') or []
            return plan
        except Exception:
            traceback.print_exc()
            return []

    def _is_query_satisfied(self, hits: List[Dict[str, Any]], context: str, iteration: int) -> bool:
        """Heuristic to evaluate if the query is sufficiently answered.
        
        Returns True if:
        - At least 3 high-quality hits, OR
        - Context length > 500 characters, OR
        - Reached max iterations (force exit)
        
        Otherwise returns False, prompting another iteration.
        """
        if iteration >= self.max_iterations:
            # Force stop at max iterations
            return True
        
        # If we have meaningful hits and context, consider satisfied
        if len(hits) >= 3 and len(context) >= 500:
            return True
        
        # If very minimal results but reached iteration 2+, stop
        if len(hits) >= 1 and iteration >= 2:
            return True
        
        return False

    def run(self, query: str, do_retrieval: bool = True, top_k: int = 5) -> Dict[str, Any]:
        """Run the agent loop: plan → execute → evaluate → repeat.
        
        Returns:
        {
            'hits': aggregated list of document hits,
            'context': joined context from all snippets,
            'actions': full action history (all iterations),
            'iterations': number of iterations performed
        }
        """
        actions_log: List[Dict[str, Any]] = []
        hits_all: List[Dict[str, Any]] = []
        context_parts: List[str] = []

        if not do_retrieval:
            actions_log.append({'iteration': 0, 'action': 'no_search'})
            return {
                'hits': [],
                'context': '',
                'actions': actions_log,
                'iterations': 1
            }

        # Simple tool registry
        tool_registry = {
            'search': tool_search,
            'summarize': summarize,
        }

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            actions_log.append({'iteration': iteration, 'phase': 'planning'})
            
            # Build summary of previous results for LLM context
            prev_summary = ""
            if hits_all:
                prev_summary = f"既に取得されたドキュメント数: {len(hits_all)}, コンテキスト文字数: {sum(len(p) for p in context_parts)}"
            
            # Ask LLM for a plan
            plan = self._plan_with_llm(query, top_k, previous_results=prev_summary)

            # If plan empty or invalid, fall back to a single search action
            if not plan:
                if iteration == 1:
                    # First iteration: use fallback
                    actions_log.append({'iteration': iteration, 'action': 'plan_failed', 'note': 'falling back to single search'})
                    plan = [{"tool": "search", "args": {"query": query, "top_k": top_k}}]
                else:
                    # Subsequent iterations: stop if planning fails
                    actions_log.append({'iteration': iteration, 'action': 'plan_failed', 'note': 'stopping iteration'})
                    break

            # Execute plan
            actions_log.append({'iteration': iteration, 'phase': 'execution', 'plan_size': len(plan)})
            for step_idx, step in enumerate(plan):
                try:
                    tool_name = step.get('tool') if isinstance(step, dict) else None
                    args = step.get('args', {}) if isinstance(step, dict) else {}
                    actions_log.append({
                        'iteration': iteration,
                        'step': step_idx,
                        'tool': tool_name,
                        'args': args
                    })
                    
                    if tool_name == 'search':
                        q = args.get('query', query)
                        tk = int(args.get('top_k', top_k))
                        
                        # Check cache first
                        cache_key = (q, tk)
                        if cache_key in self._search_cache:
                            res = self._search_cache[cache_key]
                            actions_log.append({
                                'iteration': iteration,
                                'step': step_idx,
                                'tool': 'search',
                                'result_count': len(res),
                                'cached': True
                            })
                        else:
                            # call the search tool and cache result
                            res = tool_registry['search'](q, top_k=tk)
                            self._search_cache[cache_key] = res
                            actions_log.append({
                                'iteration': iteration,
                                'step': step_idx,
                                'tool': 'search',
                                'result_count': len(res),
                                'cached': False
                            })
                        
                        # aggregate hits
                        for h in (res or []):
                            hits_all.append(h)
                            snippet = h.get('snippet') or h.get('text') or ''
                            if snippet:
                                context_parts.append(snippet)
                    elif tool_name == 'summarize':
                        ctx = args.get('context', '\n---\n'.join(context_parts[-5:]))  # Use last 5 parts if not specified
                        max_pts = int(args.get('max_points', 3))
                        # call the summarize tool
                        key_pts = tool_registry['summarize'](ctx, max_points=max_pts)
                        actions_log.append({
                            'iteration': iteration,
                            'step': step_idx,
                            'tool': 'summarize',
                            'key_points_count': len(key_pts),
                            'key_points': key_pts
                        })
                    else:
                        actions_log.append({
                            'iteration': iteration,
                            'step': step_idx,
                            'tool': tool_name,
                            'note': 'unsupported_tool'
                        })
                except Exception as e:
                    actions_log.append({
                        'iteration': iteration,
                        'step': step_idx,
                        'error': str(e)
                    })
                    traceback.print_exc()

            # Evaluate if query is satisfied
            context = '\n---\n'.join([p for p in context_parts if p])
            is_satisfied = self._is_query_satisfied(hits_all, context, iteration)
            actions_log.append({'iteration': iteration, 'phase': 'evaluation', 'satisfied': is_satisfied})
            
            if is_satisfied:
                break

        context = '\n---\n'.join([p for p in context_parts if p])
        
        # Cache statistics
        cache_stats = {
            'cache_size': len(self._search_cache),
            'cache_entries': list(self._search_cache.keys())
        }
        
        return {
            'hits': hits_all,
            'context': context,
            'actions': actions_log,
            'iterations': iteration,
            'cache_stats': cache_stats
        }


def get_orchestrator() -> Orchestrator:
    return Orchestrator()
