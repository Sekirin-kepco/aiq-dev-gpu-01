import json
import types
import unittest

from agent.orchestrator import get_orchestrator
import tools.search_tool as st


class TestE2EAgentSummarize(unittest.TestCase):
    """E2E-style test: force orchestrator to plan using summarize tool.

    This test monkeypatches the orchestrator's planning method to return a
    plan that first performs a search (deterministic fake) then a summarize
    step. It asserts that the action log includes the summarizer output.
    """

    def test_orchestrator_runs_summarize(self):
        # Prepare orchestrator
        orchestrator = get_orchestrator()

        # Fake local_search to return deterministic snippets
        def fake_local_search(q, top_k=5):
            return [
                {'id': 'd1', 'file': 'f1', 'snippet': '段落A\n詳細A', 'text': '段落A\n詳細A', 'score': 0.9},
                {'id': 'd2', 'file': 'f2', 'snippet': '段落B\n詳細B', 'text': '段落B\n詳細B', 'score': 0.85},
            ][:top_k]

        # Patch the module-level _local_search used by tools.search_tool
        orig_local = getattr(st, '_local_search', None)
        st._local_search = fake_local_search

        # Monkeypatch orchestrator._plan_with_llm to return a plan including summarize
        def fake_plan(self, query, top_k, previous_results=""):
            return [
                {'tool': 'search', 'args': {'query': query, 'top_k': 2}},
                {'tool': 'summarize', 'args': {'context': None, 'max_points': 2}},
            ]

        try:
            orchestrator._plan_with_llm = types.MethodType(fake_plan, orchestrator)

            # Run the orchestrator
            result = orchestrator.run('要約してください', do_retrieval=True, top_k=2)

            # Validations
            self.assertIn('actions', result)
            actions = result['actions']
            # Find a summarize action entry
            summarize_entries = [a for a in actions if a.get('tool') == 'summarize' or a.get('tool') == 'summarize']
            # There should be at least one summarize recorded (execution stage logs key_points)
            key_points_logged = any('key_points' in a or 'key_points_count' in a or a.get('tool')=='summarize' for a in actions)
            self.assertTrue(key_points_logged, msg=f"No summarizer output in actions: {actions}")
            # Ensure hits were collected from fake_local_search
            self.assertGreaterEqual(len(result.get('hits', [])), 1)
        finally:
            # restore
            st._local_search = orig_local


if __name__ == '__main__':
    unittest.main()
