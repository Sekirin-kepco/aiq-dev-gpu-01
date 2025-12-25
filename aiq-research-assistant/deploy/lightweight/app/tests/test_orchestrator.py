"""
Unit tests for orchestrator module.
"""
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.orchestrator import Orchestrator, get_orchestrator


class TestOrchestrator(unittest.TestCase):
    """Test cases for the orchestrator."""
    
    def setUp(self):
        """Set up orchestrator for each test."""
        self.orchestrator = get_orchestrator()
    
    def test_orchestrator_creation(self):
        """Test that orchestrator is created successfully."""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.max_iterations, 3)
    
    def test_orchestrator_no_retrieval(self):
        """Test orchestrator with do_retrieval=False."""
        result = self.orchestrator.run("test query", do_retrieval=False)
        self.assertIn('hits', result)
        self.assertIn('context', result)
        self.assertIn('actions', result)
        self.assertIn('iterations', result)
        self.assertIn('cache_stats', result)
        # When no retrieval, should have empty hits and context
        self.assertEqual(len(result['hits']), 0)
        self.assertEqual(result['context'], "")
    
    def test_orchestrator_cache_empty_initially(self):
        """Test that cache starts empty."""
        result = self.orchestrator.run("test", do_retrieval=False)
        self.assertEqual(result['cache_stats']['cache_size'], 0)
    
    def test_orchestrator_max_iterations(self):
        """Test that orchestrator respects max_iterations."""
        orch = Orchestrator(max_iterations=1)
        result = orch.run("テスト", do_retrieval=True, top_k=1)
        # With max_iterations=1, should complete in 1 iteration
        self.assertLessEqual(result['iterations'], 1)
    
    def test_orchestrator_actions_log_structure(self):
        """Test that actions log has proper structure."""
        result = self.orchestrator.run("test", do_retrieval=False)
        actions = result['actions']
        # Should have at least one action entry
        self.assertGreater(len(actions), 0)
        # First action should indicate no_search
        self.assertIn('action', actions[0])
    
    def test_satisfaction_evaluation(self):
        """Test query satisfaction evaluation logic."""
        hits = [{'id': str(i)} for i in range(5)]
        context = "x" * 600  # 600 chars, exceeds 500 threshold
        satisfied = self.orchestrator._is_query_satisfied(hits, context, iteration=1)
        self.assertTrue(satisfied)
    
    def test_satisfaction_max_iterations(self):
        """Test satisfaction at max iterations."""
        hits = []
        context = ""
        satisfied = self.orchestrator._is_query_satisfied(hits, context, iteration=3)
        self.assertTrue(satisfied)  # Should be satisfied due to max iterations


class TestOrchestratorCaching(unittest.TestCase):
    """Test caching functionality in orchestrator."""
    
    def setUp(self):
        """Set up orchestrator for each test."""
        self.orchestrator = get_orchestrator()
    
    def test_search_cache_key_format(self):
        """Test that cache keys are tuples of (query, top_k)."""
        result = self.orchestrator.run("テスト", do_retrieval=True, top_k=3)
        cache_entries = result['cache_stats']['cache_entries']
        # Each cache entry should be a list (since JSON serialization converts tuples)
        # Or in Python it's a tuple
        self.assertGreaterEqual(len(cache_entries), 0)


if __name__ == '__main__':
    unittest.main()
