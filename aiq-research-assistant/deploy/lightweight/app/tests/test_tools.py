"""
Unit tests for tools module.
"""
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.summarizer import summarize
from tools.search_tool import search


class TestSummarizer(unittest.TestCase):
    """Test cases for the summarizer tool."""
    
    def test_summarize_basic(self):
        """Test basic summarization with newline-separated text."""
        context = "最初の行です。\n2番目の行です。\n3番目の行です。\n4番目の行です。"
        result = summarize(context, max_points=3)
        self.assertEqual(len(result), 3)
        self.assertIn("最初の行です。", result[0])
        self.assertIn("2番目の行です。", result[1])
    
    def test_summarize_empty_context(self):
        """Test summarization with empty context."""
        result = summarize("", max_points=3)
        self.assertEqual(len(result), 0)
    
    def test_summarize_none_context(self):
        """Test summarization with None context."""
        result = summarize(None, max_points=3)
        self.assertEqual(len(result), 0)
    
    def test_summarize_fewer_lines_than_max_points(self):
        """Test when context has fewer lines than max_points."""
        context = "最初の行。\n2番目の行。"
        result = summarize(context, max_points=5)
        self.assertEqual(len(result), 2)
    
    def test_summarize_single_line(self):
        """Test with single-line context."""
        context = "これは1行のみのコンテキストです。"
        result = summarize(context, max_points=3)
        self.assertEqual(len(result), 1)
    
    def test_summarize_truncation(self):
        """Test that long lines are truncated."""
        long_line = "a" * 200
        lines = [long_line, "b" * 100, "c" * 50]
        context = "\n".join(lines)
        result = summarize(context, max_points=3, max_chars_per_point=150)
        # Check that first line is truncated
        self.assertLess(len(result[0]), len(long_line))
        self.assertTrue(result[0].endswith("..."))


class TestSearchTool(unittest.TestCase):
    """Test cases for the search tool."""
    
    def test_search_returns_list(self):
        """Test that search returns a list."""
        result = search("テスト", top_k=3)
        self.assertIsInstance(result, list)
    
    def test_search_result_has_expected_keys(self):
        """Test that search results have expected keys."""
        result = search("テスト", top_k=1)
        if result:
            for item in result:
                self.assertIn('id', item)
                self.assertIn('file', item)
                self.assertIn('snippet', item)
                self.assertIn('text', item)
                self.assertIn('score', item)

    def test_search_cache_behavior(self):
        """Test that search_tool caches underlying local_search calls."""
        import tools.search_tool as st

        # backup original local_search
        orig = getattr(st, '_local_search', None)

        call_counter = {'n': 0}

        def fake_local_search(q, top_k=5):
            call_counter['n'] += 1
            return [
                {'id': 'd1', 'file': 'f1', 'snippet': 's1', 'text': 's1', 'score': 0.9},
                {'id': 'd2', 'file': 'f2', 'snippet': 's2', 'text': 's2', 'score': 0.8},
            ][:top_k]

        try:
            st._local_search = fake_local_search
            # reset internal cache/counters if present
            if hasattr(st, '_CACHE'):
                st._CACHE.clear()
            if hasattr(st, '_CACHE_ORDER'):
                st._CACHE_ORDER.clear()
            if hasattr(st, '_call_count'):
                st._call_count = 0

            r1 = st.search('foo', top_k=2)
            self.assertEqual(st._call_count, 1)
            r2 = st.search('foo', top_k=2)
            # call_count should not increase due to cache
            self.assertEqual(st._call_count, 1)
            self.assertEqual(r1, r2)

            # different top_k triggers new underlying call
            r3 = st.search('foo', top_k=1)
            self.assertEqual(st._call_count, 2)
        finally:
            # restore original
            st._local_search = orig


if __name__ == '__main__':
    unittest.main()
