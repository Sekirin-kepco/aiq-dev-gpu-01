"""Minimal mock of the aiq_aira package used for local smoke tests.
This provides a small AiraClient with a from_env initializer and an answer() method.
Replace with the real package in production or when installing the real `aiq_aira`.
"""

from .client import AiraClient

__all__ = ["AiraClient"]
