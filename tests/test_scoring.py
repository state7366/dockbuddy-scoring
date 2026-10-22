# -*- coding: utf-8 -*-
"""test_scoring.py —— 轻量测试(不联网、不加载大模型)。"""
import unittest

from dockbuddy_scoring.foldx_engine import FoldXError
from dockbuddy_scoring.msa_stats import position_freqs


class TestMsaStats(unittest.TestCase):
    def test_position_freqs(self):
        freqs = position_freqs(["AAAA", "AACC"], 4)
        self.assertEqual(len(freqs), 4)
        # 位点0: A=1.0
        self.assertAlmostEqual(freqs[0].get("A", 0.0), 1.0)
        # 位点2: A=0.5, C=0.5
        self.assertAlmostEqual(freqs[2].get("A", 0.0), 0.5)
        self.assertAlmostEqual(freqs[2].get("C", 0.0), 0.5)


class TestFoldXEngine(unittest.TestCase):
    def test_foldx_error(self):
        e = FoldXError("test")
        self.assertIsInstance(e, RuntimeError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
