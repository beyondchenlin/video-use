"""Tests for auto_edl, the deterministic fast-route cut builder."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "helpers"))

import auto_edl


# is_filler recognizes standalone fillers in both languages
class IsFillerTests(unittest.TestCase):
    # a chinese filler character is removable
    def test_zh_filler(self):
        self.assertTrue(auto_edl.is_filler({"text": "嗯"}))

    # repeated filler characters are removable
    def test_zh_repeat(self):
        self.assertTrue(auto_edl.is_filler({"text": "嗯嗯"}))

    # punctuation around a filler is ignored
    def test_zh_punct(self):
        self.assertTrue(auto_edl.is_filler({"text": "嗯，"}))

    # english fillers match case insensitively
    def test_en_filler(self):
        self.assertTrue(auto_edl.is_filler({"text": "um"}))
        self.assertTrue(auto_edl.is_filler({"text": "UH"}))

    # real content words are kept
    def test_real_word(self):
        self.assertFalse(auto_edl.is_filler({"text": "我们"}))

    # laughter is kept because ha is not in the filler set
    def test_laugh_kept(self):
        self.assertFalse(auto_edl.is_filler({"text": "哈哈"}))


# build_intervals cuts fillers and long silences while merging short gaps
class BuildIntervalsTests(unittest.TestCase):
    # a filler word is excluded and splits the surrounding intervals
    def test_filler_removed(self):
        words = [
            {"type": "word", "text": "我们", "start": 0.0, "end": 0.3},
            {"type": "word", "text": "嗯", "start": 0.4, "end": 0.6},
            {"type": "word", "text": "开始", "start": 0.7, "end": 1.0},
            {"type": "word", "text": "继续", "start": 2.0, "end": 2.3},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 3)
        # no kept interval covers the filler span 0.4 to 0.6
        for start, end in intervals:
            self.assertFalse(start < 0.6 and end > 0.4)

    # adjacent words across a short gap merge into one interval
    def test_short_gap_merges(self):
        words = [
            {"type": "word", "text": "你好", "start": 0.0, "end": 0.3},
            {"type": "word", "text": "世界", "start": 0.35, "end": 0.7},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 1)

    # adjacent words across a long silence split
    def test_long_gap_splits(self):
        words = [
            {"type": "word", "text": "a", "start": 0.0, "end": 0.3},
            {"type": "word", "text": "b", "start": 1.5, "end": 1.8},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 2)

    # non word entries such as spacing are ignored
    def test_spacing_ignored(self):
        words = [
            {"type": "word", "text": "a", "start": 0.0, "end": 0.3},
            {"type": "spacing", "start": 0.3, "end": 0.9},
            {"type": "word", "text": "b", "start": 0.9, "end": 1.2},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 2)


# build_edl returns a valid single source edl payload
class BuildEdlTests(unittest.TestCase):
    # ranges and totals reflect the kept intervals
    def test_structure(self):
        transcript = {
            "words": [
                {"type": "word", "text": "你好", "start": 0.0, "end": 0.4},
                {"type": "word", "text": "世界", "start": 0.45, "end": 0.9},
            ]
        }
        edl = auto_edl.build_edl(transcript, "S", "/src/in.mp4")
        self.assertEqual(edl["sources"], {"S": "/src/in.mp4"})
        self.assertEqual(len(edl["ranges"]), 1)
        self.assertEqual(edl["ranges"][0]["source"], "S")
        self.assertGreater(edl["total_duration_s"], 0)


if __name__ == "__main__":
    unittest.main()
