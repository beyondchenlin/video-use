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
            {"type": "word", "text": "我们", "start": 0.0, "end": 1.2},
            {"type": "word", "text": "嗯", "start": 1.3, "end": 1.5},
            {"type": "word", "text": "开始", "start": 1.6, "end": 2.8},
            {"type": "word", "text": "继续", "start": 4.0, "end": 5.2},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 3)
        # no kept interval covers the filler span 1.3 to 1.5
        for start, end in intervals:
            self.assertFalse(start < 1.5 and end > 1.3)

    # adjacent words across a short gap merge into one interval
    def test_short_gap_merges(self):
        words = [
            {"type": "word", "text": "你好", "start": 0.0, "end": 1.0},
            {"type": "word", "text": "世界", "start": 1.05, "end": 2.0},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 1)

    # adjacent words across a long silence split
    def test_long_gap_splits(self):
        words = [
            {"type": "word", "text": "a", "start": 0.0, "end": 1.0},
            {"type": "word", "text": "b", "start": 2.5, "end": 3.5},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 2)

    # non word entries such as spacing are ignored
    def test_spacing_ignored(self):
        words = [
            {"type": "word", "text": "a", "start": 0.0, "end": 1.0},
            {"type": "spacing", "start": 1.0, "end": 2.0},
            {"type": "word", "text": "b", "start": 2.0, "end": 3.0},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 2)


# MIN_SEG keeps tiny fragments out of the edit and the doc stays in step
class MinSegTests(unittest.TestCase):
    # the constant matches the documented fast-route value
    def test_min_seg_matches_the_documented_value(self):
        self.assertEqual(auto_edl.MIN_SEG, 1.0)

    # an isolated range shorter than MIN_SEG is absorbed into its neighbor
    def test_short_fragment_is_absorbed(self):
        words = [
            {"type": "word", "text": "前面", "start": 0.0, "end": 2.0},
            {"type": "word", "text": "很短", "start": 2.5, "end": 2.75},
        ]
        intervals = auto_edl.build_intervals(words)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0][1], 2.81)


# build_edl returns a valid single source edl payload or fails loudly
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

    # a transcript with only fillers raises instead of returning an empty edl
    def test_all_fillers_raises(self):
        transcript = {
            "words": [
                {"type": "word", "text": "嗯", "start": 0.0, "end": 0.3},
                {"type": "word", "text": "呃", "start": 0.4, "end": 0.7},
            ]
        }
        with self.assertRaisesRegex(ValueError, "no usable ranges"):
            auto_edl.build_edl(transcript, "S", "/src/in.mp4")

    # a transcript without word timings raises the same guard
    def test_no_words_raises(self):
        with self.assertRaisesRegex(ValueError, "no usable ranges"):
            auto_edl.build_edl({"words": []}, "S", "/src/in.mp4")


if __name__ == "__main__":
    unittest.main()
