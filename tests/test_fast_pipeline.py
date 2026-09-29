"""Tests for the fast pipeline sentence cues and output timeline mapping."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "helpers"))

import fast_pipeline


# build timed words with even spacing for the cue tests
def make_words(items):
    words = []
    t = 0.0
    for text in items:
        words.append(
            {"type": "word", "text": text, "start": round(t, 3), "end": round(t + 0.3, 3)}
        )
        t += 0.4
    return words


# sentence_cues prefers a punctuation boundary over a hard length cut
class SentenceCuesTests(unittest.TestCase):
    # a cap inside a clause extends forward to the next punctuation mark
    def test_extend_to_next_punctuation_instead_of_cutting(self):
        words = make_words([
            "一百零一", "到", "一百五，", "然后", "你", "选", "哪一个，",
            "尽量", "也", "让", "他", "写，", "让", "别的", "灰度",
            "我明白", "这个", "可以", "实现。",
        ])
        cues = fast_pipeline.sentence_cues(words, 0.6, 20)
        first_text = "".join(w["text"] for w in cues[0])
        self.assertTrue(first_text.endswith("写，"))
        self.assertNotEqual(first_text, "一百零一到一百五，然后你选哪一个，尽量也")

    # a very long clause still breaks when no punctuation is nearby
    def test_hard_cut_when_no_punctuation_is_near(self):
        words = make_words(["一二三四五"] * 12)
        cues = fast_pipeline.sentence_cues(words, 0.6, 20, max_overrun=4)
        self.assertGreater(len(cues), 1)
        longest = max(len("".join(w["text"] for w in cue)) for cue in cues)
        self.assertLessEqual(longest, 20 + 4 + 5)

    # a long gap still ends a cue without punctuation
    def test_long_gap_ends_a_cue(self):
        words = make_words(["你好", "世界"])
        words[1]["start"] = 5.0
        words[1]["end"] = 5.3
        cues = fast_pipeline.sentence_cues(words, 0.6, 20)
        self.assertEqual(len(cues), 2)


# map_cues_to_output follows Rule 5 and drops fillers the edl removed
class MapCuesToOutputTests(unittest.TestCase):
    # times shift by the segment offset and stay on the output timeline
    def test_rule_five_offsets(self):
        edl = {
            "sources": {"S": "x"},
            "ranges": [
                {"source": "S", "start": 10.0, "end": 14.0},
                {"source": "S", "start": 20.0, "end": 24.0},
            ],
        }
        transcript = {
            "words": [
                {"type": "word", "text": "先", "start": 10.5, "end": 10.9},
                {"type": "word", "text": "后", "start": 20.5, "end": 20.9},
            ]
        }
        cues = fast_pipeline.map_cues_to_output(transcript, edl)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0]["start"], 0.5, places=3)
        self.assertAlmostEqual(cues[1]["start"], 4.5, places=3)

    # fillers cut by the edl are not burned into the subtitle text
    def test_fillers_are_dropped(self):
        edl = {
            "sources": {"S": "x"},
            "ranges": [{"source": "S", "start": 0.0, "end": 5.0}],
        }
        transcript = {
            "words": [
                {"type": "word", "text": "嗯", "start": 0.4, "end": 0.6},
                {"type": "word", "text": "内容", "start": 1.0, "end": 1.4},
            ]
        }
        cues = fast_pipeline.map_cues_to_output(transcript, edl)
        text = "".join(c["text"] for c in cues)
        self.assertNotIn("嗯", text)
        self.assertIn("内容", text)


# compare_cues reports only the sentences where the two passes disagree
class CompareCuesTests(unittest.TestCase):
    # punctuation and spacing differences do not count as a disagreement
    def test_matching_passes_have_no_diffs(self):
        first = [{"start": 0.0, "end": 1.0, "text": "你好。"}]
        recheck = [{"start": 0.0, "end": 1.0, "text": "你好 "}]
        self.assertEqual(fast_pipeline.compare_cues(first, recheck), [])

    # an asr difference is returned with its output time and both texts
    def test_differing_sentence_is_reported(self):
        first = [{"start": 2.0, "end": 3.0, "text": "已拍未拍。"}]
        recheck = [{"start": 2.0, "end": 3.0, "text": "已拍会拍。"}]
        diffs = fast_pipeline.compare_cues(first, recheck)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["start"], 2.0)
        self.assertEqual(diffs[0]["first"], "已拍未拍。")
        self.assertEqual(diffs[0]["recheck"], "已拍会拍。")


# cjk_force_style keeps the safe margin by default and accepts an override
class CaptionStyleTests(unittest.TestCase):
    # the default margin stays at the vertical platform safe zone
    def test_default_margin(self):
        self.assertIn("MarginV=90", fast_pipeline.cjk_force_style())

    # an explicit margin is rendered into the style
    def test_custom_margin(self):
        self.assertIn("MarginV=60", fast_pipeline.cjk_force_style(60))


if __name__ == "__main__":
    unittest.main()
