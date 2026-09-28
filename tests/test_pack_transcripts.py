"""Tests for pack_transcripts, focused on language-aware phrase joining."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "helpers"))

import pack_transcripts as pack


# join_words keeps adjacent chinese words together without spaces
class JoinWordsTests(unittest.TestCase):
    # chinese words concatenate with no spaces
    def test_chinese_no_spaces(self):
        self.assertEqual(pack.join_words(["今天", "给", "大家"]), "今天给大家")

    # english words stay space separated
    def test_english_spaces(self):
        self.assertEqual(pack.join_words(["hello", "world"]), "hello world")

    # punctuation already attached to a chinese word stays tight to the next word
    def test_attached_punctuation(self):
        self.assertEqual(pack.join_words(["工具，", "它"]), "工具，它")

    # standalone punctuation glues onto the previous word
    def test_standalone_punctuation(self):
        self.assertEqual(pack.join_words(["hello", ","]), "hello,")


# group_into_phrases renders readable phrases for both languages
class GroupPhrasesTests(unittest.TestCase):
    # chinese words inside one phrase join without spaces
    def test_chinese_phrase(self):
        words = [
            {"type": "word", "text": "今天", "start": 0.0, "end": 0.3, "speaker_id": "speaker_0"},
            {"type": "word", "text": "给", "start": 0.3, "end": 0.5, "speaker_id": "speaker_0"},
            {"type": "word", "text": "大家", "start": 0.5, "end": 0.8, "speaker_id": "speaker_0"},
        ]
        phrases = pack.group_into_phrases(words, 0.5)
        self.assertEqual(len(phrases), 1)
        self.assertEqual(phrases[0]["text"], "今天给大家")

    # english phrases keep spaces between words
    def test_english_phrase(self):
        words = [
            {"type": "word", "text": "hello", "start": 0.0, "end": 0.3, "speaker_id": "speaker_0"},
            {"type": "word", "text": "world", "start": 0.3, "end": 0.6, "speaker_id": "speaker_0"},
        ]
        phrases = pack.group_into_phrases(words, 0.5)
        self.assertEqual(phrases[0]["text"], "hello world")


if __name__ == "__main__":
    unittest.main()
