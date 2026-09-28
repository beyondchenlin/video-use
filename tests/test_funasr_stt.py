"""unit tests for the local funasr chinese engine in helpers funasr_stt py

they cover chinese word grouping against per-character timestamps, punctuation
attachment, speaker ids, spacing entries, text joining, the transcript payload
contract, and the missing-install preflight message. no model is loaded and no
network call is made; the model used by transcribe_wav is replaced with a fake.
"""

import builtins
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


# load a helper straight from its file path under a private module name
def load_helper(name: str, alias: str):
    path = Path(__file__).parents[1] / "helpers" / name
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


funasr_stt = load_helper("funasr_stt.py", "video_use_funasr_stt")


# one chinese sentence with per-character timestamps in ms and a 700ms gap before the second clause
def chinese_sentence() -> dict:
    return {
        "text": "今天天气很好，我们出去玩。",
        "timestamp": [
            [0, 300], [300, 600], [600, 900], [900, 1200], [1200, 1500], [1500, 1800],
            [2500, 2800], [2800, 3100], [3100, 3400], [3400, 3700], [3700, 4000],
        ],
        "spk": 0,
    }


# tokenizing picks jieba for chinese and a regex for english
class TokenizeTests(unittest.TestCase):
    # chinese text splits into words that concatenate back together
    def test_chinese_tokenized_into_words(self):
        tokens = funasr_stt.tokenize_sentence("今天天气很好")
        self.assertTrue(all(tok.strip() for tok in tokens))
        self.assertEqual("".join(tokens), "今天天气很好")

    # english text splits into words followed by punctuation
    def test_english_tokenized_into_words_and_punct(self):
        self.assertEqual(
            funasr_stt.tokenize_sentence("hi there,"),
            ["hi", "there", ","],
        )

    # blank or whitespace only text yields no tokens
    def test_blank_sentence_is_empty(self):
        self.assertEqual(funasr_stt.tokenize_sentence("   "), [])

    # cjk detection finds chinese characters and rejects english
    def test_has_cjk(self):
        self.assertTrue(funasr_stt.has_cjk("中文"))
        self.assertFalse(funasr_stt.has_cjk("english"))


# per character timestamps are grouped into word entries
class WordsFromSentenceTests(unittest.TestCase):
    # words concatenate back to the sentence with punctuation attached
    def test_chinese_words_grouped_and_punct_attached(self):
        words = funasr_stt.words_from_sentence(chinese_sentence())
        texts = [w["text"] for w in words]
        self.assertEqual("".join(texts), "今天天气很好，我们出去玩。")
        self.assertTrue(any(t.endswith("，") for t in texts))
        self.assertTrue(texts[-1].endswith("。"))

    # each word spans the timestamps of its characters
    def test_word_timestamps_span_their_characters(self):
        words = funasr_stt.words_from_sentence(chinese_sentence())
        self.assertAlmostEqual(words[0]["start"], 0.0, places=3)
        self.assertAlmostEqual(words[-1]["end"], 4.0, places=3)
        # the clause after the 700ms gap starts at 2.5s
        self.assertAlmostEqual(words[3]["start"], 2.5, places=3)

    # the spk field becomes a speaker id on every word
    def test_speaker_id_from_spk(self):
        words = funasr_stt.words_from_sentence(chinese_sentence())
        self.assertTrue(all(w["speaker_id"] == "speaker_0" for w in words))

    # a sentence without spk yields no speaker id
    def test_missing_speaker_is_none(self):
        sentence = chinese_sentence()
        del sentence["spk"]
        words = funasr_stt.words_from_sentence(sentence)
        self.assertTrue(all(w["speaker_id"] is None for w in words))

    # running out of timestamps keeps words instead of crashing
    def test_short_timestamps_do_not_crash(self):
        sentence = chinese_sentence()
        sentence["timestamp"] = sentence["timestamp"][:2]
        words = funasr_stt.words_from_sentence(sentence)
        self.assertTrue(words)
        self.assertIn("今天", words[0]["text"])

    # punctuation before the first word is handled safely
    def test_leading_punctuation_does_not_crash(self):
        sentence = {"text": "，你好", "timestamp": [[0, 300], [300, 600]], "spk": 1}
        words = funasr_stt.words_from_sentence(sentence)
        self.assertTrue(words)
        self.assertEqual(words[0]["speaker_id"], "speaker_1")


# spacing entries mark the long gaps between words
class SpacingTests(unittest.TestCase):
    # a gap above the threshold becomes one spacing entry
    def test_long_gap_gets_spacing(self):
        entries = funasr_stt.add_spacing(funasr_stt.words_from_sentence(chinese_sentence()))
        spacing = [e for e in entries if e["type"] == "spacing"]
        self.assertEqual(len(spacing), 1)
        self.assertAlmostEqual(spacing[0]["start"], 1.8, places=2)
        self.assertAlmostEqual(spacing[0]["end"], 2.5, places=2)

    # a gap below the threshold produces no spacing
    def test_short_gap_gets_no_spacing(self):
        sentence = {
            "text": "你好啊",
            "timestamp": [[0, 200], [200, 400], [400, 500]],
            "spk": 0,
        }
        entries = funasr_stt.add_spacing(funasr_stt.words_from_sentence(sentence))
        self.assertFalse(any(e["type"] == "spacing" for e in entries))

    # spacing keeps the speaker id of the preceding word
    def test_spacing_inherits_speaker(self):
        entries = funasr_stt.add_spacing(funasr_stt.words_from_sentence(chinese_sentence()))
        spacing = next(e for e in entries if e["type"] == "spacing")
        self.assertEqual(spacing["speaker_id"], "speaker_0")


# joined text spaces english words but not chinese words
class JoinTextTests(unittest.TestCase):
    # chinese words join without spaces and keep punctuation
    def test_chinese_text_has_no_word_spaces(self):
        entries = funasr_stt.add_spacing(funasr_stt.words_from_sentence(chinese_sentence()))
        text = funasr_stt.join_text(entries)
        self.assertEqual(text, "今天天气很好，我们出去玩。")

    # english words join with a single space
    def test_english_text_has_word_spaces(self):
        words = [
            {"type": "word", "text": "hi", "start": 0.0, "end": 0.2},
            {"type": "word", "text": "there", "start": 0.3, "end": 0.8},
        ]
        self.assertEqual(funasr_stt.join_text(words), "hi there")


# transcribe wav returns the contract the other helpers read
class TranscribeContractTests(unittest.TestCase):
    # fake model that returns one recognized sentence
    class FakeModel:
        # return the sentence info the real model would produce
        def generate(self, wav, **kwargs):
            return [{"sentence_info": [chinese_sentence()]}]

    # run transcribe wav with the model replaced by the fake
    def run_transcribe(self):
        with patch.object(funasr_stt, "get_model", return_value=self.FakeModel()):
            return funasr_stt.transcribe_wav(Path("sample.wav"), hotwords="今天")

    # the payload carries the engine library model language and text
    def test_top_level_fields(self):
        payload = self.run_transcribe()
        self.assertEqual(payload["engine"], "funasr")
        self.assertEqual(payload["library"], "funasr")
        self.assertEqual(payload["language_code"], "zh")
        self.assertIn("paraformer", payload["model"])
        self.assertEqual(payload["text"], "今天天气很好，我们出去玩。")

    # every word is well formed and timestamps stay monotonic
    def test_word_entries_are_well_formed(self):
        payload = self.run_transcribe()
        words = [e for e in payload["words"] if e["type"] == "word"]
        for word in words:
            self.assertTrue(set(word) >= {"type", "text", "start", "end"})
            self.assertGreaterEqual(word["end"], word["start"])
        bounds = [e["end"] for e in payload["words"]]
        self.assertEqual(bounds, sorted(bounds))

    # the payload includes spacing for the long gap
    def test_spacing_entries_present(self):
        payload = self.run_transcribe()
        self.assertTrue(any(e["type"] == "spacing" for e in payload["words"]))


# a missing funasr install fails with the exact extra
class PreflightTests(unittest.TestCase):
    # import shim that raises for funasr only
    def fake_import(self, real_import):
        # inner import function that raises only for funasr
        def shim(name, *args, **kwargs):
            if name == "funasr":
                raise ImportError("no funasr")
            return real_import(name, *args, **kwargs)

        return shim

    # preflight exits naming the stt funasr extra
    def test_missing_funasr_exits_with_extra(self):
        real_import = builtins.__import__
        with patch.object(builtins, "__import__", side_effect=self.fake_import(real_import)):
            with self.assertRaises(SystemExit) as ctx:
                funasr_stt.preflight()
        self.assertIn("stt-funasr", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
