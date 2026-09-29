"""Tests for the precut helper output path resolution."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


# load precut py straight from its file path under a private module name
MODULE_PATH = Path(__file__).resolve().parents[1] / "helpers" / "precut.py"
SPEC = importlib.util.spec_from_file_location("video_use_precut", MODULE_PATH)
assert SPEC and SPEC.loader
precut = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(precut)


# resolve_output_path must treat missing directories as directories
class ResolveOutputPathTests(unittest.TestCase):
    # no output argument writes beside the input video
    def test_default_is_beside_the_video(self):
        video = Path("videos") / "clip.mp4"
        self.assertEqual(
            precut.resolve_output_path(video, None),
            video.with_name("clip.precut.mp4"),
        )

    # an existing directory receives the default file name
    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = precut.resolve_output_path(Path("clip.mp4"), temp_dir)
        self.assertEqual(out, Path(temp_dir) / "clip.precut.mp4")

    # a bare name with no suffix is treated as a directory even when missing
    def test_missing_bare_name_is_a_directory(self):
        out = precut.resolve_output_path(Path("clip.mp4"), "newdir")
        self.assertEqual(out, Path("newdir") / "clip.precut.mp4")

    # a trailing separator marks a directory that may not exist yet
    def test_trailing_separator_is_a_directory(self):
        value = "newdir" + os.sep
        out = precut.resolve_output_path(Path("clip.mp4"), value)
        self.assertEqual(out, Path("newdir") / "clip.precut.mp4")

    # a name with a suffix is an explicit output file
    def test_name_with_suffix_is_a_file(self):
        out = precut.resolve_output_path(Path("clip.mp4"), "final.mp4")
        self.assertEqual(out, Path("final.mp4"))


if __name__ == "__main__":
    unittest.main()
