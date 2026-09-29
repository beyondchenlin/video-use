"""Tests for the final composite loudness wiring in helpers render py."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


# load render py straight from its file path under a private module name
MODULE_PATH = Path(__file__).resolve().parents[1] / "helpers" / "render.py"
SPEC = importlib.util.spec_from_file_location("video_use_render", MODULE_PATH)
assert SPEC and SPEC.loader
render = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render)


# build_final_composite must route through the shared loudnorm helper
class FinalCompositeLoudnormTests(unittest.TestCase):
    # loudnorm composites to a prenorm file then normalizes into the output
    def test_loudnorm_normalizes_the_composite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            edit_dir = Path(temp_dir)
            base = edit_dir / "base.mp4"
            base.write_bytes(b"base")
            out = edit_dir / "final.mp4"
            calls = {}

            # fake the shared helper and record the paths it was given
            def fake_norm(composite, final, preview=False):
                calls["composite"] = Path(composite)
                calls["final"] = Path(final)
                calls["preview"] = preview
                Path(final).write_bytes(b"final")
                return True

            with (
                patch.object(render, "apply_loudnorm_two_pass", side_effect=fake_norm),
                patch.object(render, "run") as run,
            ):
                render.build_final_composite(
                    base, [], None, out, edit_dir, loudnorm=True
                )

            self.assertEqual(calls["composite"], edit_dir / "final.prenorm.mp4")
            self.assertEqual(calls["final"], edit_dir / "final.mp4")
            self.assertFalse((edit_dir / "final.prenorm.mp4").exists())
            self.assertTrue(out.exists())
            self.assertEqual(run.call_args.args[0][-1], str(edit_dir / "final.prenorm.mp4"))

    # the preview flag reaches the shared normalization helper
    def test_preview_flag_is_forwarded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            edit_dir = Path(temp_dir)
            base = edit_dir / "base.mp4"
            base.write_bytes(b"base")
            out = edit_dir / "final.mp4"
            with (
                patch.object(render, "apply_loudnorm_two_pass", return_value=True) as norm,
                patch.object(render, "run"),
            ):
                render.build_final_composite(
                    base, [], None, out, edit_dir,
                    loudnorm=True, loudnorm_preview=True,
                )
            self.assertTrue(norm.call_args.kwargs["preview"])

    # without loudnorm the composite is written straight to the output
    def test_without_loudnorm_writes_the_output_directly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            edit_dir = Path(temp_dir)
            base = edit_dir / "base.mp4"
            base.write_bytes(b"base")
            out = edit_dir / "final.mp4"
            with (
                patch.object(render, "apply_loudnorm_two_pass") as norm,
                patch.object(render, "run") as run,
            ):
                render.build_final_composite(base, [], None, out, edit_dir)
            norm.assert_not_called()
            self.assertEqual(run.call_args.args[0][-1], str(out))


if __name__ == "__main__":
    unittest.main()
