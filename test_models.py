"""Tests for models.py's resolve_adapter() dispatch and spec parsing.

This covers only the spec-parsing/dispatch logic and the exact library calls
each generic adapter makes (asserted via mocks) — not real model loading,
which needs a GPU/Delta allocation and gated model access; see loadtest.py for
that kind of end-to-end validation.

Runs without torch/open_clip/transformers/huggingface_hub installed: `torch`
is stubbed in sys.modules before `models` is imported (models.py imports it
unconditionally at module level), and open_clip/transformers/huggingface_hub
are stubbed per-test via mock.patch.dict, since models.py imports those
lazily inside the functions that need them.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _NoGrad:
    """Stands in for torch.no_grad used as a decorator: @torch.no_grad()."""

    def __call__(self, func):
        return func

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if "torch" not in sys.modules:
    _torch_stub = mock.MagicMock(name="torch_stub")
    _torch_stub.no_grad = _NoGrad
    _torch_stub.Tensor = object
    sys.modules["torch"] = _torch_stub

import models  # noqa: E402


def _fake_open_clip():
    module = mock.MagicMock(name="open_clip_stub")
    module.create_model_and_transforms.return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
    module.get_tokenizer.return_value = mock.MagicMock()
    return module


def _fake_transformers():
    module = mock.MagicMock(name="transformers_stub")
    module.CLIPModel.from_pretrained.return_value = mock.MagicMock()
    module.CLIPProcessor.from_pretrained.return_value = mock.MagicMock()
    return module


def _fake_huggingface_hub(download_path="/tmp/fake-checkpoint.pt"):
    module = mock.MagicMock(name="huggingface_hub_stub")
    module.hf_hub_download.return_value = download_path
    return module


class TestResolveAdapterDispatch(unittest.TestCase):
    def test_known_name_dispatches_to_ADAPTERS(self):
        fake_loader = mock.MagicMock(return_value="the-adapter")
        with mock.patch.dict(models.ADAPTERS, {"test": fake_loader}):
            result = models.resolve_adapter("test", device="cpu", hf_token="tok")
        fake_loader.assert_called_once_with(device="cpu", hf_token="tok")
        self.assertEqual(result, "the-adapter")

    def test_openclip_prefix_strips_prefix_before_dispatch(self):
        with mock.patch.object(models, "load_open_clip_hub", return_value="oc-adapter") as loader:
            result = models.resolve_adapter("openclip:wisdomik/QuiltNet-B-32", device="cpu", hf_token=None)
        loader.assert_called_once_with("wisdomik/QuiltNet-B-32", device="cpu", hf_token=None)
        self.assertEqual(result, "oc-adapter")

    def test_hfclip_prefix_strips_prefix_before_dispatch(self):
        with mock.patch.object(models, "load_hf_clip", return_value="hf-adapter") as loader:
            result = models.resolve_adapter("hfclip:openai/clip-vit-base-patch32", device="cpu", hf_token=None)
        loader.assert_called_once_with("openai/clip-vit-base-patch32", device="cpu", hf_token=None)
        self.assertEqual(result, "hf-adapter")

    def test_unknown_spec_raises_with_supported_formats(self):
        with self.assertRaises(ValueError) as ctx:
            models.resolve_adapter("not-a-real-spec", device="cpu", hf_token=None)
        message = str(ctx.exception)
        self.assertIn("conch", message)
        self.assertIn("openclip:", message)
        self.assertIn("hfclip:", message)


class TestLoadOpenClipHub(unittest.TestCase):
    def test_hf_hub_form_loads_directly_from_repo(self):
        fake_open_clip = _fake_open_clip()
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip}):
            adapter = models.load_open_clip_hub("wisdomik/QuiltNet-B-32", device="cpu", hf_token="tok")

        fake_open_clip.create_model_and_transforms.assert_called_once_with(
            "hf-hub:wisdomik/QuiltNet-B-32", pretrained=None
        )
        fake_open_clip.get_tokenizer.assert_called_once_with("hf-hub:wisdomik/QuiltNet-B-32")
        self.assertEqual(adapter.name, "openclip:wisdomik/QuiltNet-B-32")
        self.assertIsNotNone(adapter.encode_texts)

    def test_arch_at_repo_form_downloads_checkpoint_then_loads_by_arch(self):
        fake_open_clip = _fake_open_clip()
        fake_hub = _fake_huggingface_hub(download_path="/cache/pathgenclip.pt")
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip, "huggingface_hub": fake_hub}):
            adapter = models.load_open_clip_hub(
                "ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt", device="cpu", hf_token="tok"
            )

        fake_hub.hf_hub_download.assert_called_once_with(
            repo_id="jamessyx/PathGen-CLIP", filename="pathgenclip.pt", token="tok"
        )
        fake_open_clip.create_model_and_transforms.assert_called_once_with(
            "ViT-B-16", pretrained="/cache/pathgenclip.pt"
        )
        self.assertEqual(adapter.name, "openclip:ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt")

    def test_arch_at_spec_missing_filename_raises_value_error(self):
        with self.assertRaises(ValueError):
            models.load_open_clip_hub("ViT-B-16@jamessyx/PathGen-CLIP/", device="cpu", hf_token=None)

    def test_arch_at_spec_missing_repo_raises_value_error(self):
        with self.assertRaises(ValueError):
            models.load_open_clip_hub("ViT-B-16@justafilename.pt", device="cpu", hf_token=None)

    def test_missing_open_clip_raises_helpful_import_error(self):
        with mock.patch.dict(sys.modules, {"open_clip": None}):
            with self.assertRaises(ImportError) as ctx:
                models.load_open_clip_hub("wisdomik/QuiltNet-B-32", device="cpu", hf_token=None)
        self.assertIn("open_clip_torch", str(ctx.exception))

    def test_load_failure_is_wrapped_with_model_context(self):
        fake_open_clip = _fake_open_clip()
        fake_open_clip.create_model_and_transforms.side_effect = OSError("no such repo")
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip}):
            with self.assertRaises(RuntimeError) as ctx:
                models.load_open_clip_hub("bogus/repo", device="cpu", hf_token=None)
        self.assertIn("bogus/repo", str(ctx.exception))
        self.assertIn("no such repo", str(ctx.exception))


class TestBespokeAdapterImportErrors(unittest.TestCase):
    """A missing dependency must surface its own name, not just "not installed".

    On Delta, MUSK imported fine but its modeling code needed fairscale; the old
    message said "MUSK is not installed", which sent us reinstalling MUSK.
    """

    def test_musk_dependency_failure_names_the_cause_and_fairscale(self):
        import types

        musk_pkg = types.ModuleType("musk")
        musk_pkg.__path__ = []  # a package, so `from musk import modeling` looks up the submodule
        musk_pkg.utils = mock.MagicMock()
        stubs = {"musk": musk_pkg, "musk.utils": musk_pkg.utils, "timm": mock.MagicMock(), "musk.modeling": None}
        with mock.patch.dict(sys.modules, stubs):
            with self.assertRaises(ImportError) as ctx:
                models.load_musk(device="cpu")
        msg = str(ctx.exception)
        self.assertIn("musk.modeling", msg)  # the underlying import that failed
        self.assertIn("fairscale", msg)  # the install hint covers the usual culprit

    def test_conch_import_failure_includes_cause(self):
        with mock.patch.dict(sys.modules, {"conch": None, "conch.open_clip_custom": None}):
            with self.assertRaises(ImportError) as ctx:
                models.load_conch(device="cpu")
        self.assertIn("conch", str(ctx.exception))
        self.assertIn("pip install git+", str(ctx.exception))


class TestMuskTokenizerPath(unittest.TestCase):
    def test_prefers_tokenizer_inside_package(self):
        with tempfile.TemporaryDirectory() as pkg:
            os.makedirs(os.path.join(pkg, "models"))
            path = os.path.join(pkg, "models", "tokenizer.spm")
            open(path, "w").close()
            self.assertEqual(models._musk_tokenizer_path(pkg), path)

    def test_falls_back_to_hf_repo_when_pip_install_omits_it(self):
        fake_hub = mock.MagicMock()
        fake_hub.hf_hub_download.return_value = "/cache/tokenizer.spm"
        with tempfile.TemporaryDirectory() as pkg, mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            self.assertEqual(models._musk_tokenizer_path(pkg, "tok"), "/cache/tokenizer.spm")
        fake_hub.hf_hub_download.assert_called_once_with(repo_id="xiangjx/musk", filename="tokenizer.spm", token="tok")

    def test_failed_fallback_explains_access(self):
        fake_hub = mock.MagicMock()
        fake_hub.hf_hub_download.side_effect = OSError("401 gated")
        with tempfile.TemporaryDirectory() as pkg, mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with self.assertRaises(RuntimeError) as ctx:
                models._musk_tokenizer_path(pkg)
        self.assertIn("401 gated", str(ctx.exception))
        self.assertIn("hf auth login", str(ctx.exception))


class TestMuskWeightsPath(unittest.TestCase):
    def test_resolves_through_hf_cache_not_musk_force_download(self):
        fake_hub = mock.MagicMock()
        fake_hub.hf_hub_download.return_value = "/cache/model.safetensors"
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            self.assertEqual(models._musk_weights_path("tok"), "/cache/model.safetensors")
        _, kwargs = fake_hub.hf_hub_download.call_args
        self.assertEqual(kwargs["repo_id"], "xiangjx/musk")
        self.assertEqual(kwargs["filename"], "model.safetensors")
        self.assertNotIn("force_download", kwargs)


class TestMuskEncoders(unittest.TestCase):
    """Encoders must use the retrieval head on both sides, one string per tokenize call."""

    def _load(self):
        import types

        musk_pkg = types.ModuleType("musk")
        musk_pkg.__path__ = []
        musk_pkg.__file__ = "/nonexistent/musk/__init__.py"
        musk_utils = mock.MagicMock()
        musk_utils.xlm_tokenizer.side_effect = lambda text, tok, max_len: ([1, 2], [0, 0])
        musk_pkg.utils = musk_utils
        model = mock.MagicMock(name="musk_model")
        model.to.return_value.eval.return_value = model
        timm = mock.MagicMock()
        timm.create_model.return_value = model
        stubs = {
            "musk": musk_pkg, "musk.utils": musk_utils, "musk.modeling": mock.MagicMock(),
            "timm": timm, "timm.data": mock.MagicMock(), "timm.data.constants": mock.MagicMock(),
            "transformers": mock.MagicMock(), "torchvision": mock.MagicMock(),
        }
        with mock.patch.dict(sys.modules, stubs), \
                mock.patch.object(models, "_musk_tokenizer_path", return_value="/t.spm"), \
                mock.patch.object(models, "_musk_weights_path", return_value="/w.safetensors"):
            adapter = models.load_musk(device="cpu")
        return adapter, model, musk_utils

    def test_text_tokenized_one_string_at_a_time_with_retrieval_head(self):
        adapter, model, musk_utils = self._load()
        adapter.encode_texts(["tumor", "stroma"])
        self.assertEqual([c.args[0] for c in musk_utils.xlm_tokenizer.call_args_list], ["tumor", "stroma"])
        kwargs = model.call_args.kwargs
        self.assertTrue(kwargs["with_head"])
        self.assertTrue(kwargs["out_norm"])

    def test_images_use_same_head_as_text(self):
        adapter, model, _ = self._load()
        adapter.encode_images(mock.MagicMock())
        kwargs = model.call_args.kwargs
        self.assertTrue(kwargs["with_head"])
        self.assertTrue(kwargs["out_norm"])
        self.assertFalse(kwargs["ms_aug"])


class TestLoadHfClip(unittest.TestCase):
    def test_loads_via_clip_model_and_processor(self):
        fake_transformers = _fake_transformers()
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            adapter = models.load_hf_clip("openai/clip-vit-base-patch32", device="cpu", hf_token="tok")

        fake_transformers.CLIPModel.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32", token="tok"
        )
        fake_transformers.CLIPProcessor.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32", token="tok"
        )
        self.assertEqual(adapter.name, "hfclip:openai/clip-vit-base-patch32")
        self.assertIsNotNone(adapter.encode_texts)

    def test_falls_back_to_hf_token_env_var(self):
        fake_transformers = _fake_transformers()
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            with mock.patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
                models.load_hf_clip("openai/clip-vit-base-patch32", device="cpu", hf_token=None)

        fake_transformers.CLIPModel.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32", token="env-token"
        )

    def test_missing_transformers_raises_helpful_import_error(self):
        with mock.patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                models.load_hf_clip("openai/clip-vit-base-patch32", device="cpu", hf_token=None)
        self.assertIn("transformers", str(ctx.exception))

    def test_load_failure_is_wrapped_with_model_context(self):
        fake_transformers = _fake_transformers()
        fake_transformers.CLIPModel.from_pretrained.side_effect = OSError("gated repo")
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            with self.assertRaises(RuntimeError) as ctx:
                models.load_hf_clip("some/gated-repo", device="cpu", hf_token=None)
        self.assertIn("some/gated-repo", str(ctx.exception))
        self.assertIn("gated repo", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
