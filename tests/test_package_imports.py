"""Public package imports must not require the private research environment."""

import importlib
import pkgutil

import multimodal_fusion
from multimodal_fusion.diagnostics.hmm_diagnostics import HMMDiagnostics
from multimodal_fusion.pipeline.hmm_annual_runner import _import_hmm_diagnostics


def test_public_package_modules_import():
    modules = pkgutil.walk_packages(
        multimodal_fusion.__path__, prefix="multimodal_fusion."
    )
    for module in modules:
        importlib.import_module(module.name)


def test_hmm_diagnostics_loader_uses_public_namespace():
    assert _import_hmm_diagnostics() is HMMDiagnostics
