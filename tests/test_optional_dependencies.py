import builtins

import pytest

from probjax.utils.optional import require_ott


def test_require_ott_has_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def import_without_ott(name, *args, **kwargs):
        if name == "ott" or name.startswith("ott."):
            raise ModuleNotFoundError("No module named 'ott'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ott)

    with pytest.raises(ImportError, match=r"probjax\[wasserstein\]"):
        require_ott()
