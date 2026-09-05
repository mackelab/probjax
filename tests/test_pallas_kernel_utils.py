import importlib
import types


common_mod = importlib.import_module("probjax.nn.pallas_kernels.kernel_utils.common")
mamba_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.mamba")
ssd_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.ssd")


def test_pallas_call_compat_omits_backend_when_unsupported(monkeypatch):
    captured = {}

    def _fake_pallas_call(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(common_mod, "pallas_call_supports_backend", lambda: False)
    monkeypatch.setattr(common_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(common_mod.pl, "pallas_call", _fake_pallas_call)

    result = common_mod.pallas_call_compat(
        "kernel",
        out_shape="shape",
        backend="triton",
        grid=(1,),
    )

    assert result == "ok"
    assert captured["args"] == ("kernel",)
    assert captured["kwargs"]["out_shape"] == "shape"
    assert captured["kwargs"]["grid"] == (1,)
    assert "backend" not in captured["kwargs"]
    assert (
        type(captured["kwargs"]["compiler_params"]).__module__.endswith("triton.core")
    )


def test_pallas_call_compat_passes_backend_when_supported(monkeypatch):
    captured = {}

    def _fake_pallas_call(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(common_mod, "pallas_call_supports_backend", lambda: True)
    monkeypatch.setattr(common_mod.pl, "pallas_call", _fake_pallas_call)

    result = common_mod.pallas_call_compat(
        "kernel",
        out_shape="shape",
        backend="triton",
        grid=(1,),
    )

    assert result == "ok"
    assert captured["args"] == ("kernel",)
    assert captured["kwargs"] == {
        "out_shape": "shape",
        "backend": "triton",
        "grid": (1,),
    }


def test_pallas_call_compat_preserves_existing_compiler_params(monkeypatch):
    captured = {}

    def _fake_pallas_call(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    custom_params = object()
    monkeypatch.setattr(common_mod, "pallas_call_supports_backend", lambda: False)
    monkeypatch.setattr(common_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(common_mod.pl, "pallas_call", _fake_pallas_call)

    result = common_mod.pallas_call_compat(
        "kernel",
        out_shape="shape",
        backend="triton",
        compiler_params=custom_params,
    )

    assert result == "ok"
    assert captured["args"] == ("kernel",)
    assert captured["kwargs"] == {
        "out_shape": "shape",
        "compiler_params": custom_params,
    }


def test_get_first_available_attr_prefers_new_compiler_params_name():
    class _NewParams:
        pass

    module = types.SimpleNamespace(__name__="fake_triton", CompilerParams=_NewParams)

    assert (
        common_mod._get_first_available_attr(
            module, "CompilerParams", "TritonCompilerParams"
        )
        is _NewParams
    )


def test_get_first_available_attr_falls_back_to_legacy_name():
    class _LegacyParams:
        pass

    module = types.SimpleNamespace(
        __name__="fake_triton", TritonCompilerParams=_LegacyParams
    )

    assert (
        common_mod._get_first_available_attr(
            module, "CompilerParams", "TritonCompilerParams"
        )
        is _LegacyParams
    )


def test_def_partition_compat_omits_replication_factors_when_unsupported(monkeypatch):
    captured = {}

    def _fake_def_partition(**kwargs):
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(
        common_mod,
        "def_partition_supports_need_replication_factors",
        lambda _: False,
    )

    result = common_mod.def_partition_compat(
        _fake_def_partition,
        partition="partition",
        sharding_rule="rule",
        need_replication_factors=("seq",),
    )

    assert result == "ok"
    assert captured["kwargs"] == {
        "partition": "partition",
        "sharding_rule": "rule",
    }


def test_def_partition_compat_passes_replication_factors_when_supported(
    monkeypatch,
):
    captured = {}

    def _fake_def_partition(**kwargs):
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(
        common_mod,
        "def_partition_supports_need_replication_factors",
        lambda _: True,
    )

    result = common_mod.def_partition_compat(
        _fake_def_partition,
        partition="partition",
        sharding_rule="rule",
        need_replication_factors=("seq",),
    )

    assert result == "ok"
    assert captured["kwargs"] == {
        "partition": "partition",
        "sharding_rule": "rule",
        "need_replication_factors": ("seq",),
    }


def test_mamba_backend_prefers_triton_without_unsafe_mosaic_opt_in(monkeypatch):
    monkeypatch.setattr(mamba_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(mamba_mod, "_is_hopper_or_newer_gpu", lambda: True)
    monkeypatch.setenv("PROBJAX_PALLAS_PREFER_MOSAIC_GPU", "1")
    monkeypatch.delenv("PROBJAX_PALLAS_UNSAFE_ENABLE_MOSAIC_GPU", raising=False)

    assert mamba_mod._pallas_backend() == "triton"


def test_mamba_backend_allows_mosaic_with_unsafe_opt_in(monkeypatch):
    monkeypatch.setattr(mamba_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(mamba_mod, "_is_hopper_or_newer_gpu", lambda: True)
    monkeypatch.setenv("PROBJAX_PALLAS_PREFER_MOSAIC_GPU", "1")
    monkeypatch.setenv("PROBJAX_PALLAS_UNSAFE_ENABLE_MOSAIC_GPU", "1")

    assert mamba_mod._pallas_backend() == "mosaic_gpu"


def test_ssd_backend_prefers_triton_without_unsafe_mosaic_opt_in(monkeypatch):
    monkeypatch.setattr(ssd_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ssd_mod, "_is_hopper_or_newer_gpu", lambda: True)
    monkeypatch.setenv("PROBJAX_PALLAS_PREFER_MOSAIC_GPU", "1")
    monkeypatch.delenv("PROBJAX_PALLAS_UNSAFE_ENABLE_MOSAIC_GPU", raising=False)

    assert ssd_mod._pallas_backend() == "triton"


def test_ssd_backend_allows_mosaic_with_unsafe_opt_in(monkeypatch):
    monkeypatch.setattr(ssd_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ssd_mod, "_is_hopper_or_newer_gpu", lambda: True)
    monkeypatch.setenv("PROBJAX_PALLAS_PREFER_MOSAIC_GPU", "1")
    monkeypatch.setenv("PROBJAX_PALLAS_UNSAFE_ENABLE_MOSAIC_GPU", "1")

    assert ssd_mod._pallas_backend() == "mosaic_gpu"
