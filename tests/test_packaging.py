"""Tests de empaquetado/exportación (rápidos, sin red).

V1. Versión única (pyproject == turing.__version__).
V2. Superficie API pública completa importable desde `turing`.
V3. from_preset + overrides coherentes.
V4. encode/decode roundtrip (incl. UTF-8 multibyte e ids inválidos).
V5. requirements declara torch; README documenta instalación.
V6. Estructura exportable (pyproject/paquetes/README/examples).
"""

import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _toml():
    try:
        import tomllib
    except ImportError:
        print("SKIP V1: tomllib no disponible (Python < 3.11)")
        return None
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def test_v1_version():
    import turing
    assert turing.__version__ and isinstance(turing.__version__, str)
    meta = _toml()
    if meta is not None:
        assert meta["project"]["version"] == turing.__version__, (
            meta["project"]["version"], turing.__version__)
    print(f"OK V1: versión {turing.__version__}")


def test_v2_api():
    import turing
    need = ["TuringConfig", "TuringModel", "TuringForCausalLM", "TuringCache",
            "TuringAttention", "TuringMoE", "TuringMTPHead", "TuringMultiscaleConv",
            "get_preset", "memory_report", "QUANT_RECIPE",
            "TrainConfig", "train_step", "train", "forward_backward",
            "save_checkpoint", "load_checkpoint",
            "ByteDataLoader", "make_chunks", "read_bytes", "split_chunks",
            "encode_text", "decode_ids"]
    missing = [n for n in need if not hasattr(turing, n)]
    assert not missing, f"falta en API pública: {missing}"
    assert set(need) <= set(turing.__all__)
    print(f"OK V2: {len(need)} símbolos públicos")


def test_v3_from_preset():
    import turing
    m = turing.TuringForCausalLM.from_preset("tiny")
    assert m.config.vocab_size == 256
    m2 = turing.TuringForCausalLM.from_preset("tiny", hidden_size=48, num_attention_heads=4,
                                              num_key_value_heads=2)
    assert m2.config.hidden_size == 48 and m2.config.head_dim == 12
    ids = torch.randint(0, 256, (1, 8))
    m2.eval()
    with torch.no_grad():
        assert m2(input_ids=ids)["logits"].shape == (1, 8, 256)
    print("OK V3: from_preset + overrides")


def test_v4_codec():
    from turing import decode_ids, encode_text
    assert decode_ids(encode_text("hola mundo")) == "hola mundo"
    assert decode_ids(encode_text("日本語 αβγ")) == "日本語 αβγ"
    assert decode_ids([72, 105, 9999]) == "Hi?"
    print("OK V4: codec byte roundtrip + multibyte + inválidos")


def test_v5_requirements_readme():
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        req = f.read()
    assert "torch>=" in req, "requirements debe declarar torch"
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        rd = f.read()
    for token in ("pip install", "TuringForCausalLM", "train_turing.py", "quickstart"):
        assert token in rd, f"README sin {token!r}"
    print("OK V5: requirements + README")


def test_v6_layout():
    import tomllib
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        meta = tomllib.load(f)
    pkgs = meta["tool"]["setuptools"]["packages"]["find"]["include"]
    assert set(pkgs) >= {"turing", "lfm2"}, pkgs
    assert meta["project"]["requires-python"] >= ">=3.10"
    assert os.path.exists(os.path.join(ROOT, "examples", "quickstart.ipynb"))
    import json
    nb = json.load(open(os.path.join(ROOT, "examples", "quickstart.ipynb"), encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 5
    print("OK V6: layout exportable + notebook válido")


if __name__ == "__main__":
    suite = [test_v1_version, test_v2_api, test_v3_from_preset,
             test_v4_codec, test_v5_requirements_readme, test_v6_layout]
    failed = []
    for fn in suite:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"FALLO {fn.__name__}: {type(e).__name__}: {e}")
            failed.append(fn.__name__)
    print(f"--- {len(suite) - len(failed)}/{len(suite)} pasaron ---")
    if failed:
        print("FALLARON:", failed)
        raise SystemExit(1)
    print("PACKAGING: LISTO")
