# TURING-1 (`turing-foundation` 0.1.0)

Arquitectura híbrida conv/atención + MoE fino + MTP en **PyTorch puro**
(sin `transformers`), con base LFM2 incluida (`lfm2/`). Diseñada para
inferencia rápida y barata en CPU/edge y NPU.

- Backbone: mayoría de convoluciones cortas con doble puerta (+ rama
  dilatada con gate a identidad LFM2) y minoría de GQA con QK-Norm/RoPE.
- Memoria: KV compartido por pares (CLA) + ventana deslizante 2/3 y
  global 1/3 con global temprano (−66% KV a 32k).
- Capacidad: MoE fino top-6/8 + experto compartido (estilo Kimi/DeepSeek),
  router softmax + sesgo aux-free con actualización operativa.
- Velocidad: 1 cabeza MTP + decodificación especulativa autocontenida
  (aceptación medida 0.86 en datos estructurados).
- 4-bit-friendly por diseño (QK-Norm, puertas acotadas, sin matrices
  gigantes nuevas) + receta PTQ documentada en `turing/utils.py`.

## Instalación

```bash
# desde el repo (cualquier equipo con Python >= 3.10)
pip install .

# o editable para desarrollar
pip install -e .

# solo CPU (ruedas ligeras)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install .
```

En notebooks en la nube:

```python
# !pip install git+https://github.com/bueormnew/TURING-1.git  # o sube la carpeta dist/
import turing
```

## Uso en 30 segundos

```python
import torch, turing

model = turing.TuringForCausalLM.from_preset("tiny")  # tiny|350m|700m|1.2b|2.6b
model.eval()
ids = torch.randint(0, 256, (1, 16))
print(model(input_ids=ids)["logits"].shape)   # (1, 16, 256)

# generar (+ stats de especulación)
gen = model.generate(ids, max_new_tokens=20)
gen, st = model.generate_speculative(ids, max_new_tokens=20, return_stats=True)
print(st)  # {"drafts":.., "accepted":.., "rate":..}

# texto byte-level sin tokenizador
ids = turing.encode_text("hola mundo")
print(turing.decode_ids(model.generate(ids, max_new_tokens=20)[0]))

# memoria estimada a 32k
turing.utils.memory_report(turing.get_preset("1.2b"))  # o print_report
```

## Entrenar

```bash
python train_turing.py --steps 60                              # humo sintético
python train_turing.py --text datos.txt --seq 256 --steps 200  # byte-LM real
```

API (`turing/train.py`): `TrainConfig`, `train_step` (acumulación, clip,
scheduler, bias MoE), `train()`, `save/load_checkpoint`, `ByteDataLoader`.

## Estructura

```
turing/      arquitectura TURING-1 (config, modules, cache, model, train, data, utils)
lfm2/        base densa LFM2 (intacta, importable)
tests/       85 pruebas: base, estabilidad, entrenamiento, capacidad,
             endurecimiento, producción, packaging
train_turing.py   CLI de entrenamiento real
examples/quickstart.ipynb
```

## Tests

```bash
set PYTHONPATH=%CD%   # Windows (export PYTHONPATH=$PWD en Linux)
python tests/test_turing_production.py   # puerta de producción (9)
python tests/test_turing_proof.py        # capacidad aprendida (9)
python tests/test_turing_train.py        # estabilidad de entreno (13)
python tests/test_turing_stability.py    # robustez (25)
python tests/test_turing_hardening.py    # escala (10)
python tests/test_packaging.py           # export (6)
python tests/test_turing.py tests/test_lfm2.py
```

## Notas

- `torch.compile` traza el modelo, pero necesita toolchain C++/CUDA en la
  máquina (no incluido aquí).
- Licencia Apache-2.0 (ver `LICENSE`).
