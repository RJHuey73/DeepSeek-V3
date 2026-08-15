# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What This Is

**DeepSeek-V3** is the official reference repository for the DeepSeek-V3 model:
a 671B-parameter Mixture-of-Experts (MoE) language model (37B activated per
token) using Multi-head Latent Attention (MLA) and DeepSeekMoE, with an
auxiliary-loss-free load-balancing strategy and a Multi-Token Prediction (MTP)
training objective. This repo is **not a training codebase and not a
general-purpose serving framework** — it ships only a minimal reference
inference demo ("DeepSeek-Infer Demo") plus weight-conversion utilities. There
is no test suite, no package config (`setup.py`/`pyproject.toml`), and no CI
beyond a GitHub `stale` issue bot. Production inference is expected to happen
via third-party frameworks (SGLang, LMDeploy, TensorRT-LLM, vLLM, LightLLM)
referenced in the README, not via this repo's `generate.py`.

Model weights themselves are **not** in this repo — they're downloaded
separately from Hugging Face (`deepseek-ai/DeepSeek-V3` /
`deepseek-ai/DeepSeek-V3-Base`) per Section 3/6 of the README.

## Layout

| Path | Purpose |
|------|---------|
| `inference/model.py` | Core model definition: `ModelArgs` dataclass (all hyperparameters), `ParallelEmbedding`, MLA attention (`MLA` class, naive vs "absorb" implementations), `MLP`/`MoE` (gated experts + shared experts + `Gate` router), `Transformer`. Supports both `bf16` and `fp8` compute paths and tensor-parallel (`world_size`/`rank` globals set via `torch.distributed`). |
| `inference/kernel.py` | Triton kernels for FP8: `act_quant` (activation quantization), `weight_dequant` (FP8→BF16 dequantization), `fp8_gemm` (FP8 matmul). Used by `model.py` when `gemm_impl == "fp8"` and by `fp8_cast_bf16.py`. |
| `inference/generate.py` | CLI entry point for the demo: loads a converted checkpoint + config, runs interactive chat or batch file inference via `torchrun`. |
| `inference/convert.py` | Converts a Hugging Face checkpoint (safetensors) into this repo's sharded, model-parallel checkpoint format (`model{rank}-mp{world_size}.safetensors`), renaming HF module names to this repo's naming (e.g. `self_attn`→`attn`, `q_proj`→`wq`) and splitting/sharding tensors per `--model-parallel`. |
| `inference/fp8_cast_bf16.py` | Standalone script to dequantize a full FP8 HF checkpoint to BF16 (does not require the model-parallel conversion step). |
| `inference/configs/*.json` | `ModelArgs` presets: `config_16B.json`, `config_236B.json` (DeepSeek-V2 scale, for reference), `config_671B.json` (DeepSeek-V3, FP8), `config_v3.1.json` (V3 with `scale_fmt: "ue8m0"` for a newer FP8 scale format). |
| `inference/requirements.txt` | Pinned deps for the demo: `torch==2.4.1`, `triton==3.0.0`, `transformers==4.46.3`, `safetensors==0.4.5`. |
| `README.md` | Primary docs: model summary, benchmark tables, download links, and Section 6 "How to Run Locally" (the canonical setup instructions). |
| `README_WEIGHTS.md` | Weight-file format documentation: `config.json` fields, Main Model vs MTP Module structure, FP8 quantization/dequantization scheme (128x128 block scaling). |
| `figures/` | Images embedded in the READMEs (logo, benchmark charts). |
| `.github/ISSUE_TEMPLATE/` | Bug report / feature request templates. |
| `.github/workflows/stale.yml` | Only CI-like automation in the repo — closes stale issues. |

## Commands

There is no build/lint/test tooling. The only workflows are weight conversion
and running the demo, both from inside `inference/`:

```bash
# 1. Install pinned deps (Linux + Python 3.10 only; Mac/Windows unsupported)
cd inference
pip install -r requirements.txt

# 2. (If starting from FP8 HF weights and you need BF16 instead) dequantize:
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8_weights --output-bf16-hf-path /path/to/bf16_weights

# 3. Convert HF checkpoint -> this repo's sharded/model-parallel format
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16

# 4. Run interactively (multi-node example: 2 nodes x 8 GPUs)
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR \
  generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json \
  --interactive --temperature 0.7 --max-new-tokens 200

# 4b. Or batch mode over a prompts file (one prompt per line)
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR \
  generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json \
  --input-file $FILE
```

`--n-experts` in `convert.py` must match `n_routed_experts` for the target
config (256 for `config_671B.json`/`config_v3.1.json`), and must be evenly
divisible by `--model-parallel`.

## Conventions

- **`ModelArgs` in `model.py` is the single source of truth for architecture
  hyperparameters.** The JSON files under `configs/` are just serialized
  instances of it (loaded via `ModelArgs(**json.load(f))` in `generate.py`).
  When adding a new model size/variant, add a new `configs/*.json`, don't hand-edit
  `model.py` defaults.
- **Global mutable state drives parallelism/precision, not `ModelArgs`
  fields.** `model.py` module-level globals `world_size`, `rank`, `block_size`,
  `gemm_impl` ("bf16"/"fp8"), `attn_impl` ("naive"/"absorb") are set from
  `torch.distributed` env vars / `ModelArgs.dtype` at load time and read
  throughout the model code — they are not passed as parameters.
- **Two MLA attention implementations coexist** (`attn_impl`): `"naive"`
  materializes full K/V; `"absorb"` uses the matrix-absorption trick for lower
  KV-cache memory. Both must stay numerically equivalent when modifying `MLA`.
- **Checkpoint naming translation is centralized in `convert.py`'s `mapping`
  dict.** HF module names (`self_attn`, `q_proj`, `gate_proj`, …) map to this
  repo's short names (`attn`, `wq`, `w1`, …) plus an optional shard dimension.
  Any new weight type must be added here or `convert.py` will assert.
- **FP8 is the native format; BF16 is a derived convenience.** Per
  `README_WEIGHTS.md`, only FP8 weights are published — `fp8_cast_bf16.py`
  dequantizes using the `weight_scale_inv` tensors (128x128 block scaling) that
  ship alongside each FP8 weight in `model.safetensors.index.json`.
- **`model.layers.61` is the MTP module, not a regular hidden layer** — note
  `convert.py` explicitly skips it (`if "model.layers.61" in name: continue`),
  since the 61-layer main model (config_671B) only uses hidden-layer indices
  0–60; layer 61 holds the Multi-Token-Prediction module's weights and needs
  separate handling.

## Gotchas

- **Platform**: the DeepSeek-Infer demo only supports **Linux with Python
  3.10**; Mac and Windows are explicitly unsupported (per README Section 6.1).
- **Hardware**: `generate.py` hardcodes `device="cuda"` and calls
  `torch.cuda.set_device(local_rank)` — this demo requires NVIDIA GPUs (not
  CPU, not directly portable to AMD/Ascom without one of the third-party
  frameworks in the README).
- **Multi-node/multi-GPU is expected for the 671B config** — the README's
  example uses 2 nodes x 8 GPUs (`--model-parallel 16`) since the full model
  doesn't fit on a single node's GPU memory even in FP8.
- **Hugging Face `transformers` does not natively support this architecture**
  (per README note) — you cannot just `AutoModel.from_pretrained(...)` the
  raw HF checkpoint with vanilla `transformers`; you must go through
  `convert.py` for this repo's demo, or use one of the listed third-party
  serving frameworks that have added native support.
- **Vocab size differs between configs**: `config_16B.json`/`config_236B.json`
  (DeepSeek-V2-era) use `vocab_size: 102400`, while `config_671B.json`/
  `config_v3.1.json` (DeepSeek-V3) use `vocab_size: 129280` — don't mix a V2
  config with V3 weights or vice versa.
- **No automated tests** — validate any change to `model.py`/`kernel.py`
  manually (e.g. running the demo end-to-end, or comparing `naive` vs
  `absorb` attention outputs) since there's no pytest/CI harness to catch
  regressions.
- Two licenses apply: code is MIT (`LICENSE-CODE`); model weights are under a
  separate Model License (`LICENSE-MODEL`) — commercial use of both is
  permitted, but they are not the same license.
