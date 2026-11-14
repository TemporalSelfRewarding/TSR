# <div align="center">Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future</div>

<div align="center">
<a href="https://arxiv.org/abs/2508.06026" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://github.com/TemporalSelfRewarding/TSR/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

Self-Rewarding Language Models let a single model act as both the policy that generates responses and the judge that scores them, producing its own preference pairs for iterative DPO training. However, the chosen and rejected responses are drawn from the same model at the same moment, so the reward signal collapses as the model improves. **Temporal Self-Rewarding (TSR)** decouples the chosen and rejected signals along a temporal axis: it anchors *chosen* responses with a stronger **future** model and grounds *rejected* responses with a weaker **past** model, keeping the preference margin informative across iterations. This repository provides the full training pipeline—Supervised Fine-Tuning (SFT), vanilla Self-Rewarding (SR), and Temporal Self-Rewarding (TSR).

**If you have any question, feel free to contact wx2013831686@gmail.com and yidongwang37@gmail.com**

## Author-verified reproduction

The reproduction results reported in [Tanhhhhtjy/tsr-reproduction](https://github.com/Tanhhhhtjy/tsr-reproduction) have been verified by the TSR authors. See that repository for the reproduction code, experimental settings, and evaluation results.

## Install environment

This corrected checkout is based on official commit `020266aa9ef3cc35bf81384c9b59f98cff62aa1c`.
Use Python 3.11 on a CUDA training host. The core training packages are pinned to the verified reproduction stack:

```bash
# Run inside this corrected checkout.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e ./alignment-handbook
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

An activated environment containing training and vLLM dependencies can run all scripts; leave `CONDA_PATH` unset.
For the original two-environment layout, export `CONDA_PATH` (with `myenv` for generation and `handbook` for training).
Export `HF_TOKEN` only when authenticated model access is needed; credentials are not written into scripts.
`SFT_BASE_MODEL` can point to an existing base-model directory. `CUDA_VISIBLE_DEVICES` selects the GPUs.
The launcher keeps global batch size 32 for supported GPU counts. Training requires GPUs; CPU tests below do not.

`requirements-reproduction-lock.txt` preserves the historical full package snapshot for reference. The supported
core versions are torch 2.5.1, Transformers 4.46.3, TRL 0.12.2, Accelerate 1.1.1, DeepSpeed 0.15.4,
PEFT 0.13.2, datasets 2.21.0, and vLLM 0.6.4.post1.

## Usage

### 1. Data

The `data/` directory contains the SFT data and the self-play seed prompts:

- [`data/sft_data/`](./data/sft_data) — Instruction Fine-Tuning (IFT) and Evaluation Fine-Tuning (EFT) data used for SFT.
- [`data/SelfPlayBeginData/`](./data/SelfPlayBeginData) — seed question sets (`qa1.jsonl` … `qa4.jsonl`) consumed during the self-rewarding iterations.

### 2. Training Pipeline

The `scripts/` directory contains three bash scripts that make up the end-to-end pipeline:

- `sft.sh` — Supervised Fine-Tuning (SFT with IFT and EFT data).
- `sr.sh` — vanilla Self-Rewarding iterative DPO training.
- `tsr.sh` — Temporal Self-Rewarding training (past-future decoupling of chosen/rejected).

Run them in order:

```bash
# Step 1: Supervised Fine-Tuning
bash scripts/sft.sh

# Step 2 (baseline): vanilla Self-Rewarding
bash scripts/sr.sh

# Step 2 (ours): Temporal Self-Rewarding
bash scripts/tsr.sh
```

Each self-rewarding script loops over iterations: it generates candidate responses with vLLM, has the model judge them, reformats the results into DPO preference pairs, and launches DPO training via `accelerate`. Intermediate data and per-iteration configs live under [`settings/`](./settings).

### Corrections applied to the official code

| Area | Corrected behavior |
|---|---|
| SFT launch | Uses the upstream ZeRO-3 configuration with bf16 computation and fp32 optimizer/master weights, matching the validated reproduction. Keeps full-token SFT loss, 3 epochs, lr 2e-6, and seed 42. The launcher saves the final weights without epoch checkpoints. |
| SFT model ID | Corrects the default to `meta-llama/Llama-3.1-8B`. |
| Batch size | All launched SFT/DPO stages have global batch 32. On 2 GPUs: SFT micro-batch 2 × accumulation 8; DPO micro-batch 1 × accumulation 16. The 8-GPU Mf0 template is also corrected from batch 16 to 32. |
| Future initialization | `iter1_tmp` (Mf1) starts from `iter0` (M1), not `iter0_tmp` (Mf0). Final iter1 M2 also starts from M1. |
| DPO BOS | A local `TSRDPOTrainer` subclass adds a missing BOS without changing TRL globally. It preserves the reproduction's post-truncation placement: 512 prompt tokens may become 513 with BOS; total sequence cap stays 2048. |
| DPO data loading | Explicitly loads only the requested local directory's `train.json`. Nested `official/train.json` and other JSON copies cannot silently double training rows. SFT continues to load IFT and EFT together. |
| SFT model card | Uses the TRL 0.12.2 model-card API; the post-save step no longer fails on `finetuned_from`. |
| Launch failure | Scripts stop at the first failed generation, judging, formatting, or training command. GPU/environment selection can be supplied externally. |
| Pair validation | Imports `sys` so mismatched prompts/row counts exit with the intended error. |

The duplicate validation file originated in the reproduction wrapper; official TSR does not itself create that
copy. The explicit local DPO loader also protects this official checkout when such a directory is supplied.

### Pair rules and the best archived configuration

Default pair behavior remains upstream: future wins a best-score tie (`>=`), rejected is the minimum across
current and M0 (current wins ties), and same-score chosen/rejected pairs are kept. Model-internal ties select the
first candidate. Self-judging is by M0 at iter0 and by M1 at iter1, including the future model's answers.

The formatters now expose explicit `--filter` (keep only chosen score > rejected score), and the future formatter
also exposes `--strict` (future must score strictly higher). These are experimental choices, not silent bug fixes.
To reproduce the pair policy of the archived LC 25.35 run, change **only iter1 D2**:

```bash
CUDA_VISIBLE_DEVICES=0,1 TSR_ITER1_FILTER_EQUAL_SCORES=1 bash scripts/tsr.sh
```

`TSR_ITER1_STRICT_FUTURE=1` selects the `>` ablation instead. Both switches default to `0`; iter0 and both D1
stages keep the original rules. Use a separate checkout/output tree for another variant: the official scripts
write fixed per-iteration filenames and do not provide the reproduction wrapper's resume/skip behavior.

The archived run used data-parallel vLLM generation, while these official scripts retain tensor parallelism.
These changes reproduce the verified pair construction and training settings; rerunning generation can change
the sampled answers. No new end-to-end GPU run or LC score is claimed for this checkout. No AlpacaEval driver is
added here; the archived reproduction repository retains that evaluation code and all original results.

### Validation

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m unittest discover -s tests -v
python tests/verify_archived_pairs.py \
  --work /path/to/work-tsr-singlecopy-20261104 \
  --report /tmp/official-pair-validation.json
```

The regression suite runs real CPU SFT/DPO optimizer steps with tiny random models, checks the model-card API,
single-file DPO loading, multi-file SFT loading, BOS handling, tie rules, launch arguments, and failure handling.
The archive check rebuilds seven saved pair sets (iter0 D1/D2, iter1 D1, four iter1 D2 variants), requires
byte-identical outputs, and verifies the real dataset loader yields 5,000/4,770 rows and 156/149 steps at batch 32.
The CPU validation environment uses torch 2.7.1 from this server, with the pinned TRL/Transformers/datasets stack;
production training remains pinned to torch 2.5.1 and has not been run on this CPU machine.

## Citation

If you find this repository useful, please cite our work.

```
@article{wang2025temporal,
  title={Temporal self-rewarding language models: Decoupling chosen-rejected via past-future},
  author={Wang, Yidong and Wang, Xin and Wang, Cunxiang and Fang, Junfeng and Wang, Qiufeng and Chu, Jianing and Meng, Xuran and Yang, Shuxun and Qin, Libo and Zhang, Yue and others},
  journal={arXiv preprint arXiv:2508.06026},
  year={2025}
}
```

## License

TSR is licensed under the [MIT License](./LICENSE).
