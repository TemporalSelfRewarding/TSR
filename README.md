# <div align="center">Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future</div>

<div align="center">
<a href="https://arxiv.org/abs/2508.06026" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://github.com/wahaha12321/TSR/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

Self-Rewarding Language Models let a single model act as both the policy that generates responses and the judge that scores them, producing its own preference pairs for iterative DPO training. However, the chosen and rejected responses are drawn from the same model at the same moment, so the reward signal collapses as the model improves. **Temporal Self-Rewarding (TSR)** decouples the chosen and rejected signals along a temporal axis: it anchors *chosen* responses with a stronger **future** model and grounds *rejected* responses with a weaker **past** model, keeping the preference margin informative across iterations. This repository provides the full training pipeline—Supervised Fine-Tuning (SFT), vanilla Self-Rewarding (SR), and Temporal Self-Rewarding (TSR).

**If you have any question, feel free to contact wx2013831686@gmail.com and yidongwang37@gmail.com**

## Install environment

Clone the repository and install the packages:

```bash
git clone https://github.com/wahaha12321/TSR
cd TSR
pip install -r requirements.txt
```

Before running scripts, set your `CONDA_PATH` and `HF_TOKEN` at the top of the corresponding file in [`scripts/`](./scripts).

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
