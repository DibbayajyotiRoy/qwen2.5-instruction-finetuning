# Qwen2.5 Instruction Fine-Tuning

A production-ready instruction fine-tuning pipeline for the Qwen2.5-Instruct family of open-weight LLMs. Built with **PyTorch**, **Hugging Face Transformers & Datasets**, **Unsloth**, and **llama.cpp**.

## Overview

This project fine-tunes open-weight Qwen2.5-Instruct checkpoints for instruction following using memory-efficient QLoRA training via Unsloth. Training and validation metrics, including gradient behaviour, are logged to TensorBoard to diagnose overfitting and select the final checkpoint. After training, the model is merged and exported to GGUF for quantized inference with llama.cpp, with a built-in benchmark comparing throughput and memory against the full-precision baseline.

## Features

- **Efficient Fine-Tuning**: QLoRA via Unsloth with 4-bit NF4 quantization, gradient checkpointing, and Flash Attention 2 support.
- **Instruction Dataset Pipeline**: Template-based formatting, prompt packing, train/validation splits, and Hugging Face `datasets` integration.
- **Data-Centric Efficiency**: Concision-aware data curation that trains models on concise, correct examples to reduce inference token count (Cost-of-Pass) without shrinking the model.
- **Knowledge Distillation**: Optionally generate chain-of-thought / soft labels from a larger teacher model and fine-tune the student on those targets, matching the distillation paradigm used by OpenAI and Anthropic for efficient small-model training.
- **Training & Diagnostics**: Transformers `Trainer` with TensorBoard logging, learning rate scheduling, early stopping, and gradient norm clipping.
- **Reproducibility**: Seeded RNG, deterministic algorithms, pinned dependency versions, and structured YAML configs.
- **Export & Quantization**: Merge LoRA adapters and export to GGUF (Q4_K_M, Q5_K_M, Q8_0) for llama.cpp.
- **Benchmarking**: Compare full-precision Hugging Face inference against quantized llama.cpp inference (tok/s, peak memory).
- **Testing**: Unit and integration tests with `pytest`, code quality with `ruff` and `mypy`.
- **Containerization**: Docker and docker-compose for reproducible training and inference environments.

## Project Structure

```
qwen2.5-instruction-finetuning/
├── README.md                      # This file
├── requirements.txt               # Pinned Python dependencies
├── environment.yml                # Conda environment specification
├── pyproject.toml                 # Build, tooling, and pytest configuration
├── Dockerfile                     # Training/inference container
├── docker-compose.yml             # Service orchestration
├── Makefile                       # Common development tasks
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
├── configs/                       # YAML training & inference configs
│   ├── qlora_config.yaml
│   ├── full_finetune_config.yaml
│   └── inference_config.yaml
├── src/                           # Source code
│   ├── data.py                    # Dataset loading, formatting, splits
│   ├── data_curation.py           # Concision curation & distillation labels
│   ├── model.py                   # Model loading with Unsloth + PEFT
│   ├── train.py                   # Training orchestration
│   ├── export_gguf.py             # Merge adapters & export to GGUF
│   ├── inference.py               # HF inference engine
│   ├── benchmark.py               # llama.cpp vs HF benchmark
│   ├── utils.py                   # Reproducibility, path helpers
│   └── logging_utils.py           # Structured logging setup
├── tests/                         # Pytest suite
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_data_curation.py
│   ├── test_model.py
│   └── test_train.py
├── scripts/                       # Bash entry points
│   ├── setup_env.sh
│   ├── run_training.sh
│   ├── run_inference.sh
│   ├── run_benchmark.sh
│   └── run_tests.sh
└── notebooks/                     # Analysis notebooks
    └── training_analysis.ipynb
```

## Quick Start

### 1. Clone & Set Up Environment

```bash
cd programs/qwen2.5-instruction-finetuning

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install Flash Attention 2 (optional, requires CUDA)
pip install flash-attn --no-build-isolation
```

Or use Conda:

```bash
conda env create -f environment.yml
conda activate qwen25-finetune
```

### 2. Configure

Copy the environment template and edit as needed:

```bash
cp .env.example .env
```

Edit `configs/qlora_config.yaml` to select your base model, dataset, and hyperparameters.

### 3. Prepare Data

The training script accepts any Hugging Face dataset with `instruction` and `response` fields (configurable). For a local JSON/JSONL file, place it under `data/` and update the config path.

```bash
python -m src.train --config configs/qlora_config.yaml --prepare_only
```

### 4. Train

```bash
bash scripts/run_training.sh configs/qlora_config.yaml
```

Monitor TensorBoard:

```bash
tensorboard --logdir=outputs/qlora/runs
```

### 5. Export to GGUF

```bash
bash scripts/run_export.sh outputs/qlora/final_checkpoint configs/qlora_config.yaml
```

### 6. Benchmark

```bash
bash scripts/run_benchmark.sh \
  outputs/qlora/final_merged \
  outputs/gguf/qwen25-q4_k_m.gguf \
  configs/inference_config.yaml
```

## Data-Centric Efficiency & Distillation

This project includes optional data-centric methods inspired by recent industry research on training smaller, faster models with fewer resources:

### Concision-Aware Curation (`data.curation`)

Recent work on inference efficiency (e.g. DatologyAI, 2026) shows that **training on concise, correct data** teaches the model to produce shorter outputs without sacrificing accuracy. This lowers *Cost-of-Pass* (FLOPs per correct answer) even when the model size is unchanged.

Enable in `configs/qlora_config.yaml`:

```yaml
data:
  curation:
    enabled: true
    keep_top_fraction: 0.75      # keep the most concise 75% of examples
    max_tokens_per_word: 2.5     # drop overly verbose responses
    min_tokens_per_example: 4
```

### Knowledge Distillation (`data.distillation`)

Following the distillation paradigm used by OpenAI and Anthropic, you can use a larger teacher model to generate chain-of-thought / high-quality targets for the student. The student is then fine-tuned on the teacher outputs, often reaching similar quality with less data and smaller size.

```yaml
data:
  distillation:
    enabled: true
    teacher_model: "Qwen/Qwen2.5-32B-Instruct"
    teacher_revision: "main"
    max_new_tokens: 512
    temperature: 0.7
```

Both curation and distillation are handled by `src/data_curation.py` and applied transparently before training.

## Configuration

All hyperparameters live in `configs/qlora_config.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `base_model` | Hugging Face model id or local path |
| `model` | `max_seq_length` | Maximum sequence length |
| `lora` | `r`, `alpha`, `dropout` | LoRA rank, scale, and dropout |
| `training` | `batch_size`, `gradient_accumulation_steps` | Effective batch = product |
| `training` | `learning_rate`, `num_train_epochs`, `warmup_ratio` | Optimizer schedule |
| `training` | `logging_steps`, `eval_steps`, `save_steps` | Trainer callback cadence |
| `data` | `dataset_name`, `split`, `template` | HF dataset or local path |
| `export` | `quantization_methods` | List of GGUF quants, e.g. `["q4_k_m", "q8_0"]` |

## Training Monitoring

- **TensorBoard**: loss curves, learning rate, gradient norm, eval loss, GPU memory.
- **Checkpoints**: saved every `save_steps` to `outputs/<run_name>/checkpoints`.
- **Early Stopping**: enabled by default with patience of 3 eval steps based on eval loss.

## Export & Quantization

After QLoRA training, adapters are merged into the base model and exported to GGUF using the llama.cpp conversion scripts. Supported quantizations:

- `q4_k_m` — balanced quality/size (recommended)
- `q5_k_m` — higher quality
- `q8_0` — near-lossless, larger size
- `f16` — full 16-bit

The export script pins the llama.cpp revision via submodule or direct clone to ensure reproducibility.

## Benchmarking

`src/benchmark.py` measures:

- **Throughput**: tokens per second (prefill + decode).
- **Peak Memory**: host and device memory via `nvidia-ml-py3` / `psutil`.
- **Perplexity**: on a held-out evaluation prompt set.
- **Quality**: optional ROUGE-L against reference responses.

Results are written to `outputs/benchmarks/<timestamp>/results.json`.

## Testing

Run the full test suite:

```bash
bash scripts/run_tests.sh
```

Run linting and type checks:

```bash
make lint
make typecheck
```

## Docker

Build and run training inside a container:

```bash
docker compose up --build train
```

Run inference:

```bash
docker compose up --build inference
```

## Reproducibility Checklist

- [ ] Use a fixed seed (`training.seed`).
- [ ] Pin dependencies (`requirements.txt`, `environment.yml`).
- [ ] Pin base model revision (`model.revision`).
- [ ] Log config and git SHA at run start.
- [ ] Save final config alongside checkpoint.
- [ ] Use Docker for cross-machine reproducibility.

## License

This project is provided as an educational and research template. Please comply with the license of Qwen2.5 (Apache 2.0) and any datasets you use.

## References & Sources

The innovative efficiency features in this project are grounded in recent research and industry practice:

1. **Concision-aware data curation for inference efficiency**  
   *Brevity is the Soul of Inference Efficiency: Inducing Concision in VLMs via Data Curation* — DatologyAI (2026).  
   [arXiv:2606.25432](https://arxiv.org/abs/2606.25432)  
   Shows that training on concise, correct data lowers *Cost-of-Pass* (FLOPs per correct answer) without shrinking the model.

2. **Value-centric data curation**  
   *A Survey on Efficient Large Language Model Training: From Data-centric Perspectives* — ACL 2025.  
   [ACL Anthology](https://aclanthology.org/2025.acl-long.1493.pdf)  
   Argues that high-information-density sample selection outperforms scale-driven data scaling in low-resource settings.

3. **Knowledge distillation with chain-of-thought / soft labels**  
   OpenAI and Anthropic distillation scaling-law work (2024–2026), and *Distilling Step-by-Step!* (Hsieh et al., 2023).  
   [arXiv:2305.02301](https://arxiv.org/abs/2305.02301)  
   Demonstrates that smaller student models can match larger teachers when trained on high-quality teacher-generated reasoning traces.

## Acknowledgements

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by Alibaba Cloud
- [Unsloth](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
