"""Training orchestration for Qwen2.5 instruction fine-tuning."""

from __future__ import annotations

import argparse
from typing import Any

from datasets import DatasetDict
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data import build_dataset, format_dataset, load_instruction_dataset
from src.data_curation import build_distillation_dataset
from src.logging_utils import get_logger, setup_logging
from src.model import build_lora_model, load_base_model
from src.utils import (
    count_trainable_parameters,
    get_output_dir,
    load_config,
    save_config,
    set_seed,
)

logger = get_logger(__name__)


def build_training_arguments(config: dict[str, Any]) -> TrainingArguments:
    """Create Transformers TrainingArguments from config.

    Args:
        config: Training configuration dictionary.

    Returns:
        Configured TrainingArguments.
    """
    training_cfg = config["training"]
    output_dir = get_output_dir(config)

    return TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        run_name=training_cfg.get("run_name"),
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        max_steps=training_cfg.get("max_steps", -1),
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg.get("warmup_ratio", 0.0),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=training_cfg.get("logging_steps", 10),
        eval_strategy=training_cfg.get("eval_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 100),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_cfg.get("greater_is_better", False),
        seed=training_cfg.get("seed", 42),
        bf16=training_cfg.get("bf16", False),
        fp16=training_cfg.get("fp16", False),
        optim=training_cfg.get("optim", "adamw_torch"),
        group_by_length=training_cfg.get("group_by_length", False),
        report_to=training_cfg.get("report_to", "tensorboard"),
        remove_unused_columns=training_cfg.get("remove_unused_columns", False),
        logging_dir=str(output_dir / "runs"),
        logging_first_step=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


def train(config: dict[str, Any]) -> Trainer:
    """Run instruction fine-tuning from a config dictionary.

    Args:
        config: Full project configuration.

    Returns:
        Trained Transformers Trainer instance.
    """
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    output_dir = get_output_dir(config)
    setup_logging(log_file=output_dir / "train.log")
    logger.info("Starting training with config:\n%s", config)
    save_config(config, output_dir / "config.yaml")

    logger.info("Loading base model: %s", config["model"]["base_model"])
    model, tokenizer = load_base_model(
        model_name=config["model"]["base_model"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=config["model"].get("dtype"),
        load_in_4bit=config["model"].get("load_in_4bit", True),
        use_gradient_checkpointing=config["model"].get("use_gradient_checkpointing", True),
        trust_remote_code=config["model"].get("trust_remote_code", False),
        revision=config["model"].get("revision", "main"),
    )

    logger.info("Applying LoRA adapters")
    model = build_lora_model(model, config["lora"])
    trainable, total = count_trainable_parameters(model)
    logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # Optional: load a larger teacher model for distillation.
    teacher_model = None
    distillation_cfg = config["data"].get("distillation", {})
    if distillation_cfg.get("enabled", False) and distillation_cfg.get("teacher_model"):
        logger.info("Loading teacher model for distillation: %s", distillation_cfg["teacher_model"])
        teacher_model, _ = load_base_model(
            model_name=distillation_cfg["teacher_model"],
            max_seq_length=config["model"]["max_seq_length"],
            dtype=config["model"].get("dtype"),
            load_in_4bit=config["model"].get("load_in_4bit", True),
            use_gradient_checkpointing=False,
            trust_remote_code=config["model"].get("trust_remote_code", False),
            revision=distillation_cfg.get("teacher_revision", "main"),
        )

    # Optional: load a reference model for perplexity-based quality scoring.
    quality_model = None
    curation_cfg = config["data"].get("curation", {})
    quality_cfg = curation_cfg.get("quality_scoring", {})
    if curation_cfg.get("enabled", False) and quality_cfg.get("enabled", False) and quality_cfg.get("model"):
        logger.info("Loading quality model for curation: %s", quality_cfg["model"])
        quality_model, _ = load_base_model(
            model_name=quality_cfg["model"],
            max_seq_length=config["model"]["max_seq_length"],
            dtype=config["model"].get("dtype"),
            load_in_4bit=config["model"].get("load_in_4bit", True),
            use_gradient_checkpointing=False,
            trust_remote_code=config["model"].get("trust_remote_code", False),
            revision=quality_cfg.get("revision", "main"),
        )

    logger.info("Building datasets")
    raw_dataset = load_instruction_dataset(
        dataset_name=config["data"]["dataset_name"],
        config_name=config["data"].get("dataset_config"),
        split=config["data"].get("split", "train"),
        cache_dir=config["data"].get("cache_dir"),
        streaming=config["data"].get("streaming", False),
        max_samples=config["data"].get("max_samples"),
    )

    eval_size = config["data"].get("eval_size", 0.05)
    streaming = config["data"].get("streaming", False)
    if eval_size and eval_size > 0 and not streaming:
        split_result = raw_dataset.train_test_split(test_size=eval_size, seed=seed)
        train_dataset_raw = split_result["train"]
        eval_dataset_raw = split_result["test"]
    else:
        train_dataset_raw = raw_dataset
        eval_dataset_raw = None

    # Apply concision / quality curation and optionally teacher-label distillation
    # on the *raw* dataset so fields like instruction/input/output are available.
    if curation_cfg.get("enabled", False) or distillation_cfg.get("enabled", False):
        train_dataset_raw = build_distillation_dataset(
            dataset=train_dataset_raw,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            data_config=config["data"],
            quality_model=quality_model,
        )
        logger.info("After curation/distillation: %d train examples", len(train_dataset_raw))

    target_field = "teacher_output" if distillation_cfg.get("enabled", False) else None
    dataset = DatasetDict({
        "train": format_dataset(
            dataset=train_dataset_raw,
            data_config=config["data"],
            tokenizer=tokenizer,
            target_field=target_field,
        ),
    })
    if eval_dataset_raw is not None:
        dataset["validation"] = format_dataset(
            dataset=eval_dataset_raw,
            data_config=config["data"],
            tokenizer=tokenizer,
            target_field=None,
        )

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        texts = examples[config["training"].get("dataset_text_field", "text")]
        return tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=config["model"]["max_seq_length"],
            return_tensors=None,
        )

    logger.info("Tokenizing datasets")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset.get("validation")

    training_args = build_training_arguments(config)
    callbacks: list[Any] = []
    if config["training"].get("early_stopping_patience") is not None and eval_dataset is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["training"]["early_stopping_patience"]
            )
        )

    # Ensure tensorboard logging directory exists.
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Beginning training")
    trainer.train()

    logger.info("Saving final model to %s", output_dir / "final_checkpoint")
    trainer.save_model(output_dir / "final_checkpoint")
    tokenizer.save_pretrained(output_dir / "final_checkpoint")

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        logger.info("Final eval metrics: %s", metrics)

    return trainer


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Instruct")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare and print dataset")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prepare_only:
        _, tokenizer = load_base_model(
            model_name=config["model"]["base_model"],
            max_seq_length=config["model"]["max_seq_length"],
            dtype=config["model"].get("dtype"),
            load_in_4bit=config["model"].get("load_in_4bit", True),
            use_gradient_checkpointing=config["model"].get("use_gradient_checkpointing", True),
            trust_remote_code=config["model"].get("trust_remote_code", False),
            revision=config["model"].get("revision", "main"),
        )
        dataset = build_dataset(
            data_config=config["data"],
            tokenizer=tokenizer,
            eval_size=config["data"].get("eval_size", 0.05),
            seed=config["training"].get("seed", 42),
        )
        print(f"Train examples: {len(dataset['train'])}")
        if "validation" in dataset:
            print(f"Validation examples: {len(dataset['validation'])}")
        print("Sample formatted example:")
        print(dataset["train"][0]["text"][:500])
        return

    train(config)


if __name__ == "__main__":
    main()
