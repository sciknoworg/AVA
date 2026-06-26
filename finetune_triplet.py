import os
import argparse
from sentence_transformers.sentence_transformer import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from src.utils import load_json, build_dataset, set_seed, MemoryCleanupCallback, make_evaluator, make_losses

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a SentenceTransformer with TripletLoss."
    )
    parser.add_argument("--train_path", type=str, default="data/train.json")
    # parser.add_argument("--validation_path", type=str, default="data/validation.json")
    parser.add_argument("--model_id", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output_dir", type=str, default="assets/mpnet-triplet")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--hyperbolic", action="store_true")
    parser.add_argument("--w_hyperbolic", type=float, default=0.1)
    parser.add_argument("--hyperbolic_c", type=float, default=0.1)
    parser.add_argument("--dpo", action="store_true")
    parser.add_argument("--rl_beta", type=float, default=0.1)
    parser.add_argument("--w_dpo", type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"File not found for --{args.train_path}")

    print(f"Loading training data from   : {args.train_path}")
    # print(f"Loading validation data from : {args.validation_path}")
    train_records      = load_json(args.train_path)
    # validation_records = load_json(args.validation_path)

    train_dataset      = build_dataset(train_records)
    # validation_dataset = build_dataset(validation_records)

    print(f"Train samples      : {len(train_dataset)}")
    # print(f"Validation samples : {len(validation_dataset)}")


    print(f"\nLoading model: {args.model_id}")
    model_kwargs = {
        "trust_remote_code": True,
        "model_kwargs": {"attn_implementation": "eager"}
    }
    model       = SentenceTransformer(args.model_id, **model_kwargs)
    ref_model   = None
    if args.dpo:
        ref_model = SentenceTransformer(args.model_id, **model_kwargs)

    triplet_loss = make_losses(model=model,
                               ref_model=ref_model,
                               use_hyperbolic=args.hyperbolic,
                               use_dpo=args.dpo,
                               rl_beta=args.rl_beta,
                               triplet_margin=args.margin,
                               c=args.hyperbolic_c,
                               w_dpo=args.w_dpo,
                               w_hyperbolic=args.w_hyperbolic)
    # evaluator = make_evaluator(eval_dataset=validation_dataset, eval_bs=args.per_device_eval_batch_size)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        greater_is_better=False,
        report_to="tensorboard",
        seed=args.seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset={"hyperbolic": train_dataset, "dpo": train_dataset} if args.dpo and args.hyperbolic else train_dataset,
        # eval_dataset=validation_dataset,
        loss=triplet_loss,
        # evaluator=evaluator,
        callbacks=[MemoryCleanupCallback()],
    )
    print("\nStarting training …")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    print(f"\nSaved final model to: {final_dir}")


if __name__ == "__main__":
    main()

