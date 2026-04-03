import os

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, set_seed, TrainerCallback

from data import LingDataCollator, load_data
from model import get_model
from options import parse_args


class GPUMonitorCallback(TrainerCallback):
    """自定义回调：在每个训练步骤后打印GPU显存使用情况"""
    def __init__(self, logging_steps=50):
        self.logging_steps = logging_steps

    def on_step_end(self, args, state, control, **kwargs):
        import torch
        if state.global_step % self.logging_steps == 0 and state.global_step > 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"[Step {state.global_step}] GPU显存: 已用 {allocated:.2f} GB / 预留 {reserved:.2f} GB / 总计 {total:.2f} GB / 剩余 {total - allocated:.2f} GB")


def resolve_output_dir(args):
    if args.ckpt and os.path.isdir(args.ckpt):
        if os.path.basename(args.ckpt).startswith("checkpoint-"):
            return os.path.dirname(args.ckpt)
        return args.ckpt
    return os.path.join(args.ckpt_dir, args.name)


def save_predictions(tokenizer, predictions, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded = [text.strip() for text in decoded]
    with open(output_path, "w") as handle:
        handle.write("\n".join(decoded))
        handle.write("\n")
    return decoded


def main():
    import torch
    import time

    args, _, _ = parse_args()
    if not any([args.do_train, args.do_eval, args.do_predict]):
        args.do_train = True

    # 打印GPU调试信息
    print("=" * 60)
    print("GPU DEBUG INFO")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"  CUDA Capability: {props.major}.{props.minor}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("CUDA不可用!")
    print("=" * 60)

    set_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    data, _, _ = load_data(args, tokenizer)

    output_dir = resolve_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    has_eval = "dev" in data
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=50,
        evaluation_strategy="steps" if args.do_train and has_eval else "no",
        eval_steps=500 if args.do_train and has_eval else None,
        save_strategy="steps" if args.do_train and has_eval else ("epoch" if args.do_train else "no"),
        save_steps=500 if args.do_train and has_eval else None,
        save_total_limit=2,
        load_best_model_at_end=bool(args.do_train and has_eval),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16 if hasattr(args, 'fp16') else False,
        bf16=args.bf16 if hasattr(args, 'bf16') else False,
    )

    model, _, _ = get_model(args, tokenizer, training_args.device)
    collator = LingDataCollator(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=data.get("train"),
        eval_dataset=data.get("dev"),
    )

    # 添加GPU监控回调
    trainer.add_callback(GPUMonitorCallback(logging_steps=50))

    if args.do_train:
        # 训练前GPU信息
        if torch.cuda.is_available():
            print(f"训练前GPU显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("开始训练...")
        train_start = time.time()
        trainer.train(resume_from_checkpoint=args.ckpt if args.ckpt else None)
        train_time = time.time() - train_start
        print(f"训练完成! 总耗时: {train_time:.2f}秒 ({train_time/60:.2f}分钟)")
        if torch.cuda.is_available():
            print(f"训练后GPU显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    if args.do_eval:
        if "dev" not in data:
            raise ValueError("No dev split found for evaluation.")
        dev_metrics = trainer.evaluate(eval_dataset=data["dev"], metric_key_prefix="dev")
        print(dev_metrics)
        if "dev_ood" in data:
            dev_ood_metrics = trainer.evaluate(eval_dataset=data["dev_ood"], metric_key_prefix="dev_ood")
            print(dev_ood_metrics)

    if args.do_predict:
        if args.split not in data:
            raise ValueError(f"Split {args.split!r} not found in dataset.")
        predictions = trainer.predict(data[args.split], metric_key_prefix=args.split).predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        save_predictions(tokenizer, predictions, args.predict_fn)
        print(f"Saved predictions to {args.predict_fn}")


if __name__ == "__main__":
    main()
