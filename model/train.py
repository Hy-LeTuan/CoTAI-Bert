from modeling_new_bert import CotaiBert, Hfwrapper
from transformers import DataCollatorForLanguageModeling, Trainer, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from datasets import load_from_disk
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def compute_metrics():
    def compute(eval_preds):
        model_output, labels = eval_preds

        # get logits and mask map from model
        logits = model_output[0]
        labels_from_model = model_output[2]

        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_from_model,
            predictions,
            average="weighted",
            zero_division=0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return compute


if __name__ == "__main__":
    original_max_position_embeddings = 256
    max_position_embeddings = 2048
    num_heads = 9
    num_kv_heads = 3
    hidden_state = 576
    mlp_dim = 1536
    head_dim = hidden_state // num_heads
    base = 10000
    device = "cuda"
    flash = True
    num_blocks = 15
    yarn = False
    ntk = False
    scaling_factor = 1.0
    immediate = False
    mask_id = 52290
    alpha = 1
    beta = 32
    scale = 16
    mscale = 0.707
    mlm_probability = 0.3

    model = CotaiBert(num_blocks,
                      hidden_state,
                      mlp_dim,
                      num_heads,
                      num_kv_heads,
                      base,
                      flash=flash,
                      device=device,
                      original_max_position_embeddings=original_max_position_embeddings,
                      max_position_embeddings=max_position_embeddings,
                      scale=scale,
                      beta=beta,
                      alpha=alpha,
                      mscale=mscale,
                      scaling_factor=scaling_factor,
                      yarn=yarn,
                      ntk=ntk,
                      mask_id=mask_id,
                      immediate=immediate).to(device)

    wrapper = Hfwrapper(model)

    optimizer = torch.optim.AdamW(lr=2e-5, params=wrapper.model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(
        "../tokenizer/trained_tokenizer/tokenizer-50k"
    )

    train_dataset = load_from_disk(
        "../../../NLPLearn/visobert-token-classification/data/tokenized_dataset_train"
    )
    val_dataset = load_from_disk(
        "../../../NLPLearn/visobert-token-classification/data/tokenized_dataset_val"
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        return_tensors="pt"
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=1e-4
    )

    output_dir = f"results_{''.join(str(mlm_probability).split('.'))}"

    if ntk:
        output_dir += "_ntk"
    elif yarn:
        output_dir += "_yarn"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/runs/",
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        logging_strategy="steps",
        logging_steps=0.02,
        torch_compile=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=14,
        bf16=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=8,
        max_steps=70000,  # high max steps for early stopping,
        save_strategy="steps",
        save_steps=0.02,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        torch_compile_backend="inductor",
    )
    trainer = Trainer(
        args=training_arguments,
        model=wrapper,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        callbacks=[early_stopping_callback],
        compute_metrics=compute_metrics()
    )

    trainer.train()
