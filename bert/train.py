from modeling_new_bert import CotaiBert, Hfwrapper
from transformers import DataCollatorForLanguageModeling, Trainer, AutoTokenizer, TrainingArguments
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
    ntk = True
    scaling_factor = 1.0
    immediate = False
    mask_id = 52290
    alpha = 1
    beta = 32
    scale = 16
    mscale = 0.707

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
    collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=True,
        mlm_probability=0.3,
        return_tensors="pt"
    )
    training_arguments = TrainingArguments(
        output_dir="./results" if not ntk else "./results_ntk",
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=0.2,
        torch_compile=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=14,
        bf16=True,
        per_device_train_batch_size=14,
        per_device_eval_batch_size=14,
        eval_accumulation_steps=14,
        max_steps=20000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        torch_compile_backend="inductor",
    )

    train_dataset = load_from_disk(
        "../../../NLPLearn/visobert-token-classification/data/tokenized_dataset_train"
    )
    val_dataset = load_from_disk(
        "../../../NLPLearn/visobert-token-classification/data/tokenized_dataset_val"
    )

    trainer = Trainer(
        args=training_arguments,
        model=wrapper,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics(tokenizer)
    )
    trainer.train()
