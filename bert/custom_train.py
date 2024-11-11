from modeling_new_bert import CotaiBert, Hfwrapper
from datasets import load_from_disk
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

class MLMDataset(Dataset):
    def __init__(self, dataset, mlm_probability=0.3, mask_id=52290):
        self.dataset = dataset
        self.mlm_probability = mlm_probability
        self.mask_id = mask_id

    def __len__(self):
        return len(self.dataset)

    def masking(self, input_ids):
        num_mask = round((input_ids.shape[-1] - 2) * self.mlm_probability)
        masked_positions = torch.randint(1, input_ids.shape[-1], size=(num_mask,))
        masked_input_ids = torch.clone(input_ids).detach().to(input_ids.device)
        masked_input_ids[masked_positions] = self.mask_id

        return masked_input_ids

    def __getitem__(self, index):
        input_ids = torch.tensor(self.dataset[index]["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(self.dataset[index]["attention_mask"], dtype=torch.long)
        masked_input_ids = self.masking(input_ids)

        return {
            "attention_mask": attention_mask,
            "input_ids": masked_input_ids,
            "labels": input_ids,
        }

def collate_fn_with_args(pad_id):
    def collate_fn(batch):
        # getting all items inside a batch
        input_ids = []
        labels = []
        attention_mask = []
        for item in batch:
            input_ids.append(item["input_ids"])
            labels.append(item["labels"])
            attention_mask.append(item["attention_mask"])

        # create paddings
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    return collate_fn

def train(model, train_dataset, test_dataset, optimizer, scheduler, pad_id=52289, steps=10, batch_size=12, eval_steps=500, save_path="./result/model_state_dict.pt", device="cuda"):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_args(pad_id=pad_id))
    test_loader = DataLoader(test_dataset, batch_size=batch_size // 4, shuffle=True, collate_fn=collate_fn_with_args(pad_id=pad_id))
    p_bar = tqdm(range(steps))
    best_loss = float("inf")
    best_val_loss = float("inf")

    train_loader_iter = iter(train_loader)

    model.train()
    for index in p_bar:
        # evaluation loop
        if index != 0 and index % eval_steps == 0:
            model.eval()
            total_f1 = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_loss = 0.0

            for batch in tqdm(test_loader):
                output = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device)
                )
                loss = output[0]
                logits = output[1].to("cpu").detach().numpy()
                mask_map = output[2].to("cpu")
                labels = output[3].to("cpu").detach().numpy()

                # calculate predictions
                predictions = np.argmax(logits, axis=-1)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0.0
                )

                total_precision += precision
                total_recall += recall
                total_f1 += total_f1
                total_loss += loss.item()

            total_precision /= len(test_loader)
            total_recall /= len(test_loader)
            total_f1 /= len(test_loader)
            total_loss /= len(test_loader)

            if total_loss < best_val_loss:
                best_val_loss = total_loss
                torch.save(model.state_dict(), save_path)

            print(f"evaluation loop | precision: {total_precision} | recall: {total_recall} | f1: {total_f1} | loss: {total_loss}")
            model.train()

        # training loop
        else:
            try:
                dataset_output = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                dataset_output = next(train_loader_iter)

            output = model(
                dataset_output["input_ids"].to(device),
                dataset_output["attention_mask"].to(device),
                dataset_output["labels"].to(device)
            )

            # calculate loss
            loss = output.loss

            # scheduler step to reduce learning rate
            scheduler.step(loss.item())

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                p_bar.set_description(f"loss improved to: {best_loss} | lr: {scheduler.get_last_lr()}", refresh=True)
            else:
                p_bar.set_description(f"best loss: {best_loss} | lr: {scheduler.get_last_lr()}")

if __name__ == "__main__":
    original_max_position_embeddings = 256
    max_position_embeddings = 2048
    num_heads = 9
    num_kv_heads = 3
    hidden_state = 576
    mlp_dim = 564  # 1536
    head_dim = hidden_state // num_heads
    base = 10000
    device = "cuda"
    flash = False
    num_blocks = 10  # 15
    yarn = False
    ntk = False
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
    train_dataset = MLMDataset(dataset=load_from_disk   (
        "/home/hyle/Documents/vscode/NLPLearn/visobert-token-classification/data/tokenized_dataset_train"))
    val_dataset = MLMDataset(dataset=load_from_disk(
        "/home/hyle/Documents/vscode/NLPLearn/visobert-token-classification/data/tokenized_dataset_val"))

    optimizer = torch.optim.AdamW(params=wrapper.parameters(), lr=3e-5)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.98, cooldown=100)

    train(model=wrapper, train_dataset=train_dataset, test_dataset=val_dataset, optimizer=optimizer, scheduler=scheduler, batch_size=8, eval_steps=200, steps=15000)
