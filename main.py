import math
import os
import time
from pathlib import Path

import nltk
import torch
import sentencepiece as spm
from matplotlib import pyplot as plt

import config
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from datasets import load_dataset

script_dir = Path(__file__).resolve().parent
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
unknown_token = "<UNK>"
padding_token = "<PAD>"
loss_curve_path = script_dir / "loss_curve.png"
model_path = script_dir / "model.pth"


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TransformerBlock, self).__init__()
        head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.W_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.W_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.W_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.W_o = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        out_features_first_ffn_step = 4 * embedding_dim
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=out_features_first_ffn_step),
            nn.ReLU(),
            nn.Linear(in_features=out_features_first_ffn_step, out_features=embedding_dim)
        )
        self.ffn_dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Key, query, value matrices - multi-head
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # dot product attention, query - key
        attention_scores = Q @ K.transpose(-2, -1)  # transpose K so we can perform matrix multiplication (inner dimensions have to be the same)
        attention_scores = attention_scores / math.sqrt(self.head_dim)  # normalize so values dont explode

        # causal mask - applied so model doesn't use future tokens to adjust behaviour
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # apply softmax to attention_scores to get a sort of probability distribution
        softmax_attention_scores = F.softmax(attention_scores, dim=-1)

        # calculate context using value matrix and softmax attention scores
        context_vector = softmax_attention_scores @ V
        context_vector = self._combine_heads(context_vector)
        context_vector = self.W_o(context_vector)
        context_vector = self.attn_dropout(context_vector)

        # attention residual connection
        x = self.norm1(x + context_vector)

        # ffn
        ffn_out = self.ffn(x)
        ffn_out = self.ffn_dropout(ffn_out)

        # residual connection
        x = self.norm2(x + ffn_out)
        return x

    def _split_heads(self, tensor):
        batch_size, seq_len, embed_dim = tensor.shape
        # [batch, seq_len, num_heads, head_dim]
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # [batch, num_heads, seq_len, head_dim] for attention computation
        return tensor.permute(0, 2, 1, 3)

    def _combine_heads(self, tensor):
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        # [batch, seq_len, num_heads, head_dim]
        tensor = tensor.permute(0, 2, 1, 3)
        # flatten last two dims: [batch, seq_len, embedding_dim]
        return tensor.contiguous().view(batch_size, seq_len, num_heads * head_dim)


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_blocks):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        self.pos_embedding = nn.Embedding(num_embeddings=config.TOKEN_SEQUENCE_LENGTH, embedding_dim=embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embedding_dim, num_heads) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(in_features=embedding_dim, out_features=vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight #tie weights from embedding matrix to output matrix - improves compute time
        self.patience = 5

    def forward(self, x):
        # Embeddings
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x)

        return self.output_layer(x)


    def fit(self, train_loader, val_loader, epochs):
        start_time = time.time()
        optimizer = optim.AdamW(self.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(device='cuda')

        # Warmup + cosine decay scheduler
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.05 * total_steps) # 5% warmup

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        trigger_times = 0
        best_val_loss = float('inf')
        global_step = 0

        train_losses, val_losses = [], []
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.train()
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                x, y = batch
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)

                #Compute unscaled loss for logging
                with torch.cuda.amp.autocast():
                    logits = self(x)
                    logits = logits.view(-1, self.vocab_size)  # [B*L, vocab_size]
                    y = y.view(-1)  # [B*L]
                    raw_loss = loss_fn(logits, y)

                # scale for accumulation, then call backward
                loss = raw_loss / config.GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()

                # optimizer step only every grad_accum steps
                if (step + 1) % config.GRAD_ACCUM_STEPS == 0:
                    # clip grads on the *unscaled* gradients (done after scaler.unscale_ below)
                    scaler.unscale_(optimizer)  # bring grads into fp32 for clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += raw_loss.item() * x.size(0)

            # handle leftover gradients at epoch end (if dataset size is not divisible by grad_accum steps)
            if (step + 1) % config.GRAD_ACCUM_STEPS != 0:
                # there are remaining grads that weren't stepped yet
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            avg_train_loss = epoch_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            #validation
            val_loss = self._compute_validation_loss(val_loader, loss_fn)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                trigger_times = 0
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, model_path)
                print(f"Saved checkpoint with val loss {best_val_loss:.4f} at epoch {epoch}")
            else:
                trigger_times += 1
                print(f"No improvement in val loss for {trigger_times} epochs.")
                if trigger_times >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print(f"Epoch {epoch}, Train loss: {avg_train_loss: .4f}, Val loss: {val_loss: .4f}")

        # load the best checkpoint if exists
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=config.DEVICE)
            self.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded best model from checkpoint saved at epoch {ckpt.get('epoch', 'unknown')} with val loss {ckpt.get('best_val_loss', 'unknown')}")
        _plot_loss_curve(train_losses, val_losses)
        end_time = time.time()
        print(f"Model trained in {end_time - start_time:.4f} seconds")

    def _compute_validation_loss(self, val_loader, loss_fn):
        total_loss = 0.0
        self.eval()
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                with torch.cuda.amp.autocast():
                    logits = self(batch_X)
                    logits = logits.view(-1, self.vocab_size)  # [B*L, vocab_size]
                    batch_y = batch_y.view(-1)  # [B*L]
                    loss = loss_fn(logits, batch_y)
                total_loss += loss.item() * batch_X.size(0)

        total_loss = total_loss / len(val_loader.dataset)
        return total_loss

def _plot_loss_curve(train_loss, val_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Saved loss curve to {loss_curve_path}")


def _prepare_training_data(encoded_ids, batch_size, token_sequence_length):
    x, y = [], []
    for i in range(0, len(encoded_ids) - token_sequence_length, token_sequence_length):
        x.append(encoded_ids[i: i + token_sequence_length])
        y.append(encoded_ids[i + 1: i + token_sequence_length + 1])

    tensor_x = torch.tensor(x, dtype=torch.long)
    tensor_y = torch.tensor(y, dtype=torch.long)

    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    train_size = int(0.8 * len(tensor_dataset))
    val_size = len(tensor_dataset) - train_size
    train_dataset, val_dataset = random_split(tensor_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader

def _train_sentencepiece(input_texts, model_prefix="spm", vocab_size=16000, model_type="bpe"):
    tmp_path = script_dir / f"{model_prefix}_train.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in input_texts:
            # write raw lines; you can also join paragraphs etc.
            f.write(line.replace("\n", " ") + "\n")

    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        model_prefix=str(script_dir / model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=""  # you can add tokens like "<mask>"
    )

    os.remove(tmp_path)
    return (script_dir / f"{model_prefix}.model"), (script_dir / f"{model_prefix}.vocab")

def _load_spm(spm_model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model_path))
    return sp

def _encode_corpus_to_ids(sp, texts):
    ids = []
    for txt in texts:
        if not txt:
            continue
        # encode as ints
        piece_ids = sp.encode(txt, out_type=int)
        ids.extend(piece_ids)
    return ids


def main():
    raw_texts = dataset["train"]['text']

    # if tokenizer model exists, load it, otherwise train
    spm_model = script_dir / "spm.model"
    if not spm_model.exists():
        print("Training SentencePiece tokenizer (this may take a while)...")
        _train_sentencepiece(raw_texts, vocab_size=16000)

    sp = _load_spm(spm_model)
    vocab_size = sp.vocab_size()

    # encode dataset into integers
    ids = _encode_corpus_to_ids(sp, raw_texts)

    train_loader, val_loader = _prepare_training_data(encoded_ids=ids, batch_size=config.BATCH_SIZE, token_sequence_length=config.TOKEN_SEQUENCE_LENGTH)

    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_blocks=config.NUM_BLOCKS
    ).to(config.DEVICE)

    model.fit(train_loader, val_loader, epochs=config.EPOCHS)


if __name__ == '__main__':
    main()
