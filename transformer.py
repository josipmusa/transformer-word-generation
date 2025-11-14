import math
import time

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim

import config

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

    def generate(self, sp, prompt, num_steps, temperature=0.8, top_k=50):
        self.eval()
        input_tokens = sp.encode(prompt, out_type=int)
        input_tensor = torch.tensor([input_tokens], device=config.DEVICE)
        for i in range(num_steps):
            with torch.no_grad():
                logits = self(input_tensor) #shape is [batch_size, seq_len, vocab_size] - we need last vector from seq_len
                logits_last = logits[:, -1, :] # sahpe is [1, vocab_size]

                # apply temperature
                logits_last = logits_last / temperature

                # top-k filtering
                if top_k is not None:
                    topk_vals, topk_indices = torch.topk(logits_last, top_k, dim=-1)
                    probs = torch.zeros_like(logits_last).scatter_(-1, topk_indices, F.softmax(topk_vals, dim=-1))
                else:
                    probs = F.softmax(logits_last, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat([input_tensor, next_token], dim=1)

        output_tokens = input_tensor[0].tolist()
        return sp.decode(output_tokens)

    def fit(self, train_loader, val_loader, epochs):
        start_time = time.time()
        optimizer = optim.AdamW(self.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(device='cuda')

        # Warmup + cosine decay scheduler
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.05 * total_steps)  # 5% warmup

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

                # Compute unscaled loss for logging
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

            # validation
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
                torch.save(checkpoint, config.MODEL_PATH)
                print(f"Saved checkpoint with val loss {best_val_loss:.4f} at epoch {epoch}")
            else:
                trigger_times += 1
                print(f"No improvement in val loss for {trigger_times} epochs.")
                if trigger_times >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print(f"Epoch {epoch}, Train loss: {avg_train_loss: .4f}, Val loss: {val_loss: .4f}")

        # load the best checkpoint if exists
        if config.MODEL_PATH.exists():
            ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
            self.load_state_dict(ckpt['model_state_dict'])
            print(
                f"Loaded best model from checkpoint saved at epoch {ckpt.get('epoch', 'unknown')} with val loss {ckpt.get('best_val_loss', 'unknown')}")
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
    plt.savefig(config.LOSS_CURVE_PATH)
    plt.close()
    print(f"Saved loss curve to {config.LOSS_CURVE_PATH}")