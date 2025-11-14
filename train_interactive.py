import os
from pathlib import Path
import argparse

import torch
import sentencepiece as spm
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, TensorDataset
import config
from transformer import Transformer

script_dir = Path(__file__).resolve().parent
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'interactive'], default='train')
    return parser.parse_args()

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

def main():
    args = parse_args()

    raw_texts = dataset["train"]['text']
    # if tokenizer model exists, load it, otherwise train
    spm_model = script_dir / "spm.model"
    if not spm_model.exists():
        print("Training SentencePiece tokenizer (this may take a while)...")
        _train_sentencepiece(raw_texts, vocab_size=16000)

    sp = _load_spm(spm_model)
    vocab_size = sp.vocab_size()

    # Initialize model
    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_blocks=config.NUM_BLOCKS
    ).to(config.DEVICE)

    if args.mode == 'train':
        encoded_ids = _encode_corpus_to_ids(sp, raw_texts)
        train_loader, val_loader = _prepare_training_data(
            encoded_ids=encoded_ids,
            batch_size=config.BATCH_SIZE,
            token_sequence_length=config.TOKEN_SEQUENCE_LENGTH
        )
        model.fit(train_loader, val_loader, epochs=config.EPOCHS)
        print("Training finished!")

    elif args.mode == 'interactive' and config.MODEL_PATH.exists():
        print("Entering interactive mode (type 'exit' to quit)")
        ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model from checkpoint at epoch {ckpt.get('epoch', 'unknown')}")
        while True:
            prompt = input("Prompt> ")
            if prompt.lower() in ["exit", "quit"]:
                break
            output = model.generate(
                sp,
                prompt,
                num_steps=10
            )
            print("Output:", output)


if __name__ == '__main__':
    main()
