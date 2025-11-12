import nltk
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
unknown_token = "<UNK>"
padding_token = "<PAD>"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token_sequence_length = 128


class Transformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=token_sequence_length, embedding_dim=embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        positions = torch.arange(token_sequence_length).unsqueeze(0).to(device)
        x = x + self.pos_embedding(positions)


def _load_training_data_tokenized():
    train_data_test = dataset["train"]['text']
    train_text = "\n".join(train_data_test)  # converts list of strings into a newline separated string
    return word_tokenize(train_text)

def _create_vocabulary(tokenized_training_data):
    vocab = {padding_token: 0, unknown_token: 1}
    next_index = 2
    for token in tokenized_training_data:
        if token not in vocab:
            vocab[token] = next_index
            next_index += 1

    return vocab

def _encode_text(tokenized_training_data, vocab):
    encoded_text = []
    for token in tokenized_training_data:
        if token not in vocab:
            encoded_text.append(vocab.get(unknown_token))
        else:
            encoded_text.append(vocab.get(token))

    return encoded_text

def _prepare_training_data(encoded_text, batch_size=64):
    x = []
    y = []
    for i in range(0, len(encoded_text), token_sequence_length):
        if len(encoded_text) - i < token_sequence_length:
            break
        x.append(encoded_text[i: i + token_sequence_length])
        y.append(encoded_text[i + 1: i + token_sequence_length + 1])

    tensor_x = torch.tensor(x, dtype=torch.long)
    tensor_y = torch.tensor(y, dtype=torch.long)

    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def main():
    tokenized_training_data = _load_training_data_tokenized()
    vocab = _create_vocabulary(tokenized_training_data)
    encoded_text = _encode_text(tokenized_training_data, vocab)

if __name__ == '__main__':
    main()
