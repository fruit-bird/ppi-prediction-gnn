import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List


class ProtBERTEmbedder:
    def __init__(self, model_name="Rostlab/prot_bert_bfd", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def embed_sequence(self, sequence: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Shape: [batch_size, sequence_length, hidden_size]
        # mean averaging of layers
        embedding = outputs.last_hidden_state.mean(axis=1).cpu().numpy()
        return embedding.squeeze()  # Shape: [sequence_length, hidden_size]


if __name__ == "__main__":
    tokenized_seq = list("MVTYDFGSDEMHD")

    embedder = ProtBERTEmbedder()
    embedding = embedder.embed_sequence(tokenized_seq)
    print(embedding.shape)  # This should be [len(sequence), 1024]
    print(embedding)
