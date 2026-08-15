from collections import Counter
import re
from typing import Iterable, Sequence


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.stoi: dict = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        self.itos: dict = {index: string for string, index in self.stoi.items()}

    @property
    def pad_token_id(self):
        return self.stoi[self.PAD_TOKEN]

    @property
    def sos_token_id(self):
        return self.stoi[self.SOS_TOKEN]

    @property
    def eos_token_id(self):
        return self.stoi[self.EOS_TOKEN]

    @property
    def unk_token_id(self):
        return self.stoi[self.UNK_TOKEN]

    def __len__(self):
        return len(self.stoi)

    def normalize_text(self, text: str):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)  
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str):
        return self.normalize_text(text).split()

    def build(self, text_list: Iterable[str]):
        counter = Counter()
        for text in text_list:
            tokens = self.tokenize(text)
            for token in tokens:
                counter.update([token])

        idx = self.__len__()
        for token, freq in counter.items():
            if(freq >= self.min_freq) and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1 

    def build_from_tokens(self, tokens_list: Iterable[Sequence[str]]):
        counter = Counter()

        for tokens in tokens_list:
            counter.update(tokens)

        idx = self.__len__()
        for token, freq in counter.items():
            if(freq >= self.min_freq) and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1 

    def encode(self, text: str, max_len: int):
        tokens = self.tokenize(text)
        tokens = tokens[:max_len-2] # chừa chỗ cho <SOS> và <EOS>

        input_ids = [self.stoi[self.SOS_TOKEN]]

        for token in tokens:
            idx = self.stoi[token] if token in self.stoi else self.stoi[self.UNK_TOKEN]
            input_ids.append(idx)

        input_ids.append(self.stoi[self.EOS_TOKEN])

        output_len = len(input_ids)

        attention_mask = [False] * output_len # True là token <PAD>

        for _ in range(output_len, max_len):
            input_ids.append(self.stoi[self.PAD_TOKEN])
            attention_mask.append(True)

        return input_ids, attention_mask

    def encode_from_tokens(self, tokens: Sequence[str], max_len: int):
        tokens = tokens[:max_len-2] # chừa chỗ cho SOS và EOS

        input_ids = [self.sos_token_id]
        for token in tokens:
            idx = self.stoi[token] if token in self.stoi else self.unk_token_id
            input_ids.append(idx)

        input_ids.append(self.eos_token_id)
        output_len = len(input_ids)
        attention_mask = [False] * output_len

        for _ in range(output_len, max_len):
            input_ids.append(self.pad_token_id)
            attention_mask.append(True)

        return input_ids, attention_mask


    def decode(self, input_ids: Iterable[int]):
        tokens = []

        for idx in input_ids:
            token = self.itos.get(int(idx), self.UNK_TOKEN)

            if token == self.EOS_TOKEN:
                break

            if token not in {self.PAD_TOKEN, self.SOS_TOKEN}:
                tokens.append(token)

        return " ".join(tokens)

