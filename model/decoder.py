from torch import Tensor, nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # --> [max_len, 1]: chuyển hàng thành cột
        # division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)) # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, text_embeddings: Tensor):
        # text_embeddings: [B, seq_len, d_model]
        return text_embeddings + self.pos_encoding[:, :text_embeddings.size(1)]

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_length, nhead, dropout, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers) # [B, seq_length, d_model]

        self.out_fc = nn.Linear(d_model, vocab_size)

    def forward(self, memory, caption, cap_padding_mask): # caption phải là dạng input_ids
        # caption có dạng [B, seq_len]
        seq_len = caption.size(1)
        device = caption.device # torch.Tensor cho phép lấy device

        tgt_emb = self.pos_encoding(self.embedding(caption))
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device, dtype=bool)

        out = self.decoder(
            tgt = tgt_emb,
            memory = memory,
            tgt_mask = tgt_causal_mask,
            tgt_key_padding_mask = cap_padding_mask,
        )

        logits = self.out_fc(out)      # [B, seq_len, d_model] --> [B, seq_len, vocab_size]
        return logits

