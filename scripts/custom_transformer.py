import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        tgt_vocab_size, 
        d_model=512, 
        nhead=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dim_feedforward=2048, 
        dropout=0.1
    ):
        super().__init__()
        
        self.model_type = 'Transformer'
        self.src_mask = None
        self.d_model = d_model
        
        self.embedding_src = nn.Embedding(src_vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.out = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding_src.weight.data.uniform_(-initrange, initrange)
        self.embedding_tgt.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        src = self.embedding_src(src) * math.sqrt(self.d_model)
        tgt = self.embedding_tgt(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(
            src, 
            tgt, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask, 
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.out(output)

    def create_mask(self, src, tgt, pad_token_id=0):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

        src_padding_mask = (src == pad_token_id)
        tgt_padding_mask = (tgt == pad_token_id)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
