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
        dropout=0.1,
        use_alibi=False
    ):
        super().__init__()
        
        self.model_type = 'Transformer'
        self.src_mask = None
        self.d_model = d_model
        self.use_alibi = use_alibi
        self.nhead = nhead
        
        self.embedding_src = nn.Embedding(src_vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, d_model)
        
        if not use_alibi:
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_encoder = None
            self.dropout_layer = nn.Dropout(dropout)
        
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
        
        if self.use_alibi:
            src = self.dropout_layer(src)
            tgt = self.dropout_layer(tgt)
            
            # Generate ALiBi bias for Source-to-Source attention (Encoder)
            # Shapes: src_mask provided is usually boolean [src_len, src_len].
            # Here we need float mask for bias integration.
            # If src_mask is None or all zeros, we just use ALiBi. 
            # If it's provided (e.g. for causal masking, though src is usually bidirectional), we add to it.
            
            if src_mask is None:
                src_seq_len = src.shape[1]
                # [nhead, seq_len, seq_len]
                alibi_bias = self.get_alibi_mask(src_seq_len, self.nhead, src.device)
                
                # Expand for batch size is handled by broadcasting in PyTorch MultiheadAttention usually:
                # attn_mask shape: (N * num_heads, L, S) or (num_heads, L, S). 
                # nn.Transformer expects (S, S) or (Batch*num_heads, S, S)
                # We need (batch_size * nhead, src_seq_len, src_seq_len) to be safe or rely on broadcasting.
                # However, PyTorch transformer allows (nhead*batch, L, S).
                bs = src.shape[0]
                src_mask = alibi_bias.repeat(bs, 1, 1) # [bs*nhead, seq_len, seq_len]
            
            # Note: For simplicity in this iteration, we only apply ALiBi to encoder self-attention (src_mask).
            # Decoder also benefits, but usually relative PE is most critical for Encoder.
            # Implementing full encoder-decoder ALiBi correctly requires modifying cross-attention which is tricky with nn.Transformer high-level API
            # without hacking mask logic significantly. We Stick to Encoder self-attention ALiBi for now.
            
        else:
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
        
    def get_alibi_mask(self, seq_len, nhead, device):
        # ALiBi: get slopes for each head
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   
            else:                                                 
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n-closest_power_of_2]
        
        slopes = torch.tensor(get_slopes(nhead), device=device).unsqueeze(1).unsqueeze(1) # [nhead, 1, 1]
        
        # Create bias matrix
        context_position = torch.arange(seq_len, device=device).unsqueeze(1)
        memory_position = torch.arange(seq_len, device=device).unsqueeze(0)
        relative_position = torch.abs(context_position - memory_position) 
        # Note: ALiBi usually uses unidirectional negative distances for decoder, 
        # but for encoder (bidirectional) relative distance is typically symmetric absolute or just relative.
        # Original ALiBi is for causal decoder. adaptation for Encoder-Decoder:
        # Encoder: use symmetric distance. Decoder: use causal distance.
        # Here we just implement the bias term. For standard transformer usage in PyTorch "src_mask" is additive attention mask.
        # We need shape [batch*nhead, seq_len, seq_len] or [nhead, seq_len, seq_len]
        
        # Construct symmetric mask for encoder (non-causal) - simplistic adaptation
        # Bias = -1 * slope * |i - j|
        bias = -1 * slopes * relative_position.unsqueeze(0) # [nhead, seq_len, seq_len]
        return bias 

