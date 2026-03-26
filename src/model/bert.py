import math
import torch
from torch import nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, tokens):
        # BERT는 Transformer 원 논문과 달리 입력 임베딩에 sqrt(d_model) 스케일링을 곱하지 않는 편입니다.
        return self.embedding(tokens.long())


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        # BERT-style truncated normal: mean=0, std=0.02, truncation at 2 std.
        nn.init.trunc_normal_(self.pos_embedding, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pos_embedding[:, :seq_len, :]


class SegmentEmbedding(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.embedding = nn.Embedding(2, embed_size)

    def forward(self, x):
        return self.embedding(x)


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, emb_size=d_model)
        self.position = PositionalEmbedding(d_model=d_model)
        self.segment = SegmentEmbedding(embed_size=d_model)
        # BERT: token+pos+segment 합친 뒤 LayerNorm, 그 다음 Dropout
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, segment_info):
        x = self.token(x) + self.position(x) + self.segment(segment_info)
        x = self.layer_norm(x)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.float()
            # BERT 공식 구현에서 사용하는 마스킹 값(큰 음수).
            scores = scores.masked_fill(mask == 0, -10000.0)

        p_attn = F.softmax(scores, dim=-1)

        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # BERT 원 논문 스타일(대부분 post-LN):
        # residual add 후 LayerNorm
        return self.norm(x + self.dropout(sublayer(x)))


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.inputEmbedding = InputEmbedding(vocab_size, hidden, dropout)

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        # Initialize weights to match common BERT implementations.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.gamma)
            nn.init.zeros_(module.beta)

    def forward(self, x, segment_info):
        mask = (x != 0).unsqueeze(1).unsqueeze(2)

        x = self.inputEmbedding(x, segment_info)

        for transformer in self.encoder_blocks:
            x = transformer(x, mask)

        return x


class BERTLM(nn.Module):
    def __init__(self, bert: BERT):
        super().__init__()
        self.bert = bert
        bert_dropout = getattr(self.bert.inputEmbedding.dropout, "p", 0.1)
        self.next_sentence = NextSentencePrediction(self.bert.hidden, dropout=bert_dropout)
        # embedding weight를 직접 넘김
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, self.bert.inputEmbedding.token.embedding.weight)

        # Heads만 BERT-style 초기화를 적용 (decoder weight는 bert embedding과 weight-tying이라 제외).
        def _init_heads_weights(module):
            if module is self.mask_lm.decoder:
                return
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.gamma)
                nn.init.zeros_(module.beta)

        self.next_sentence.apply(_init_heads_weights)
        self.mask_lm.apply(_init_heads_weights)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class BERTLM_NoNSP(nn.Module):
    """NSP(Next Sentence Prediction) 없이 MLM(Masked Language Model)만 사용하는 BERT"""
    def __init__(self, bert: BERT):
        super().__init__()
        self.bert = bert
        # embedding weight를 직접 넘김
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, self.bert.inputEmbedding.token.embedding.weight)

        # MLM Head만 BERT-style 초기화를 적용 (decoder weight는 bert embedding과 weight-tying이라 제외).
        def _init_heads_weights(module):
            if module is self.mask_lm.decoder:
                return
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.gamma)
                nn.init.zeros_(module.beta)

        self.mask_lm.apply(_init_heads_weights)

    def forward(self, x, segment_label=None):
        # segment_label은 호환성을 위해 받지만 사용하지 않음 (단일 문장 처리)
        if segment_label is None:
            segment_label = torch.zeros_like(x)
        x = self.bert(x, segment_label)
        return self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        # BERT pooler: [CLS] -> Dense -> Tanh
        self.dense = nn.Linear(hidden, hidden)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        # NSP classifier
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, x):
        pooled = self.activation(self.dense(x[:, 0]))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)  # raw logits


class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden, embedding_weights):
        super().__init__()
        vocab_size = embedding_weights.size(0)

        # BERT MLM head: transform(dense+gelu+LN) -> decoder(weight tying)
        self.transform_dense = nn.Linear(hidden, hidden)
        self.transform_act = GELU()
        self.transform_layer_norm = LayerNorm(hidden)

        self.decoder = nn.Linear(hidden, vocab_size, bias=False)
        self.decoder.weight = embedding_weights  # weight tying
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        x = self.transform_dense(x)
        x = self.transform_act(x)
        x = self.transform_layer_norm(x)
        logits = self.decoder(x) + self.bias
        return logits
