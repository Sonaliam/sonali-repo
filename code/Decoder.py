class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=38):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout)for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(hid_dim)
        self.l1 = nn.Linear(hid_dim * 3, 1)
        self.l2 = nn.Linear(hid_dim * 3, 1)
        self.embed = nn.Embedding(len(vocab), embed_size) 

    def forward(self, trg, enc_art, trg_mask, art_mask, imgs, enc_ent, enc_ient):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        trg_decoder_out = trg
        
        for layer in self.layers:
            trg_decoder_out, trg_a, trg_v, attention = layer(trg_decoder_out, enc_art, trg_mask, art_mask, imgs, enc_ent, enc_ient)

        output = self.fc_out(trg_decoder_out)

        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.MSAtta = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.MSAtti = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.MSAtte = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.MSAttie = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.v = nn.Linear(hid_dim, hid_dim)
        self.relu = nn.ReLU()
        self.w1 = nn.Linear(hid_dim*4, hid_dim)
        self.w2 = nn.Linear(hid_dim*4, hid_dim)
        self.w3 = nn.Linear(hid_dim*4, hid_dim)
        self.w4 = nn.Linear(hid_dim*4, hid_dim)
        self.tanh = nn.Tanh()

    def forward(self, trg, enc_art, trg_mask, art_mask, imgs, enc_ents, enc_ients):
        imgs = self.relu(self.v(imgs))
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg0, attention0 = self.MSAtta(trg, enc_art, enc_art)
        _trg1, attention1 = self.MSAtti(trg, imgs, imgs)
        _trg2, attention2 = self.MSAtte(trg, enc_ents, enc_ents)
        _trg3, attention3 = self.MSAttie(trg, enc_ients, enc_ients)
        trg1_ = _trg0
        trg2_ = _trg1
        concat = torch.cat((_trg0,_trg1,_trg2,_trg3), 2)
        _trg_ = self.tanh(self.w1(concat)) * _trg0 + self.tanh(self.w2(concat)) * _trg1 + self.tanh(self.w3(concat)) * _trg2 + self.tanh(self.w4(concat)) * _trg3
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg_)) 
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, trg1_, trg2_, attention0
