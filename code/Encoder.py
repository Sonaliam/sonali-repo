class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=302):
        super().__init__()

        self.pos_embedding = nn.Embedding(hid_dim, hid_dim)
        self.lin = nn.Linear(1024, hid_dim)
        self.lin_i = nn.Linear(2048, hid_dim)
        self.lin_e = nn.Linear(300, hid_dim) 
        self.lin_ie = nn.Linear(300, hid_dim) 

        self.msatt_art = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)

        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm2 = nn.LayerNorm(hid_dim)
        self.msatt_im_art = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.msatt_ents = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.multimodel_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.g = nn.Linear(hid_dim, hid_dim)

    def forward(self, art, art_mask, imgs, ents, ients):

        imgs = self.lin_i(imgs)

        art = nn.Flatten()(art)
        art = self.lin(art)

        batch_size = art.shape[0]
        art_len = art.shape[1]
        pos = torch.arange(0, art_len).unsqueeze(0).repeat(batch_size, 1).to(device)   
        art = self.dropout((art * self.scale) + self.pos_embedding(pos))

        _art, _ = self.msatt_art(art, art, art, art_mask)
        art = self.self_attn_layer_norm(art + self.dropout(_art))
        ents = self.lin_e(ents)

        ients = self.lin_ie(ients)
        batch_size = ients.shape[0]
        ients_len = ients.shape[1]
        pos = torch.arange(0, ients_len).unsqueeze(0).repeat(batch_size, 1).to(device)  
        ients = self.dropout((ients * self.scale) + self.pos_embedding(pos))
        
        ents_mask = None
        _ents, _ = self.msatt_ents(ents, ents, ents, ents_mask)
        ents = self.self_attn_layer_norm(ents + self.dropout(_ents))

        art = self.CA_fea(art, art_mask, imgs)
        ents = self.CA_fea(ents, ents_mask, imgs)
        ients = self.CA_fea(art, art_mask, ients)

        return art, imgs, ents, ients


    def CA_fea(self,art, art_mask, imgs):
        imgs = torch.mean(imgs, dim=1).unsqueeze(dim=1)
        art_ca, _ = self.msatt_im_art(imgs, art, art, art_mask)

        g = self.tanh(self.g(imgs))
        art_ca = g * (art_ca)

        art_bar =  torch.empty(size=(1, 1, hid_dim)) 

        for i in range(art.shape[1]):
          art2 = art_ca + ((1-g) * art[:,i,:])
          art_bar = torch.hstack((art_bar, art2))

        art_bar = art_bar[:,1:]

        art = self.positionwise_feedforward(art_bar)
        art = self.ff_layer_norm1(art_bar + self.dropout(art))

        return art
