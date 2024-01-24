class NIC(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 art_pad_idx,
                 trg_pad_idx):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.art_pad_idx = 0
        self.trg_pad_idx = 0
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def make_art_mask(self, art):
        art_mask = (art != self.art_pad_idx).unsqueeze(1).unsqueeze(2)
        return art_mask

    def make_caps_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, arts, caps, imgs, ents, ients):
        art_mask = self.make_art_mask(arts)
        caps_mask = self.make_caps_mask(caps)
        enc_art, imgs, enc_ent, enc_ient  = self.encoder(arts, art_mask, imgs, ents, ients)
        output, attention = self.decoder(caps, enc_art, caps_mask, art_mask, imgs, enc_ent, enc_ient)
        return output, attention, art_mask
