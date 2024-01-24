maxlen = 15
SPECIALS2IDX = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
decoder_lr = 0.0001
fine_tune_encoder = False
hid_dim = 512
pf_dim = 256
crop_size = 224
model_path = "checkpoint/"

data_name = ''
checkpoint = None
epochs = 4
start_epoch = 0

def main():
    global epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, train_logger, dev_logger

    if checkpoint is None:
        enc = Encoder(len(vocab), hid_dim, 1, n_heads, pf_dim, 0.1)
        dec = Decoder(len(vocab), hid_dim, n_layers, n_heads, pf_dim, 0.1)
        model = Model(enc, dec, 0, 0)
        optimizer = optim.Adam(model.parameters(), lr=decoder_lr, betas=(0.9, 0.98), eps=1e-7)

        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        model.apply(initialize_weights)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['decoder']
        optimizer = optim.Adam(model.parameters(), lr=decoder_lr)

    model = model.to(device)

    train_log_dir = os.path.join(model_path, 'train')
    dev_log_dir = os.path.join(model_path, 'dev')
    train_logger = Logger(train_log_dir)
    dev_logger = Logger(dev_log_dir)

    criterion = nn.CrossEntropyLoss(ignore_index=SPECIALS2IDX['<pad>'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    for epoch in range(start_epoch, epochs):
        train(model = model,
              data_loader_train=data_loader_train,
              criterion=criterion,
              optimizer = optimizer,
              epoch=epoch,
              logger=train_logger,
              logging=True)
        if epoch > 0:
            recent_bleu4 = validate(model=model,
                                    val_loader=val_loader,
                                    criterion=criterion,
                                    vocab=vocab,
                                    epoch=epoch,
                                    logger=dev_logger,
                                    logging=True)

          is_best = bleu_score > best_bleu4 
          best_bleu4 = max(float(best_bleu4), float(bleu_score))
          if not is_best:
              epochs_since_improvement += 1
          else:
              epochs_since_improvement = 0

        if epoch <= 4:
            recent_bleu4 = 0
            is_best = True

        save_checkpoint(data_name, epoch, epochs_since_improvement, model, optimizer, bleu_score, is_best)


# Define a transform to pre-process the training images.
transform_train = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Obtain the data loader.
data_loader_train = get_loader(transform=None,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False,
                         cocoapi_loc = '',)
val_loader = get_loader(transform=None,
                         mode='test',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True,
                         cocoapi_loc = '',)

nb_tokens  = len(data_loader_train.dataset.vocab)


if __name__ == '__main__':
    main()