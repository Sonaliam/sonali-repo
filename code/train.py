def train(model, data_loader_train, optimizer, criterion, epoch, logger, logging=True):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    t = tqdm(data_loader_train, desc='Train %d' % epoch)

    for i, (imgs, caps, arts, caplens, ents, ients) in enumerate(t):
        data_time.update(time.time() - start)
        imgs = imgs.to(device)
        caps = caps.to(device)
        arts = arts.to(device)
        ents = ents.to(device)
        ients = ents.to(device)

        output, _, mask = model(arts, caps[:, :-1], imgs, ents, ients)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        caps = caps[:, 1:].contiguous().view(-1)
        loss = criterion(output, caps)

        optimizer.zero_grad()

        decode_lengths = [c - 1 for c in caplens]
        losses.update(loss.item(), sum(decode_lengths))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        batch_time.update(time.time() - start)

        start = time.time()

    # log into tf series
    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, epochs, losses.avg, np.exp(losses.avg)))
    if logging:
        logger.scalar_summary('loss', losses.avg, epoch)
        logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)
