from imports import *
from generator import WaveNetModel
from discriminator import Discriminator



def write_audio(lq, hq, gen, epoch, training='train'):
    lq = lq[0][0].detach().numpy()
    hq = hq[0][0].detach().numpy()
    gen = gen[0][0].detach().numpy()
    sf.write(f'GAN/Results/{training}/E{epoch}_LQ.wav', lq, 44100)
    sf.write(f'GAN/Results/{training}/E{epoch}_HQ.wav', hq, 44100)
    sf.write(f'GAN/Results/{training}/E{epoch}_GEN.wav', gen, 44100)

def convert_batch_to_log_mel_stft(batch, sample_rate, n_fft=2048, window_size=1024, n_mels=128):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_mels*16, win_length=window_size, hop_length=window_size // 4, n_mels=n_mels)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_transform(batch))
    return log_mel_spectrogram


def discriminator_loss(real_preds, fake_preds):
    real_loss = torch.mean(torch.relu(1.0 - real_preds))
    fake_loss = torch.mean(torch.relu(1.0 + fake_preds))
    loss = (real_loss + fake_loss) / 2
    return loss


def gen_and_disc_train_example(gen, disc, clean_batch, fx_batch, optimizer_gen, optimizer_disc, device, cnt, window, n_mels):
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)        
        fake_sound = gen(clean_batch)
        fx_batch = fx_batch[:, :, :fake_sound.size(2)]
        log_mel_real = convert_batch_to_log_mel_stft(fx_batch, 44100, window_size=window, n_mels=n_mels)
        log_mel_fake = convert_batch_to_log_mel_stft(fake_sound, 44100, window_size=window, n_mels=n_mels)
        real_preds = disc(log_mel_real)
        fake_preds = disc(log_mel_fake)
        loss_disc = discriminator_loss(real_preds, fake_preds)
        loss_fake = torch.mean(real_preds - fake_preds) / 2
        loss_gen = loss_fake
        if cnt % 2 == 1:
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()
        else:
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()
        loss_gen = loss_gen.item()
        loss_disc = loss_disc.item()
        real_preds_avg = torch.mean(real_preds).item()
        fake_preds_avg = torch.mean(fake_preds).item()
        return loss_gen, loss_disc, real_preds_avg, fake_preds_avg


def gen_and_discs_train_epoch(gen, discs, train_data, optimizer_gen, optimizer_discs, device, epoch, windows, n_mels):
    gen.train()
    for disc in discs:
        disc.train()   
    train_losses_gen, train_losses_disc, real_avg, fake_avg = ([[] for _ in range(len(discs))] for _ in range(4))
    cnt = 0
    for clean_batch, fx_batch in tqdm.tqdm(train_data):
        if cnt == 100:
            break ## FIXME later
        for i, disc in enumerate(discs):
            loss_gen, loss_disc, real_preds, fake_preds = gen_and_disc_train_example(gen, disc, clean_batch, fx_batch, optimizer_gen, optimizer_discs[i], device, cnt, windows[i], n_mels[i])
            train_losses_gen[i].append(loss_gen)
            train_losses_disc[i].append(loss_disc)
            real_avg[i].append(real_preds)
            fake_avg[i].append(fake_preds)
            if cnt % 10 == 0:
                print(f"Epoch {epoch}: batch{cnt}: disc{i} - loss_gen = {format(loss_gen*100, '.2f')}%, loss_disc = {format(loss_disc*100, '.2f')}%, real_preds = {format(real_preds, '.2f')}, fake_preds = {format(fake_preds, '.2f')}")
        cnt += 1
    avg_train_loss_gen = [np.mean(losses) * 100 for losses in train_losses_gen]        
    avg_train_loss_disc = [np.mean(losses) * 100 for losses in train_losses_disc]
    real_avg_avg = [np.mean(avg) for avg in real_avg]
    fake_avg_avg = [np.mean(avg) for avg in fake_avg]
    
    return avg_train_loss_gen, avg_train_loss_disc, real_avg_avg, fake_avg_avg


def spectral_loss(fx_batch, fake_audio, window_size, n_mels):
    log_mel_real = convert_batch_to_log_mel_stft(fx_batch, 44100, window_size=window_size, n_mels=n_mels)
    log_mel_fake = convert_batch_to_log_mel_stft(fake_audio, 44100, window_size=window_size, n_mels=n_mels)
    err = (log_mel_real - log_mel_fake) ** 2
    rmss = (log_mel_real ** 2)
    return torch.sum(err) / torch.sum(rmss) / len(fx_batch)

def gen_spectral_validation(gen, valid_data, device, epoch, valid_best_loss, rand_idx, window, n_mels):
    gen.eval()
    valid_losses = []
    i = 0
    for clean_batch, fx_batch in tqdm.tqdm(valid_data):
        if i == 100:
            break
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = gen(clean_batch)
        # Len matching
        min_len = min(fx_batch.size(2), predictions.size(2))
        fx_batch = fx_batch[:, :, -min_len:]
        predictions = predictions[:, :, -min_len:]
        loss = spectral_loss(fx_batch, predictions, window, n_mels)
        valid_losses.append(loss.item())
        if i == rand_idx:
            sample_clean, sample_fx, sample_pred = clean_batch, fx_batch, predictions
        i += 1
    avg_valid_loss = np.mean(valid_losses)
    if avg_valid_loss < valid_best_loss:
        write_audio(sample_clean, sample_fx, sample_pred, epoch, 'validation')
    return avg_valid_loss * 100    

def gen_and_disc_train(train_data, valid_data, device, params_dict, load_models=False):
    train_avg_gen_losses = []
    train_avg_disc_losses = []
    valid_losses = []
    num_repeats, dilation_depth, num_channels, kernel_size, epochs_num, lr = params_dict['dilation_repeats'], params_dict['dilation_depth'], params_dict['num_channels'], params_dict['kernel_size'], params_dict['epochs_num'], params_dict['lr']

    # Initialize models
    gen = WaveNetModel(num_repeats, dilation_depth, num_channels, kernel_size)
    discs = [Discriminator((3,10), (1,2), 9), Discriminator(5, 1, 9), Discriminator((10,2), 1, 5)]
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=lr)
    optimizers_discs = [torch.optim.Adam(disc.parameters(), lr=lr) for disc in discs]

    # Load models if exist
    if load_models:
        gen.load_state_dict(torch.load(f'Code/trained_generators/gen_best_model_R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}.pt'))
        for i, disc in enumerate(discs):
            disc.load_state_dict(torch.load(f'Code/trained_discriminators/disc_best_model_{2**(i+9)}.pt'))

    # Training loop
    valid_best_loss = float('inf')
    for epoch in range(1,epochs_num+1): 
        gen_losses, disc_losses, real_preds_avgs, fake_preds_avgs = gen_and_discs_train_epoch(gen, discs, train_data, optimizer_gen, optimizers_discs, device, epoch, windows=[512,1024,2048], n_mels=[64,128,256])
        print(f'Epoch {epoch}/{epochs_num}:')
        print(f'Generator loss = {format(np.mean(gen_losses), ".2f")}%')
        for disc_idx, disc in enumerate(discs):
            print(f'Discriminator loss for disc[{disc_idx}] = {format(disc_losses[disc_idx], ".2f")}%')
            print(f'Real and fake preds for disc[{disc_idx}]: {format(real_preds_avgs[disc_idx], ".2f")}, {format(fake_preds_avgs[disc_idx], ".2f")}')
        gen_loss = np.mean(gen_losses)
        disc_loss = np.mean(disc_losses)
        train_avg_gen_losses.append(gen_loss)
        train_avg_disc_losses.append(disc_loss)
        avg_valid_loss = gen_spectral_validation(gen, valid_data, device, epoch, valid_best_loss, 0, 1024, 128)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch {epoch}/{epochs_num}: validation loss = {format(avg_valid_loss, '.2f')}%")
        if avg_valid_loss < valid_best_loss:
            print(f"Loss improvment: old loss {format(valid_best_loss, '.2f')}%, new loss {format(avg_valid_loss, '.2f')}%")
            valid_best_loss = avg_valid_loss
            torch.save(gen.state_dict(), f'Models/GAN/Best/G_R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}.pt')
            for i, disc in enumerate(discs):
                torch.save(disc.state_dict(), f'Models/GAN/Best/D_W{2**(i+9)}.pt')
        torch.save(gen.state_dict(), f'Models/GAN/G_R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}_E{epoch}.pt')
        for i, disc in enumerate(discs):
            torch.save(disc.state_dict(), f'Models/GAN/D_W{2**(i+9)}_E{epoch}.pt')
    return train_avg_gen_losses, train_avg_disc_losses, valid_losses, valid_best_loss