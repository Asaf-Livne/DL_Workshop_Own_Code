from imports import *
from models import Generator, MSD, MPD, gen_feature_loss, gen_adv_loss, disc_adv_loss, gen_spec_loss
from data import HFDataLoader, HFDataSet


import torch

def train_example(gen, gen_opt, msd, msd_opt, mpd, mpd_opt, lq_batch, hq_batch, device, cfg):
    hq_batch, lq_batch = hq_batch.to(device), lq_batch.to(device)
    gen.train()
    msd.train()
    mpd.train()
    gen_opt.zero_grad()
    msd_opt.zero_grad()
    mpd_opt.zero_grad()
    
    # Generate batch using the generator
    gen_batch = gen(lq_batch)

    # MSD Loss
    msd_real_preds, msd_gen_preds, _, _ = msd(hq_batch, gen_batch.detach())
    msd_adv_loss, _, _ = disc_adv_loss(msd_real_preds, msd_gen_preds)

    # MPD Loss
    mpd_real_preds, mpd_gen_preds, _, _ = mpd(hq_batch, gen_batch.detach())
    mpd_adv_loss, _, _ = disc_adv_loss(mpd_real_preds, mpd_gen_preds)

    # Backpropagation for Discriminators
    msd_adv_loss.backward()
    msd_opt.step()

    mpd_adv_loss.backward()
    mpd_opt.step()

    # Second discrimination for training the generator
    msd_real_preds, msd_gen_preds, msd_real_features, msd_gen_features = msd(hq_batch, gen_batch)
    mpd_real_preds, mpd_gen_preds, mpd_real_features, mpd_gen_features = mpd(hq_batch, gen_batch)

    # Generator Loss
    adv_loss = gen_adv_loss(msd_gen_preds) + gen_adv_loss(mpd_gen_preds)
    feat_loss = gen_feature_loss(msd_real_features, msd_gen_features) + gen_feature_loss(mpd_real_features, mpd_gen_features)
    spec_loss = gen_spec_loss(hq_batch, gen_batch, cfg)

    # Total Generator Loss
    lambda_feat = 2
    lambda_spec = 0.1
    gen_loss = adv_loss + lambda_feat * feat_loss + lambda_spec * spec_loss

    # Backpropagation for Generator
    gen_loss.backward()
    gen_opt.step()

    return {
        'gen_loss': gen_loss.item(),
        'adv_loss': adv_loss.item(),
        'feat_loss': feat_loss.item(),
        'spec_loss': spec_loss.item(),
        'msd_loss': msd_adv_loss.item(),
        'mpd_loss': mpd_adv_loss.item()
    }

    
def valid_example(gen, lq_batch, hq_batch, device, cfg):
    hq_batch, lq_batch = hq_batch.to(device), lq_batch.to(device)
    gen.eval()

    with torch.no_grad():
        # Generate batch using the generator
        gen_batch = gen(lq_batch)

        # Spectral Loss
        spec_loss = gen_spec_loss(hq_batch, gen_batch, cfg)

    return spec_loss.item()

def train_epoch(gen, gen_opt, msd, msd_opt, mpd, mpd_opt, train_data, valid_data, device, cfg):
    gen_losses = []
    gen_spec_losses = []
    gen_adv_losses = []
    gen_feat_losses = []
    msd_losses = []
    mpd_losses = []
    for lq_batch, hq_batch in tqdm.tqdm(train_data):
        loss = train_example(gen, gen_opt, msd, msd_opt, mpd, mpd_opt, lq_batch, hq_batch, device, cfg)
        gen_spec_losses.append(loss['spec_loss'])
        gen_adv_losses.append(loss['adv_loss'])
        gen_feat_losses.append(loss['feat_loss'])
        gen_losses.append(loss['gen_loss'])
        msd_losses.append(loss['msd_loss'])
        mpd_losses.append(loss['mpd_loss'])
    avg_spec_loss = np.mean(gen_spec_losses)
    avg_adv_loss = np.mean(gen_adv_losses)
    avg_feat_loss = np.mean(gen_feat_losses)
    avg_gen_loss = np.mean(gen_losses)
    avg_msd_loss = np.mean(msd_losses)
    avg_mpd_loss = np.mean(mpd_losses)
    valid_losses = []
    for lq_batch, hq_batch in tqdm.tqdm(valid_data):
        loss = valid_example(gen, lq_batch, hq_batch, device, cfg)
        valid_losses.append(loss)
    avg_valid_loss = np.mean(valid_losses)
    return {
        'avg_gen_loss': avg_gen_loss,
        'avg_adv_loss': avg_adv_loss,
        'avg_feat_loss': avg_feat_loss,
        'avg_spec_loss': avg_spec_loss,
        'avg_msd_loss': avg_msd_loss,
        'avg_mpd_loss': avg_mpd_loss, 
        'avg_valid_loss': avg_valid_loss
    }


def print_losses(epoch, losses):
    print(f'Epoch {epoch} Losses:')
    for loss, value in losses.items():
        print(f'{loss}: {value:.4f}')
    print('\n')
    print('------------------------------------')
    print('\n')

def train_process(gen, gen_opt, msd, msd_opt, mpd, mpd_opt, train_data, valid_data, device, epochs, cfg, load=None):

    valid_best_loss = float('inf')
    all_losses = []

    for epoch in range(epochs):
        losses = train_epoch(gen, gen_opt, msd, msd_opt, mpd, mpd_opt, train_data, valid_data, device, cfg)
        print_losses(epoch, losses)

        # Save model
        torch.save(gen.state_dict(), f'Models/HiFi-GAN/Last/gen.pt')
        torch.save(msd.state_dict(), f'Models/HiFi-GAN/Last/msd.pt')
        torch.save(mpd.state_dict(), f'Models/HiFi-GAN/Last/mpd.pt')
        torch.save(gen.state_dict(), f'Models/HiFi-GAN/All/gen_E{epoch}.pt')
        torch.save(msd.state_dict(), f'Models/HiFi-GAN/All/msd_E{epoch}.pt')
        torch.save(mpd.state_dict(), f'Models/HiFi-GAN/All/mpd_E{epoch}.pt')
        if losses['avg_valid_loss'] < valid_best_loss:
            torch.save(gen.state_dict(), f'Models/HiFi-GAN/Best/gen.pt')
            torch.save(msd.state_dict(), f'Models/HiFi-GAN/Best/msd.pt')
            torch.save(mpd.state_dict(), f'Models/HiFi-GAN/Best/mpd.pt')
            valid_best_loss = losses['avg_valid_loss']
            print(f"Loss improvement: old loss {valid_best_loss:.4f}, new loss {losses['avg_valid_loss']:.4f}")
            print('\n')
        all_losses.append(losses)
    return all_losses
        


def train(cfg):
    gen = Generator(cfg)
    msd = MSD(cfg['msd_poolings'])
    mpd = MPD(cfg['mpd_periods'])
    if cfg['load'] != None:
        load = cfg['load']
        gen.load_state_dict(torch.load(f'Models/HiFi-GAN/{load}/gen.pt'))
        msd.load_state_dict(torch.load(f'Models/HiFi-GAN/{load}/msd.pt'))
        mpd.load_state_dict(torch.load(f'Models/HiFi-GAN/{load}/mpd.pt'))

    gen_opt = torch.optim.Adam(gen.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
    msd_opt = torch.optim.Adam(msd.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
    mpd_opt = torch.optim.Adam(mpd.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using {device} device")
    gen.to(device)
    msd.to(device)
    mpd.to(device)
    train_data = HFDataLoader(HFDataSet(cfg['train_lq_path'], cfg['train_hq_path']), cfg['batch_size'], True, 4)
    valid_data = HFDataLoader(HFDataSet(cfg['valid_lq_path'], cfg['valid_hq_path']), cfg['batch_size'], False, 4)
    all_losses = train_process(gen, gen_opt, msd, msd_opt, mpd, mpd_opt, train_data, valid_data, device, cfg['epochs'],cfg , cfg['load'])
    pickle.dump(all_losses, open(f'Results/HiFi-GAN/Stats/Training/all_losses.pkl', 'wb'))
    return all_losses

