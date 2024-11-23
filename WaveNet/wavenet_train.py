from imports import *
from wavenet import WaveNetModel


def write_audio(lq, hq, gen, epoch, training='train'):
    lq = lq[0][0].detach().numpy()
    hq = hq[0][0].detach().numpy()
    gen = gen[0][0].detach().numpy()
    sf.write(f'Results/WaveNet/{training}/E{epoch}_LQ.wav', lq, 44100)
    sf.write(f'Results/WaveNet/{training}/E{epoch}_HQ.wav', hq, 44100)
    sf.write(f'Results/WaveNet/{training}/E{epoch}_GEN.wav', gen, 44100)

def ESR_loss(preds, labels):
    labels = labels[:, :, -preds.size(2):]
    error = preds - labels
    rmse = torch.sum(torch.square(error))
    rmss = torch.sum(torch.square(labels))
    esr = torch.abs(rmse / rmss)
    return esr



def gen_train_epoch(model, train_data, optimizer, device, epoch):
    model.train()
    train_losses = []
    stft = auraloss.freq.STFTLoss().to(device)
    smooth = torch.nn.SmoothL1Loss().to(device)
    for clean_batch, fx_batch in tqdm.tqdm(train_data):
        optimizer.zero_grad()
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = model(clean_batch)
        min_len = min(predictions.size(2), fx_batch.size(2))
        fx_batch, predictions = fx_batch[:, :, :min_len], predictions[:, :, :min_len]
        #loss = (stft(predictions, fx_batch) + ESR_loss(predictions, fx_batch))/3
        loss = ESR_loss(predictions, fx_batch) 
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)
    return avg_train_loss * 100



def gen_valid_epoch(model, valid_data, device):
    model.eval()
    valid_losses = [] 
    i = 0
    for clean_batch, fx_batch in tqdm.tqdm(valid_data):
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = model(clean_batch)
        loss = min(ESR_loss(predictions, fx_batch), ESR_loss(predictions, -1*fx_batch))
        valid_losses.append(loss.item())
        if i == 0:
            write_audio(clean_batch, fx_batch, predictions, 1, 'validation')
            i += 1
    avg_valid_loss = np.mean(valid_losses)
    return avg_valid_loss * 100



def gen_train(train_data, valid_data, device, params_dict):
    train_losses = []
    valid_losses = []
    num_repeats = params_dict['dilation_repeats']
    dilation_depth = params_dict['dilation_depth']
    num_channels = params_dict['num_channels']
    kernel_size = params_dict['kernel_size']
    epochs_num = params_dict['epochs_num']
    lr = params_dict['lr']
    gen = WaveNetModel(num_repeats, dilation_depth, num_channels, kernel_size)
    #gen.load_state_dict(torch.load(f'Models/WaveNet/R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}_E{epochs_num}_L95.92.pt'))
    optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    valid_best_loss = float('inf')
    for epoch in range(1,epochs_num+1):
        avg_train_loss = gen_train_epoch(gen, train_data, optimizer, device, epoch)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}/{epochs_num}: training loss = {format(avg_train_loss, '.2f')}%")
        avg_valid_loss = gen_valid_epoch(gen, valid_data, device)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch {epoch}/{epochs_num}: validation loss = {format(avg_valid_loss, '.2f')}%")
        torch.save(gen.state_dict(), f'Models/WaveNet/R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}_E{epoch}_L{avg_valid_loss}.pt')
        if avg_valid_loss < valid_best_loss:
            print(f"Loss improvment: old loss {format(valid_best_loss, '.2f')}%, new loss {format(avg_valid_loss, '.2f')}%")
            valid_best_loss = avg_valid_loss
    return train_losses, valid_losses, valid_best_loss


