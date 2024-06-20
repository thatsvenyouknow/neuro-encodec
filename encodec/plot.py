import torch
import matplotlib.pyplot as plt
import torchaudio

@torch.no_grad()
def plot_compare_time(target, pred, window, fs):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(target.cpu().squeeze()[window], marker='o', ms=2.5, label='target')
    ax.plot(pred.cpu().squeeze()[window], marker='o', ms=2.5, label='predicted', alpha=0.5)
    ax.set_xlabel(f'time @ {fs} Hz')
    ax.legend()
    fig.show()

@torch.no_grad()
def plot_compare_frequency(target, pred, fs):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.psd(target.cpu().squeeze(), Fs=fs, label='target')
    ax.psd(pred.cpu().squeeze(), Fs=fs, label='predicted')
    ax.legend()
    fig.show()

@torch.no_grad()
def plot_compare_histogram(target, pred):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(target.cpu().squeeze(), bins=40, alpha=0.5, label='target')
    ax.hist(pred.cpu().squeeze(), bins=40, alpha=0.5, label='predicted')
    ax.legend()
    fig.show()

@torch.no_grad()
def plot_compare_spectrogram(target, pred, fs):

    to_spec = torchaudio.transforms.Spectrogram(hop_length=fs//1000)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

    spec_target = torch.log(to_spec(target.cpu().squeeze()))
    spec_pred = torch.log(to_spec(pred.cpu().squeeze()))

    axs[0].set_title('Target')
    axs[0].imshow(spec_target.squeeze())
    axs[0].set_ylabel('frequency bins')
    axs[0].set_xlabel('time')

    axs[1].set_title('Predicted')
    axs[1].imshow(spec_pred.squeeze())
    axs[1].set_ylabel('frequency bins')
    axs[1].set_xlabel('time')

    fig.tight_layout()
    fig.show()

def plot_learning_curve(train_losses, val_losses, title='loss'):
    fig, ax = plt.subplots(figsize=(10, 4))


    ax.plot(train_losses, marker='o', label='Training')
    ax.set_xlabel('# epochs')
    ax.set_ylabel('loss')

    ax.plot(val_losses, marker='o', label='Validation')
    ax.set_xlabel('# epochs')
    ax.set_ylabel('loss')
    ax.legend()
    ax.set_title(title)
    fig.show()