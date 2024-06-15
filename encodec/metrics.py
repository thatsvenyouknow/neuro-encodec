#Encodec Metrics
import torch
from torch import nn

class ReconstructionLossTime(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat):
        return nn.functional.l1_loss(x, x_hat)
    

class GenLoss(nn.Module):
    """
    This class is used to compute the loss of the generator.

    Adapted from https://github.com/ZhikangNiu/encodec-pytorch/blob/main/losses.py
    """
    def __init__(self, fs: int=19531):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.l1Loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, logits_fake, fmap_real, fmap_fake):
        """
        Args:
            fmap_real (list): fmap_real is the output of the discriminator when the input is the real audio. 
            logits_fake (_type_): logits_fake is the list of every sub discriminator output of the Multi discriminator 
            fmap_fake (_type_): fmap_fake is the output of the discriminator when the input is the fake audio.

        Returns:
            l_g: adversarial loss for the generator
            l_feat: relative feature matching (perceptual) loss for the generator
        """
        l_g = torch.tensor([0.0], device='cuda', requires_grad=True)
        l_feat = torch.tensor([0.0], device='cuda', requires_grad=True)

        for tt1 in range(len(fmap_real)): # len(fmap_real) = 3
            l_g = l_g + torch.mean(self.relu(1 - logits_fake[tt1])) / len(logits_fake)
            for tt2 in range(len(fmap_real[tt1])): # len(fmap_real[tt1]) = 5
                # l_feat = l_feat + l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2].detach()))
                l_feat = l_feat + self.l1Loss(fmap_real[tt1][tt2], fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))

        KL_scale = len(fmap_real)*len(fmap_real[0]) # len(fmap_real) == len(fmap_fake) == len(logits_real) == len(logits_fake) == disc.num_discriminators == K
        l_feat /= KL_scale
        K_scale = len(fmap_real) # len(fmap_real[0]) = len(fmap_fake[0]) == L
        l_g /= K_scale

        return l_g, l_feat


class DiscLoss(nn.Module):
    """
    This class is used to compute the loss of the discriminator.
    Note: disc.num_discriminators = len(logits_real) = len(logits_fake) = 3

    Adapted from https://github.com/ZhikangNiu/encodec-pytorch/blob/main/losses.py
    """
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, logits_real, logits_fake):
        """
        Args:
            logits_real (List[torch.Tensor]): logits_real = disc_model(input_wav)[0]
            logits_fake (List[torch.Tensor]): logits_fake = disc_model(model(input_wav)[0])[0]
        
        Returns:
            lossd: discriminator loss
        """
        lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
        for tt1 in range(len(logits_real)):
            lossd = lossd + torch.mean(self.relu(1-logits_real[tt1])) + torch.mean(self.relu(1+logits_fake[tt1]))
        lossd = lossd / len(logits_real)
        return lossd
    

# class ReconstructionLossFrequency(nn.Module):
#     """
#     Reconstruction loss in the frequency domain.

#     Note: The original implementation may be found here: https://github.com/google-research/google-research/blob/master/ged_tts/distance_function/spectral_ops.py
#     """
#     def __init__(self, e = range(5, 12), fs = 19531):
#         super().__init__()
#         self.mel_spectrograms = nn.ModuleList([
#             transforms.MelSpectrogram(
#                 sample_rate=fs,  # Assuming a sample rate of 16 kHz
#                 n_fft=2 ** i,
#                 hop_length=2 ** i // 4,
#                 n_mels=64,
#                 normalized=True
#             ) for i in e
#         ])

#     def forward(self, x, x_hat):
#         loss = 0
#         for mel_spec in self.mel_spectrograms:
#             S_x = mel_spec(x)
#             S_x_hat = mel_spec(x_hat)
#             loss += torch.mean(torch.abs(S_x - S_x_hat)) + torch.mean((S_x - S_x_hat) ** 2)
#         return loss / len(self.mel_spectrograms)
