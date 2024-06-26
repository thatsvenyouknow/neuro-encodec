import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import pathlib
from sklearn.model_selection import train_test_split
from lightning.fabric import Fabric
from lightning.fabric.utilities import AttributeDict
from tqdm import tqdm
from encodec import EncodecModel

from encodec.msstftd import MultiScaleSTFTDiscriminator
from dataset import EncodecDataset, get_random_chunk, composed_transform, list_files_with_paths
from metrics import ReconstructionLossTime, GenLoss, DiscLoss
from plot import plot_compare_time, plot_compare_frequency, plot_compare_histogram, plot_compare_spectrogram, plot_learning_curve

def create_dataloader(data_files, batch_size, num_workers, pin_memory, drop_last):
    dataset = EncodecDataset(data_files=data_files)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        )

def log_losses(loss_dict, losses, n, n_disc, train = True):
    for key, loss in loss_dict.items():
        if key == "disc" and train:
            losses[key].append(loss / n_disc if n_disc > 0 else 0.0)
        else:
            losses[key].append(loss / n)
    return losses

def log_learning_curves(train_losses, val_losses):
    for key in train_losses:
        plot_learning_curve(train_losses[key], val_losses[key], f'{key}_loss')

def train(
        data_dir: str="./data/", 
        batch_size: int=8,
        bw: float=3.0, 
        fs: int=19531,
        num_workers: int=0,
        device: str="cuda",
        epochs: int=2,
        lr: float=5e-5,
        betas: tuple=(0.5, 0.999),
        save_path: str="./pretrains/pretrained.ckpt",
        pretrained: str=None,
        plotting: bool=True,
        **kwargs
        ):
    if pretrained:
        fabric.load(pretrained, state)
        model = state.model
        disc = state.disc
        optimizer = state.optimizer
        optimizer_disc = state.optimizer_disc
    else:
        model = EncodecModel.encodec_model_24khz(pretrained=True)

    model.set_target_bandwidth(bw)
    model.sample_rate = fs
    data_dir = pathlib.Path(data_dir)
    data_files = list_files_with_paths(data_dir)
    train_files, val_files = train_test_split(data_files, test_size=0.2)

    #Define dataloaders
    train_dataloader = create_dataloader(
        data_files=train_files, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True
        )
    
    val_dataloader = create_dataloader(
        data_files=val_files, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True
        )
    
    #Set up discriminator
    disc = MultiScaleSTFTDiscriminator(filters=16).to(device)

    #Set up optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    #Set up Fabric
    fabric = Fabric(accelerator=device, precision="16-mixed")
    fabric.launch()
    model, optimizer = fabric.setup(model, optimizer)
    disc, optimizer_disc = fabric.setup(disc, optimizer_disc)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    #Set up loss functions
    reconstruction_loss = ReconstructionLossTime()
    gen_loss = GenLoss()
    disc_loss = DiscLoss()

    #For plotting
    train_losses = {
        "model": [], "rec": [], "disc": [], "gen": [], "feat": [], "quant_residual": []
    }
    val_losses = {
        "model": [], "rec": [], "disc": [], "gen": [], "feat": [], "quant_residual": []
    }

    for epoch in range(epochs):
        # Training loop
        train_loss_dict = {key: 0.0 for key in train_losses}
        progress_bar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f"Epoch {epoch+1}", 
            ncols=200
            )
        model.train()
        disc.train()
        n_disc_updates = 0 #keeps track of the number of discriminator updates

        for i, inp in progress_bar:
            optimizer.zero_grad()
            optimizer_disc.zero_grad()

            #Feed data through model
            outputs, l_w = model(inp, res_loss=True)

            #Reconstruction loss
            l_t = reconstruction_loss(outputs, inp)

            #GAN-Generator losses
            logits_fake, fmaps_fake = disc(outputs)
            logits_real, fmaps_real = disc(inp)
            l_g, l_feat = gen_loss(logits_fake, fmaps_real, fmaps_fake)

            #Update model (generator)   
            model_loss = l_t + l_g + l_feat + l_w
            fabric.backward(model_loss, model=model) 
            optimizer.step()

            #Update discriminator with probability 2/3
            if torch.rand(1).item() < 2/3:
                logits_fake, fmaps_fake = disc(outputs.detach()) #Detach to avoid backpropagation through generator
                logits_real, fmaps_real = disc(inp)
                l_d = disc_loss(logits_real, logits_fake)
                fabric.backward(l_d, model=disc) 
                optimizer_disc.step()
                train_loss_dict["disc"] += l_d.item()
                n_disc_updates += 1
            
            #Aggregate losses
            train_loss_dict["model"] += model_loss.item()
            train_loss_dict["rec"] += l_t.item()
            train_loss_dict["gen"] += l_g.item()
            train_loss_dict["feat"] += l_feat.item()
            train_loss_dict["quant_residual"] += l_w.item()

            #Update progress bar
            progress_bar.set_postfix({
                "model_loss": train_loss_dict["model"] / (i + 1),
                "rec_loss": train_loss_dict["rec"] / (i + 1), 
                "disc_loss": train_loss_dict["disc"] / n_disc_updates if n_disc_updates > 0 else 0.0, 
                "gen_loss": train_loss_dict["gen"] / (i + 1), 
                "feat_loss": train_loss_dict["feat"] / (i + 1),
                "quant_residual_loss": train_loss_dict["quant_residual"] / (i + 1)
            }) 

        if plotting:
                n_disc_updates = 0.0 if not n_disc_updates else n_disc_updates
                log_losses(train_loss_dict, train_losses, i+1, n_disc_updates, train=True)
                
        # Validation loop
        model.eval()
        disc.eval()
        val_loss_dict = {key: 0.0 for key in val_losses}

        with torch.inference_mode():
            for inp in val_dataloader:
                outputs, l_w = model(inp, res_loss=True)

                #Reconstruction loss
                l_t = reconstruction_loss(outputs, inp)

                #GAN-based losses
                logits_fake, fmaps_fake = disc(outputs)
                logits_real, fmaps_real = disc(inp)
                l_g, l_feat = gen_loss(logits_fake, fmaps_real, fmaps_fake)
                l_d = disc_loss(logits_fake, logits_real)

                #Total model/generator loss
                model_loss = l_t + l_g + l_feat + l_w

                #Aggregate losses
                val_loss_dict["model"] += model_loss.item()
                val_loss_dict["rec"] += l_t.item()
                val_loss_dict["disc"] += l_d.item()
                val_loss_dict["gen"] += l_g.item()
                val_loss_dict["feat"] += l_feat.item()
                val_loss_dict["quant_residual"] += l_w.item()

        val_len = len(val_dataloader)
        
        if plotting:
            log_losses(val_loss_dict, val_losses, val_len, n_disc = None, train=False)

        print(f"""Epoch {epoch+1} | Validation Model Loss: {val_loss_dict["model"]/val_len:.4f},
              Reconstruction Loss: {val_loss_dict["rec"]/val_len:.4f}, 
              Discriminator Loss: {val_loss_dict["disc"]/val_len:.4f},
              Generator Loss: {val_loss_dict["gen"]/val_len:.4f},
              Feature Loss: {val_loss_dict["feat"]/val_len:.4f},
              Quantization Residual Loss: {val_loss_dict["quant_residual"]/val_len:.4f}""")

        plot_compare_time(inp[0,...], outputs[0,...], window=slice(0, 4000), fs=fs) if plotting else None

    if plotting:
        log_learning_curves(train_losses, val_losses)

    state = AttributeDict(model=model, disc=disc, optimizer=optimizer, optimizer_disc=optimizer_disc, config=None)    
    fabric.save(save_path, state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuro-Encodec Training")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=".",
        help="Path to config file")
    args = parser.parse_args()
    train() #train(args.config_path)