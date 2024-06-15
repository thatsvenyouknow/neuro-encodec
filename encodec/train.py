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

def train(
        data_dir: str="./data/", 
        batch_size: int=8,
        bw: float=3.0, 
        fs: int=19531,
        num_workers: int=0,
        device: str="cuda",
        epochs: int=5,
        lr: float=5e-5,
        betas: tuple=(0.5, 0.999),
        save_path: str="./pretrains/pretrained.ckpt",
        pretrained: str=None,
        plotting: bool=False,
        **kwargs
        ):
    if pretrained:
        fabric.load(pretrained, state)
        model = state.model
        disc = state.disc
        optimizer = state.optimizer
        optimizer_disc = state.optimizer_disc
    else:
        model = EncodecModel.encodec_model_24khz(pretrained=False)

    model.set_target_bandwidth(bw)
    model.sample_rate = fs
    data_dir = pathlib.Path(data_dir)
    data_files = list_files_with_paths(data_dir)
    train_files, val_files = train_test_split(data_files, test_size=0.2)

    #Define datasets and dataloaders
    train_dataset = EncodecDataset(
        data_files=train_files, 
        )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        )
            
    val_dataset = EncodecDataset(
        data_files=val_files, 
        )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
    train_model_losses, train_rec_losses, train_disc_losses, train_gen_losses, train_feat_losses, train_quant_residual_losses = [], [], [], [], [], []
    val_model_losses, val_rec_losses, val_disc_losses, val_gen_losses, val_feat_losses, val_quant_residual_losses = [], [], [], [], [], []

    for epoch in range(epochs):
        # Training loop
        train_model_loss, train_rec_loss, train_disc_loss, train_gen_loss, train_feat_loss, quant_residual_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
                train_disc_loss += l_d.item()
                n_disc_updates += 1
            
            #Aggregate losses
            train_model_loss += model_loss.item()
            train_rec_loss += l_t.item()
            train_gen_loss += l_g.item()
            train_feat_loss += l_feat.item()
            quant_residual_loss += l_w.item()
            
            if plotting:
                train_model_losses.append(train_model_loss / (i + 1))
                train_rec_losses.append(train_rec_loss / (i + 1))
                train_disc_losses.append(train_disc_loss / n_disc_updates if n_disc_updates > 0 else 0.0)
                train_gen_losses.append(train_gen_loss / (i + 1))
                train_feat_losses.append(train_feat_loss / (i + 1))
                train_quant_residual_losses.append(quant_residual_loss / (i + 1))

            #Update progress bar
            progress_bar.set_postfix({
                "model_loss": train_model_loss / (i + 1),
                "rec_loss": train_rec_loss / (i + 1), 
                "disc_loss": train_disc_loss/n_disc_updates if n_disc_updates > 0 else 0.0, 
                "gen_loss": train_gen_loss / (i + 1), 
                "feat_loss": train_feat_loss / (i + 1),
                "quant_residual_loss": quant_residual_loss / (i + 1)
                }) 

        # Validation loop
        model.eval()
        disc.eval()
        val_model_loss, val_rec_loss, val_disc_loss, val_gen_loss, val_feat_loss, val_quant_residual_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
                val_model_loss += model_loss.item()
                val_rec_loss += l_t.item()
                val_disc_loss += l_d.item()
                val_gen_loss += l_g.item()
                val_feat_loss += l_feat.item()
                val_quant_residual_loss += l_w.item()

        val_len = len(val_dataloader)
        
        if plotting:
            val_model_losses.append(val_model_loss/val_len)
            val_rec_losses.append(val_rec_loss/val_len)
            val_disc_losses.append(val_disc_loss/val_len)
            val_gen_losses.append(val_gen_loss/val_len)
            val_feat_losses.append(val_feat_loss/val_len)
            val_quant_residual_losses.append(val_quant_residual_loss/val_len)

        print(f"""Epoch {epoch+1} | Validation Model Loss: {val_model_loss/val_len:.4f},
              Reconstruction Loss: {val_rec_loss/val_len:.4f}, 
              Discriminator Loss: {val_disc_loss/val_len:.4f},
              Generator Loss: {val_gen_loss/val_len:.4f},
              Feature Loss: {val_feat_loss/val_len:.4f},
              Quantization Residual Loss: {l_w.item():.4f}""")

        plot_compare_time(inp, outputs, window=slice(0, 1000), fs=fs) if plotting else None
        
    if plotting:
        plot_learning_curve(train_model_losses, val_model_losses)
        plot_learning_curve(train_rec_losses, val_rec_losses)
        plot_learning_curve(train_disc_losses, val_disc_losses)
        plot_learning_curve(train_gen_losses, val_gen_losses)
        plot_learning_curve(train_feat_losses, val_feat_losses)
        plot_learning_curve(train_quant_residual_losses, val_quant_residual_losses)


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