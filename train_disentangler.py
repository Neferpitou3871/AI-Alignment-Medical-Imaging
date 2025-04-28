
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.datamodules import SimpleDataModule
from data.datasets import CXPDataset_Loader, CXPDataset, MIMICCXRDataset_Loader, MIMICCXRDataset
from models.embedders.latent_embedders import VAE
from models.disentanglers.mine import MINE
from models.disentanglers.perturbation_network import Perturbator
from models.disentanglers.disentangler import Disentangler
from pytorch_lightning import Trainer
from torchvision import transforms as T

import argparse
from pathlib import Path
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Assume `MINE`, `PerturbationNetwork`, and `Embedder` classes are defined elsewhere
# `Disentangler` class is also defined as provided

def main(args):
    # ----------------- Data Preparation -----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / args.dataset / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if args.DEVICE == 'cuda' else None

    # --------------- Dataset Loading --------------------
    if args.dataset == 'CXP':
        dataset = CXPDataset_Loader(root_dir=args.DATA_DIR, windows=not args.LINUX, filter_view='Frontal')
        ds_train = CXPDataset(data_list=dataset.train_ds, image_resize=64)
        ds_val = CXPDataset(data_list=dataset.val_ds, image_resize=64)
        ds_test = CXPDataset(data_list=dataset.test_ds, image_resize=64)
    elif args.dataset == 'MIMIC':
        dataset = MIMICCXRDataset_Loader(root_dir=args.DATA_DIR, windows=not args.LINUX, filter_view='Frontal')
        ds_train = MIMICCXRDataset(data_list=dataset.train_ds, image_resize=64)
        ds_val = MIMICCXRDataset_Loader(data_list=dataset.val_ds, image_resize=64)
        ds_test = MIMICCXRDataset(data_list=dataset.test_ds, image_resize=64)
    else:
        raise ValueError("dataset argument must be MIMIC or CXP")

    dm = SimpleDataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=args.batch_size, 
        num_workers=4,
        pin_memory=True
    ) 

    # ----------------- Model Initialization -----------------
    embedder = VAE.load_from_checkpoint(args.embedder_checkpoint)
    
    mine_net = MINE()
    perturb_net = Perturbator()

    # Initialize Disentangler Lightning Module
    model = Disentangler(
        mine_net=mine_net,
        perturb_net=perturb_net,
        embedder=embedder,
        lambda_hyper=args.lambda_hyper,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # ----------------- Training Setup -----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'd_runs' / args.dataset / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor='val/perturb_loss',
        save_top_k=3,
        mode='min',
    )
    
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=args.log_every
    )

    # ----------------- Start Training -----------------
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Disentangler model with MINE and Perturbation networks")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--embedder_checkpoint", type=str, required=True, help="Path to the pretrained embedder checkpoint")
    parser.add_argument("--lambda_hyper", type=float, default=1e-3, help="Lambda hyperparameter for the min-max objective")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--log_every", type=int, default=100, help="Log every n steps")

    args = parser.parse_args()
    main(args)
