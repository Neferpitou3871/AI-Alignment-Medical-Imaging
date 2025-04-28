from pathlib import Path
from datetime import datetime

import torch 
from torch.utils.data import ConcatDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

import sys
from datetime import datetime 
#
from data.datamodules import SimpleDataModule
from data.datasets import CXPDataset_Loader, CXPDataset, MIMICCXRDataset_Loader, MIMICCXRDataset
from models.embedders.latent_embedders import VAE
from loss.perceivers import LPIPS

import argparse
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Define the main function
if __name__ == "__main__":
    # --------------- Argument Parsing --------------------
    parser = argparse.ArgumentParser(description="Train a latent embedder on the MIMIC-CXR dataset")
    parser.add_argument("--DATA_DIR", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--dataset", type=str, default="CXP", required=True, help="CXP/MIMIC")
    parser.add_argument("--LINUX", action="store_true", default=True, help="Specify if running on Linux (default: Windows)")
    parser.add_argument("--DEVICE", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use (default: auto-detect)")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--save_ckpt_every", type=int, default=5000, help="Save checkpoint every n steps")
    parser.add_argument("--log_every", type=int, default=1000, help="Logging frequency in steps")
    args = parser.parse_args()

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'embedders' / args.dataset / str(current_time)
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

    # --------------- Model Setup --------------------
    model = VAE(
        in_channels=1, 
        out_channels=1, 
        emb_channels=8,
        spatial_dims=2,
        hid_chs=[64, 128, 256, 512], 
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        deep_supervision=1,
        use_attention='none',
        perceiver=LPIPS, 
        perceiver_kwargs={"model_dir": "downloaded_net"},
        loss=torch.nn.L1Loss,
        embedding_loss_weight=1e-6
    )

    min_max = "min"

    # --------------- Callbacks --------------------
    early_stopping = EarlyStopping(
        monitor="val/loss",
        min_delta=0.0,
        patience=30,
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor='val/loss',
        every_n_train_steps=args.save_ckpt_every,
        save_last=True,
        save_top_k=3,
        mode=min_max,
    )

    class PrintNewLineCallback(Callback):
        def on_epoch_start(self, trainer, pl_module):
            print("\n")
            
    print_newline_callback = PrintNewLineCallback()

    # --------------- Trainer Setup --------------------
    trainer = Trainer(
        accelerator="auto", 
        devices=1,
        default_root_dir=str(path_run_dir),
        callbacks=[early_stopping, checkpointing, print_newline_callback],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=args.log_every, 
        limit_val_batches=50, 
        min_epochs=2,
        max_epochs=args.n_epochs,
        num_sanity_val_steps=2
    )

    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
