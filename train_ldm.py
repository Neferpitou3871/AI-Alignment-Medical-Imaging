import argparse
from email.mime import audio
from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
import numpy as np 

from data.datamodules import SimpleDataModule
from data.datasets import CXPDataset_Loader, CXPDataset, MIMICCXRDataset_Loader, MIMICCXRDataset
from models.embedders.latent_embedders import VAE
from models.pipelines import DiffusionPipeline
from models.noise_estimators import UNet
from external.stable_diffusion.unet_openai import UNetModel
from models.noise_schedulers import GaussianNoiseScheduler
from models.embedders import LabelEmbedder, TimeEmbbeding

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a latent embedder on the MIMIC-CXR dataset")
    parser.add_argument("--DATA_DIR", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--dataset", type=str, default="CXP", required=True, help="CXP/MIMIC")
    parser.add_argument("--embedder_checkpoint", type=str, required=True, help="Path to the pretrained embedder checkpoint")
    parser.add_argument("--cond_emb_dim", type=str, default = 512, help="dimension of the conditional embedding")
    parser.add_argument("--LINUX", action="store_true", default=True, help="Specify if running on Linux (default: Windows)")
    parser.add_argument("--DEVICE", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use (default: auto-detect)")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--save_ckpt_every", type=int, default=5000, help="Save checkpoint every n steps")
    parser.add_argument("--log_every", type=int, default=1000, help="Logging frequency in steps")
    args = parser.parse_args()

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
        ds_train = ds_train,
        ds_val = ds_val,
        batch_size=args.batch_size, 
        num_workers=1,
        pin_memory=True,
        weights = args.train_weights
    )  
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'ldm' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    # ------------ Initialize Model ------------
    cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'emb_dim': args.cond_emb_dim,
        'condition_keys':["race"],
        'num_classes':{'race':3}
    }

    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': args.cond_emb_dim # need to align with cond embedder stable diffusion uses 4*model_channels (model_channels is about 256)
    }
    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch':8, 
        'out_ch':8, 
        'spatial_dims':2,
        'hid_chs':  [  256, 256],
        'kernel_sizes':[3, 3],
        'strides':     [1, 2],
        'time_embedder':time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder':cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block':True,
        'use_attention':'none',
    }


    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    latent_embedder = VAE
   
    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint = args.embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        optimizer_kwargs={'lr':1e-4},
        lr_scheduler = torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs = {
            'step_size': 10,  # Decrease LR every 10 epochs
            'gamma': 0.1
        },
        sample_every_n_steps=args.sample_every_n_steps,
        condition_keys = ['race']
    )
    
    # pipeline_old = pipeline.load_from_checkpoint('runs/2022_11_27_085654_chest_diffusion/last.ckpt')
    # pipeline.noise_estimator.load_state_dict(pipeline_old.noise_estimator.state_dict(), strict=True)

    # -------------- Training Initialization ---------------

    early_stopping = EarlyStopping(
        monitor='val/loss',
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode='min'
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor='val/loss',
        every_n_train_steps=args.save_and_sample_every,
        save_last=True,
        save_top_k=2,
        mode='min',
    )
    class PrintNewLineCallback(Callback):
        def on_epoch_start(self, arg1=None, arg2=None, arg3=None):
            print("\n")
    print_newline_callback = PrintNewLineCallback()
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, print_newline_callback],
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=args.save_and_sample_every, 
        auto_lr_find=False,
        # limit_train_batches=1000,
        limit_val_batches=1e5,
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)