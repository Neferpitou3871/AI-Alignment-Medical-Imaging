from pathlib import Path 
import logging
from datetime import datetime
from tqdm import tqdm

import numpy as np 
import torch
import torchvision.transforms.functional as tF
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, Subset

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim

from medical_diffusion.data.datasets import CXPDataset_Loader, CXPDataset
from medical_diffusion.models.embedders.latent_embedders import VAE

# ----------------Settings --------------
batch_size = 2
max_samples = 100 # set to None for all 
DATA_ROOT_DIR = r"C:\Users\mhr_k\Data\CheXpert-Simp"
LINUX = False

path_out = Path.cwd()/'results'
path_out.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

#data
dataset = CXPDataset_Loader(root_dir = DATA_ROOT_DIR, windows=not LINUX)
ds_test = CXPDataset(data_list = dataset.test_ds[:4],  image_resize=32)
dm_real = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

logger.info(f"Samples Real: {len(ds_test)}")


# --------------- Load Model ------------------
model = VAE.load_from_checkpoint('linux_runs/2024_03_14_215013/last.ckpt')
model.to(device)

# from diffusers import StableDiffusionPipeline
# with open('auth_token.txt', 'r') as file:
#     auth_token = file.read()
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32,  use_auth_token=auth_token)
# model = pipe.vae
# model.to(device)


# ------------- Init Metrics ----------------------
calc_lpips = LPIPS().to(device)


# --------------- Start Calculation -----------------
mmssim_list, mse_list = [], []
for real_batch in tqdm(dm_real):
    imgs_real_batch = real_batch[0].to(device)

    imgs_real_batch = tF.normalize(imgs_real_batch/255, 0.5, 0.5) # [0, 255] -> [-1, 1]
    with torch.no_grad():
        imgs_fake_batch = model(imgs_real_batch)[0].clamp(-1, 1) 

    # -------------- LPIP -------------------
    calc_lpips.update(imgs_real_batch, imgs_fake_batch) # expect input to be [-1, 1]

    # -------------- MS-SSIM + MSE -------------------
    for img_real, img_fake in zip(imgs_real_batch, imgs_fake_batch):
        img_real, img_fake = (img_real+1)/2, (img_fake+1)/2  # [-1, 1] -> [0, 1]
        mmssim_list.append(mmssim(img_real[None], img_fake[None], normalize='relu')) 
        mse_list.append(torch.mean(torch.square(img_real-img_fake)))


# -------------- Summary -------------------
mmssim_list = torch.stack(mmssim_list)
mse_list = torch.stack(mse_list)

lpips = 1-calc_lpips.compute()
logger.info(f"LPIPS Score: {lpips}")
logger.info(f"MS-SSIM: {torch.mean(mmssim_list)} ± {torch.std(mmssim_list)}")
logger.info(f"MSE: {torch.mean(mse_list)} ± {torch.std(mse_list)}")