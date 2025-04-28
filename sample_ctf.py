from pathlib import Path
import torch 
from torchvision import utils 
from PIL import Image
import torchvision.transforms as transforms
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.data.datasets import CXPDataset_Loader, CXPDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os

LINUX=False
DATA_ROOT_DIR = r"C:\Users\mhr_k\Data\CheXpert-Simp"
checkpoint_path = r"C:\Users\mhr_k\OneDrive\Documents\AI-Research\code\medfusion\linux_runs\2024_08_29_082814_ldm\epoch=12-step=39000.ckpt"
batch_size = 4

def rgb2gray(img):
    # img [B, C, H, W]
    return  ((0.3 * img[:,0]) + (0.59 * img[:,1]) + (0.11 * img[:,2]))[:, None]
    # return  ((0.33 * img[:,0]) + (0.33 * img[:,1]) + (0.33 * img[:,2]))[:, None]

def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b-b.min())/(b.max()-b.min()) for b in img])

def get_ctf_labels(label:int):
    if label == 0:
        return [1, 2]
    elif label == 1:
        return [0, 2]
    elif label == 2:
        return [0, 1]
    else:
        raise ValueError("illegal label input:", label)

if __name__ == "__main__":
    dataset = CXPDataset_Loader(root_dir = DATA_ROOT_DIR, windows=not LINUX, filter_race=True, race_range = ['white', 'black', 'asian'])
    ds_test = CXPDataset(data_list = dataset.test_ds[:32],  image_resize=256, filter_race=True)
    test_loader = DataLoader(ds_test, batch_size = batch_size, shuffle = False)
    path_out = Path.cwd()/'results/ctf_samples/sample_1'
    path_out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    # ------------ Load Model ------------
    # pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
    pipeline = DiffusionPipeline.load_from_checkpoint(checkpoint_path)
    pipeline.to(device)

    # --------- Generate Samples  -------------------
    steps = 10
    use_ddim = True 
    images = {}

    sample_each = 2
    for batch_idx, batch in enumerate(test_loader):

        imgs = batch["source"].to(device)
        labels = batch["race"].to(device) # could be 0, 1, 2
        patientIDs = batch["patientID"]
        subdirs = list()
        for i,pid in enumerate(patientIDs):
            if not os.path.exists(path_out / str(pid)):
                os.makedirs(path_out / str(pid))
            #utils.save_image(imgs[i], path_out / str(pid) / "original.png")
            img_normalized = (imgs[i] + 1) / 2.   #(1,256,256)
            transforms.ToPILImage()(img_normalized.squeeze(0)).save(path_out / str(pid) / "original.png")
            subdirs.append(path_out / str(pid))

        # --------- Conditioning ---------
        newlabels =  np.array([[0, 1, 2] for i in range(batch_size)]) #revert
        for label_i in range(3): #there are two possible counterfactual labels
            new_labels = torch.from_numpy(newlabels[:, label_i]).to(device)
            assert new_labels.shape == (batch_size, )
            condition = {'race': new_labels} #, 'sex':batch["sex"].to(device)}
            # ----------- Run --------
            res_list = list()
            for i in range(sample_each):
                results = pipeline.sample_ctf(imgs, condition=condition, guidance_scale=1, steps=steps, use_ddim=use_ddim) #[B, ]
                for j, subdir in enumerate(subdirs):
                    nl = new_labels[j].item()
                    save_dir = subdir / str(nl)
                    if os.path.exists(save_dir) == False:
                        os.makedirs(save_dir)
                    #utils.save_image(results[j], save_dir / (str(i) + ".png"))
                    img_normalized = (results[j] + 1) / 2.   #(1,256,256)
                    transforms.ToPILImage()(img_normalized.squeeze(0)).save(save_dir / (str(i) + ".png"))
            # results = pipeline.sample(n_samples, (4, 64, 64), guidance_scale=1, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim )

            # --------- Save result ---------------
            #results = (results+1)/2  # Transform from [-1, 1] to [0, 1]
            #results = results.clamp(0, 1)
    
    csv = pd.DataFrame(ds_test.data_list[:64])
    csv = csv.loc[csv.groupby('PatientID').apply(lambda x: x.index.max())]
    csv.to_csv(path_out / "label.csv")

    
        