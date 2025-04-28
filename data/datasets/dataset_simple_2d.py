
import torch.utils.data as data 
import torch 
from torch import nn
from pathlib import Path 
from torchvision import transforms as T
import pandas as pd
import numpy as np
import sys
import os

from PIL import Image

from medical_diffusion.data.augmentation.augmentations_2d import Normalize, ToTensor16bit

class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = 'tif', # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform = None,
        image_resize = None,
        augment_horizontal_flip = False,
        augment_vertical_flip = False, 
        image_crop = None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

        if transform is None: 
            self.transform = T.Compose([
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5)
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        # img = Image.open(path_item) 
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return Image.open(path_item).convert('RGB') 
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images 
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None 

import math
class CXPDataset(data.Dataset):

    def __init__(
        self,
        data_list,
        transform=None,
        image_resize = 256,
        augment_horizontal_flip = False,
        augment_vertical_flip = False,
        image_crop = None,
        filter_race=False
    ):
        self.data_list = data_list
        if transform is None: 
            self.transform = T.Compose([
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                T.ToTensor(),
                # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                # ToTensor16bit(),
                # Normalize(), # [0, 1.0]
                # T.ConvertImageDtype(torch.float),
                T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

        self.race_id = {
            'white':0,
            'black':1,
            'asian':2
        }
        self.filter_race = filter_race
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        item = self.data_list[index]
        img = Image.open(item["image"]).convert("L")
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        img_tensor = self.transform(img)
        assert len(img_tensor.shape)==3
        if self.filter_race:
            return {'source': img_tensor, 
                    'target':1, 
                    'race':self.race_id[item["Race"]], 
                    'sex':0 if item["Sex"] == 'Female' else 1,
                    'image_path':item['image'],
                    'patientID':item["PatientID"],
                    'Cardiomegaly': item['Cardiomegaly'], 
                    'Edema': item["Edema"],
                    "Consolidation": item['Consolidation'],
                    'Atelectasis': item['Atelectasis'],
                    "Pleural Effusion": item["Pleural Effusion"]
                    }
        else:
            return {'source':img_tensor,
                    'target':1}

class CXPDataset_Loader():

    def __init__(
        self,
        root_dir,
        seed=0,
        val_frac=0.2,
        test_frac=0.2,
        windows = False,
        filter_view = None,
        filter_race = False,
        race_range = ['white', 'black']
    ):
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.root_dir = root_dir
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.windows = windows
        self.seed = seed

        df_path = root_dir + "\\full_df.csv" if windows else root_dir + "/full_df.csv"
        if os.path.exists(df_path):
            self.df = pd.read_csv(df_path)
        else:
            if windows:
                csv_train = self.preprocess_csv(root_dir + "\\train_CheXbert.csv")
                csv_valid = self.preprocess_csv(root_dir + "\\valid.csv")
                self.excel_demo = self.preprocess_demo(root_dir+"\\CHEXPERT_DEMO.csv")
            else:
                csv_train = self.preprocess_csv(root_dir + "/train_CheXbert.csv")
                csv_valid = self.preprocess_csv(root_dir + "/valid.csv")
                self.excel_demo = self.preprocess_demo(root_dir+"/CHEXPERT_DEMO.csv")
            self.csv = pd.concat([csv_train, csv_valid], axis=0)
            #For now, the pathology information is not relevant, we could discard
            self.df = pd.merge(self.csv, self.excel_demo[["PatientID", "PRIMARY_RACE"]], on="PatientID", how="inner")
            self.df.to_csv(df_path)
        if filter_view is not None:
            assert filter_view in ["Frontal", "Lateral"]
            self.df = self.df[self.df["Frontal/Lateral"] == filter_view]
        self.datalist = self.create_data_list(self.df)
        if not filter_race:
            self.train_ds, self.val_ds, self.test_ds = self._generate_data_list()
        else:
            self.train_ds, self.val_ds, self.test_ds = self._generate_data_list_diffusion(race_range=race_range)

    def preprocess_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df[["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", 
                 'Cardiomegaly', 'Edema', "Consolidation", 'Atelectasis', "Pleural Effusion"]]
        df["PatientID"] = df['Path'].str.extract(r'patient(\d+)').astype(int)
        if self.windows:
            df["Path"] = df["Path"].apply(lambda x:x[14:].replace("/","\\"))
        else:
            df["Path"] = df["Path"].apply(lambda x:x[14:])
        return df

    def preprocess_demo(self, excel_path):
        df = pd.read_csv(excel_path)
        df['PatientID'] = df['PATIENT'].str.extract(r'patient(\d+)').astype(int)
        return df

    def create_data_list(self, df:pd.DataFrame):
        data_list = list()
        if self.windows:
            df = df
        for i, row in df.iterrows():
            data_list.append(
                {"image": os.path.join(self.root_dir, row["Path"]), 
                 "PatientID": row["PatientID"],
                 "Sex":row["Sex"], 
                 "Age":row["Age"],
                 "View":row["Frontal/Lateral"],
                 "Race":row["PRIMARY_RACE"],
                 'Cardiomegaly': row['Cardiomegaly'], 
                 'Edema': row["Edema"],
                 "Consolidation":row['Consolidation'],
                 'Atelectasis':row['Atelectasis'],
                 "Pleural Effusion":row["Pleural Effusion"]
                 }
            )
        return data_list

    def _generate_data_list(self):
        ds_train = list()
        ds_val = list()
        ds_test = list()
        np.random.seed(self.seed)
        for d in self.datalist:
            rann = np.random.rand()
            if rann < self.test_frac:
                ds_test.append(d)
            elif rann < self.val_frac + self.test_frac:
                ds_val.append(d)
            else:
                ds_train.append(d)
        print(f"train:{len(ds_train)}|val:{len(ds_val)}|test:{len(ds_test)}")
        return ds_train, ds_val, ds_test
    
    def _generate_data_list_diffusion(self, race_range = ['white', 'black']):
        ds_train = list()
        ds_val = list()
        ds_test = list()
        race_map = {
            'Other': 'other',
            'White, non-Hispanic': 'white',
            'Black or African American': 'black',
            'White': 'white',
            'Native Hawaiian or Other Pacific Islander': 'other',
            'Asian': 'asian',
            'Asian, non-Hispanic': 'asian',
            'Unknown': 'other',
            'Native American, non-Hispanic': 'other',
            'Race and Ethnicity Unknown': 'other',
            'White, Hispanic': 'white',
            'nan': 'other',
            'Other, Hispanic': 'other',
            'Black, non-Hispanic': 'black',
            'American Indian or Alaska Native': 'other',
            'Patient Refused': 'other',
            'Other, non-Hispanic': 'other',
            'Pacific Islander, Hispanic': 'other',
            'Black, Hispanic': 'black',
            'Pacific Islander, non-Hispanic': 'other',
            'White or Caucasian': 'white',
            'Asian, Hispanic': 'asian',
            'Native American, Hispanic': 'other',
            'Asian - Historical Conv': 'asian'
        }
        num_nan = 0
        num_race = 0
        np.random.seed(2024)
        for d in self.datalist:
            if isinstance(d["Race"], float) and math.isnan(d["Race"]):
                num_nan += 1
                continue
            elif d['Race'] in race_map:
                r = d["Race"]
                if race_map[r] == "other":
                    continue
            else:
                raise KeyError(f"!!!!race {r} not in keys")
            d["Race"] = race_map[r]
            if d["Race"] not in race_range:
                num_race += 1
                continue
            rann = np.random.rand()
            if rann < self.test_frac:
                ds_test.append(d)
            elif rann < self.val_frac + self.test_frac:
                ds_val.append(d)
            else:
                ds_train.append(d)
        print(f"train:{len(ds_train)}|val:{len(ds_val)}|test:{len(ds_test)}, {num_nan} nan samples discarded, {num_race} other racial discarded")
        return ds_train, ds_val, ds_test
    
    def _generate_train_weights(self):
        weights = list()
        for d in self.train_ds:
            if d["Race"] == 'black':
                weights.append(10)
            elif d["Race"] == 'asian':
                weights.append(5)
            elif d["Race"] == "white":
                weights.append(1)
            else:
                raise ValueError(f"Race value {d['Race']} is not allowed")
        return weights
    

class MIMICCXRDataset(data.Dataset):

    def __init__(
        self,
        data_list,
        transform=None,
        image_resize=256,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        image_crop=None,
        filter_race=False
    ):
        self.data_list = data_list
        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5)
            ])
        else:
            self.transform = transform

        self.race_id = {
            'white': 0,
            'black': 1,
            'asian': 2
        }
        self.filter_race = filter_race

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        img = Image.open(item["image"]).convert("L")
        img_tensor = self.transform(img)
        if self.filter_race:
            return {
                'source': img_tensor,
                'target': 1,
                'race': self.race_id.get(item["Race"], -1),
                'sex': 0 if item["Sex"] == 'F' else 1,
                'image_path': item['image'],
                'patientID': item["PatientID"],
                'Cardiomegaly': item['Cardiomegaly'],
                'Edema': item["Edema"],
                'Consolidation': item['Consolidation'],
                'Atelectasis': item['Atelectasis'],
                'Pleural Effusion': item['Pleural Effusion']
            }
        else:
            return {'source': img_tensor, 'target': 1}



class MIMICCXRDataset_Loader:

    def __init__(
        self,
        root_dir,
        seed=0,
        val_frac=0.2,
        test_frac=0.2,
        filter_view=None,
        filter_race=False,
        race_range=['white', 'black']
    ):
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.root_dir = root_dir
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

        df_path = os.path.join(root_dir, "mimic_cxr_full_df.csv")
        if os.path.exists(df_path):
            self.df = pd.read_csv(df_path)
        else:
            csv_path = os.path.join(root_dir, "mimic-cxr.csv")
            demo_path = os.path.join(root_dir, "MIMIC_CXR_DEMO.csv")
            self.df = self._preprocess_data(csv_path, demo_path)
            self.df.to_csv(df_path, index=False)

        if filter_view:
            assert filter_view in ["Frontal", "Lateral"]
            self.df = self.df[self.df["View"] == filter_view]

        self.datalist = self.create_data_list(self.df)
        if not filter_race:
            self.train_ds, self.val_ds, self.test_ds = self._generate_data_splits()
        else:
            self.train_ds, self.val_ds, self.test_ds = self._generate_data_splits_race_filtered(race_range)

    def _preprocess_data(self, csv_path, demo_path):
        df = pd.read_csv(csv_path)
        df = df[["Path", "Sex", "Age", "View", "Race", 
                 'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']]
        demo_df = pd.read_csv(demo_path)
        df = pd.merge(df, demo_df[["PatientID", "Race"]], on="PatientID", how="inner")
        return df

    def create_data_list(self, df):
        data_list = []
        for _, row in df.iterrows():
            data_list.append({
                "image": os.path.join(self.root_dir, row["Path"]),
                "PatientID": row["PatientID"],
                "Sex": row["Sex"],
                "Age": row["Age"],
                "View": row["View"],
                "Race": row["Race"],
                'Cardiomegaly': row['Cardiomegaly'],
                'Edema': row["Edema"],
                'Consolidation': row['Consolidation'],
                'Atelectasis': row['Atelectasis'],
                'Pleural Effusion': row['Pleural Effusion']
            })
        return data_list

    def _generate_data_splits(self):
        np.random.seed(self.seed)
        ds_train, ds_val, ds_test = [], [], []
        for d in self.datalist:
            rann = np.random.rand()
            if rann < self.test_frac:
                ds_test.append(d)
            elif rann < self.val_frac + self.test_frac:
                ds_val.append(d)
            else:
                ds_train.append(d)
        print(f"train:{len(ds_train)}|val:{len(ds_val)}|test:{len(ds_test)}")
        return ds_train, ds_val, ds_test

    def _generate_data_splits_race_filtered(self, race_range):
        np.random.seed(self.seed)
        ds_train, ds_val, ds_test = [], [], []
        race_map = {
            'White': 'white',
            'Black or African American': 'black',
            'Asian': 'asian',
            # Add other mappings as needed
        }

        for d in self.datalist:
            d["Race"] = race_map.get(d["Race"], 'other')
            if d["Race"] not in race_range:
                continue
            rann = np.random.rand()
            if rann < self.test_frac:
                ds_test.append(d)
            elif rann < self.val_frac + self.test_frac:
                ds_val.append(d)
            else:
                ds_train.append(d)

        print(f"train:{len(ds_train)}|val:{len(ds_val)}|test:{len(ds_test)}")
        return ds_train, ds_val, ds_test
