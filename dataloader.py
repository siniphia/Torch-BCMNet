import os
import torch
import numpy as np
import pandas as pd
import pydicom as dicom
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# size params
RSZ_W = 256
RSZ_H = 256
REAL_W = 50
REAL_H = 50

# path params
PROJECT_NAME = 'MULTISINUS'
PROJECT_PATH = "/home/lkr/PROJECTS/multisinus_torch/"
DATA_PATH = "/data/SNUBH/sinusitis/RAW_DEID/"
TRAIN_PATH = "sinus_train_deid/"
VTSNUBH_PATH = "sinus_vtsnubh_deid/"
VTSNUH_PATH = "sinus_vtsnuh_deid/"
XLSX_PATH = "/home/lkr/DATASETS/sinusitis/"


class SinusitisDataset(Dataset):
    def __init__(self, data_path, xlsx_path, run_type, data_size, transform):
        """
        Description
            Prepare image file path and label information before composing dataset
        Parameters
            data_path: path containing raw dicom files
            xlsx_path: xlsx file path
            run_type: select columns by run type - 'TRAIN', 'VAL', 'TEST'
            data_size: final dataset size
            transform: torchvision.transforms object for image augmentation
        """
        self.images = []
        self.infos = []
        self.transform = transform

        # 1 - Select xlsx columns by run type
        if run_type == 'TRAIN':
            lt_label, rt_label = 'LT_LABEL_USER1', 'RT_LABEL_USER1'
        elif run_type == 'VAL':
            lt_label, rt_label = 'LT_LABEL_CT', 'RT_LABEL_CT'
        elif run_type == 'TEST':
            lt_label, rt_label = 'LT_LABEL_CT', 'RT_LABEL_CT'
        else:
            print('Invalid run type')
            return

        # 2 - Read dataframe from xlsx file
        df = pd.read_excel(xlsx_path)
        df.set_index("SEED_NUM")
        df = df[['SEED_NUM', 'FILENAME', 'REVERSE', lt_label, rt_label,
                 'LT_COORD_X2', 'LT_COORD_Y2', 'RT_COORD_X2', 'RT_COORD_Y2',
                 'SPACING_X', 'SPACING_Y', 'ORIGINAL_H', 'ORIGINAL_W']]

        # 3 - Collect image path and information
        for idx, row in df.iterrows():
            # delete invalid images
            if int(row['REVERSE']) >= 2:
                continue
            # delete invalid labels
            if (int(row[lt_label]) >= 4) or (int(row[rt_label]) >= 4):
                continue
            # create binary labels
            else:
                lt_lbl, rt_lbl = 0, 0
                if int(row[lt_label]) > 0: lt_lbl = 1
                if int(row[rt_label]) > 0: rt_lbl = 1

                self.images += [os.path.join(data_path, row['FILENAME'])]
                self.infos += [[lt_lbl, rt_lbl,
                               int(row['LT_COORD_X2']), int(row['LT_COORD_Y2']),
                               int(REAL_W / float(row['SPACING_X'])), int(REAL_H / float(row['SPACING_Y'])),
                               int(row['RT_COORD_X2']), int(row['RT_COORD_Y2']),
                               int(REAL_W / float(row['SPACING_X'])), int(REAL_H / float(row['SPACING_Y'])),
                               int(row['REVERSE'])]]

        # 4 - Set data size
        if data_size != 0:
            self.images = self.images[:data_size]
            self.infos = self.infos[:data_size]

    def __getitem__(self, idx):
        """
        Description
            Create single data slice containing both maxiliary image patches and binary labels
        Return:
            Dictionary {'lt_maxil': lt_maxil, 'rt_maxil': rt_maxil, 'label': label}
        """
        # 1 - Read raw image and process reversed one
        img_path = self.images[idx]
        raw_img = dicom.read_file(img_path).pixel_array.astype('float32')
        if self.infos[idx][10] >= 1:  # AGFA
            mask = np.full_like(raw_img, np.amax(raw_img), raw_img.dtype)
            raw_img = np.subtract(mask, raw_img).astype('float32')

        # 2 - Create dataset
        info = self.infos[idx]
        lt_maxil = raw_img[info[3] - info[5] // 2:info[3] + info[5] // 2, info[2] - info[4] // 2:info[2] + info[4] // 2]
        rt_maxil = raw_img[info[7] - info[9] // 2:info[7] + info[9] // 2, info[6] - info[8] // 2:info[6] + info[8] // 2]
        lt_maxil = np.expand_dims(np.asarray(lt_maxil), -1)
        rt_maxil = np.expand_dims(np.asarray(rt_maxil), -1)

        lt_label = info[0]
        rt_label = info[1]
        if lt_label == 0 and rt_label == 0:
            label = 0
        elif lt_label == 0 and rt_label == 1:
            label = 1
        elif lt_label == 1 and rt_label == 0:
            label = 2
        elif lt_label == 1 and rt_label == 1:
            label = 3
        else:
            label = None
            print('Invalid Label : left %d, right %d' % (lt_label, rt_label))

        # 3 - Data augmentation
        if self.transform:
            lt_maxil = self.transform(lt_maxil)
            rt_maxil = self.transform(rt_maxil)

        # 4 - Final sample slice
        # sample = {'lt_maxil': lt_maxil, 'rt_maxil': rt_maxil, 'label': label}
        return lt_maxil, rt_maxil, label

    def __len__(self):
        return len(self.images)


def get_transform(flip, cont, trsl):
    """
    Description
        Resize and normalize basically
        Flipping, random contrast and translation are optional
    """
    transforms = []
    transforms.append(T.ToPILImage())
    transforms.append(T.Resize(size=(RSZ_H, RSZ_W)))
    if flip:
        transforms.append(T.RandomHorizontalFlip(p=1))
    if cont:
        transforms.append(T.ColorJitter(contrast=(0.5, 1.5)))
    if trsl:
        transforms.append(T.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.5], std=[0.5]))

    return T.Compose(transforms)
