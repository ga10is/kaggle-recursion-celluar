import os
import numpy as np
import pandas as pd

import cv2
import albumentations as A
import albumentations.pytorch as ATorch
from albumentations import ImageOnlyTransform

import torch
from torch.utils.data import Dataset

from . import config
from .common.logger import get_logger


class NormalizePerImage(ImageOnlyTransform):
    def __init__(self, max_pixel_value=255.0, always_apply=False, p=1.0):
        super(NormalizePerImage, self).__init__(always_apply, p)

    def apply(self, image, **params):
        """
        Parameters
        ----------
        image: np.ndarray of np.uint8 or np.float32, shape of [w, h, c]

        Returns
        -------
        normed: np.ndarray
        """
        normed = image.astype(np.float32)
        n_channels = image.shape[2]
        mean = np.mean(image.reshape(-1, n_channels), axis=0)
        std = np.std(image.reshape(-1, n_channels), axis=0)
        normed -= mean
        normed /= (std + 1e-8)
        return normed

    def get_trainsform_init_args_names(self):
        return 'max_pixel_value'


alb_trn_trnsfms = A.Compose([
    # A.CLAHE(p=1),
    A.Rotate(limit=10, p=1),
    # A.RandomSizedCrop((IMG_SIZE[0]-32, IMG_SIZE[0]-10), *INPUT_SIZE),
    A.RandomCrop(*config.INPUT_SIZE),
    # A.HueSaturationValue(val_shift_limit=20, p=0.5),
    # A.RandomBrightnessContrast(),
    # A.Resize(*INPUT_SIZE),
    NormalizePerImage(),
    # A.Normalize(
    # mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    # std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    # ),
    ATorch.transforms.ToTensor()
], p=1)


'''
    A.Normalize(
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
        always_apply=False
    ),
'''


alb_val_trnsfms = A.Compose([
    # A.CLAHE(p=1),
    # A.Resize(*INPUT_SIZE),
    A.CenterCrop(*config.INPUT_SIZE),
    A.Normalize(
        mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    ATorch.transforms.ToTensor()
], p=1)

alb_tst_trnsfms = A.Compose([
    # A.CLAHE(),
    A.Resize(*config.TEST_INPUT_SIZE),
    A.Normalize(
        mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    ATorch.transforms.ToTensor()
], p=1)


class CellerDataset(Dataset):
    target_name = 'sirna'
    image_name = ''
    id_name = 'id_code'
    site = 1

    def __init__(self, df, image_dir, transform, mode):
        self.df_org = df.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

        # Random Selection
        if mode == 'train':
            self.update()
        elif mode in ['valid', 'predict']:
            self.df_selected = self.df_org
        else:
            raise ValueError('Unexpected mode: %s' % mode)

    def __len__(self):
        return self.df_selected.shape[0]

    def __getitem__(self, idx):
        image_names = get_image_name(self.df_selected, idx, self.site)
        image = self.__load_image(image_names)
        # image = self.__load_image_with_rxrx(self.df_selected, idx)
        if self.mode in ['train', 'valid']:
            label = self.df_selected.iloc[idx][self.target_name]
        elif self.mode == 'predict':
            label = -1
        return image, torch.Tensor([label])

    '''
    def __load_image_with_rxrx(self, df, idx):
        dataset = 'train' if self.mode in ['train', 'valid'] else 'test'

        rcd = df.iloc[idx]
        experiment = rcd['experiment']
        plate = rcd['plate']
        well = rcd['well']
        image = rio.load_site('train', experiment, plate, well, 1)

        augmented = self.transform(image=image)
        image = augmented['image']
        return image
    '''

    def __load_image(self, image_names):
        """
        Parameters
        ----------
        image_names: list of str
        """

        images = []
        for image_name in image_names:
            image_path = os.path.join(self.image_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError('Not found image: %s' % image_path)
            images.append(image)
        images = np.stack(images, axis=2)

        augmented = self.transform(image=images)
        image = augmented['image']
        return image

    def update(self):
        if self.mode != 'train':
            raise ValueError('CellerDataset is not train mode.')
        self.df_selected = self.random_selection(config.N_SAMPLES)

    def random_selection(self, n_samples):
        g = self.df_org.groupby(self.target_name)[self.id_name]
        # selected = []
        # selected.append(np.random.choice(g.get_group(0).tolist(), n_samples, replace=False))
        # selected.append(np.random.choice(g.get_group(1).tolist(), n_samples, replace=False))
        selected = [np.random.choice(g.get_group(i).tolist(
        ), n_samples, replace=False) for i in range(config.N_CLASSES)]
        selected = np.concatenate(selected, axis=0)

        df_new = pd.DataFrame({self.id_name: selected})
        df_new = df_new.merge(self.df_org, on=self.id_name, how='left')
        get_logger().info('num of selected_images: %d' % len(df_new))

        return df_new


def get_image_name(df, idx, site):
    """
    Returns
    -------
    file_paths: list
        image path list of 6 channels
    """
    rcd = df.iloc[idx]
    experiment = rcd['experiment']
    plate = 'Plate%d' % rcd['plate']
    well = rcd['well']

    file_names = ['%s_s%d_w%d.png' % (well, site, i + 1) for i in range(6)]
    file_paths = [os.path.join(experiment, plate, file_name)
                  for file_name in file_names]
    return file_paths
