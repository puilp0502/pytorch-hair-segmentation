import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class CelebHairDataset(Dataset):
    def __init__(self, root_dir, train=True, val=True, joint_transforms=None,
                 image_transforms=None, mask_transforms=None, gray_image=False):
        """
        Args:
            root_dir (str): root directory of dataset
            joint_transforms (torchvision.transforms.Compose): tranformation on both data and target
            image_transforms (torchvision.transforms.Compose): tranformation only on data
            mask_transforms (torchvision.transforms.Compose): tranformation only on target
            gray_image (bool): whether to return gray image image or not.
                               If True, returns img, mask, gray.
        """
        mode = 0 if train else (1 if val else 2)
        eval_partition_file = os.path.join(root_dir, 'list_eval_partition.txt')
        with open(eval_partition_file) as f:
            eval_partition_list = f.readlines()
        eval_partition_lut = {}
        for line in eval_partition_list:
            fname, f_mode = line.strip().split(' ')
            eval_partition_lut[fname.replace('.jpg', '')] = int(f_mode)
        img_dir = os.path.join(root_dir, 'img_align_celeba')
        mask_dir = os.path.join(root_dir, 'segmentation_masks')

        self.mask_list = list(sorted(os.listdir(mask_dir)))
        self.mask_list = list(filter(lambda s: eval_partition_lut.get(s.replace('.bmp', '')) == mode, self.mask_list))
        self.mask_list = [s.replace('.bmp', '') for s in self.mask_list]

        self.mask_path_list = [os.path.join(mask_dir, fname) + '.bmp' for fname in self.mask_list]
        self.img_path_list = [os.path.join(img_dir, fname) + '.jpg' for fname in self.mask_list]

        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.gray_image = gray_image

    def __getitem__(self,idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)

        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path)

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)
            
        if self.gray_image:
            gray = img.convert('L')
            gray = np.array(gray,dtype=np.float32)[np.newaxis,]/255

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        
        if self.gray_image:
            return img, (mask > 0.75).to(dtype=torch.float32), gray
        else:
            return img, (mask > 0.75).to(dtype=torch.float32)

    def __len__(self):
        return len(self.mask_path_list)

    def get_class_label(self, filename):
        """
        0: straight: frame00001-00150
        1: wavy: frame00151-00300
        2: curly: frame00301-00450
        3: kinky: frame00451-00600
        4: braids: frame00601-00750
        5: dreadlocks: frame00751-00900
        6: short-men: frame00901-01050
        """
        idx = int(filename.strip('Frame').strip('-gt.pbm'))

        if 0 < idx <= 150:
            return 0
        elif 150 < idx <= 300:
            return 1
        elif 300 < idx <= 450:
            return 2
        elif 450 < idx <= 600:
            return 3
        elif 600 < idx <= 750:
            return 4
        elif 750 < idx <= 900:
            return 5
        elif 900 < idx <= 1050:
            return 6
        raise ValueError
