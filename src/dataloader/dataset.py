import os
from torch.utils.data import Dataset
import cv2
import numpy as np

class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="GCPS_train.txt",
            val_file_dir="GCPS_val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.imread(os.path.join(self._base_dir, 'test', 'd1',case + '.png'))
        # image = \
        #     cv2.imread(os.path.join(self._base_dir, 'd2_mask', case + '.png'), cv2.IMREAD_GRAYSCALE)[
        #         ..., None]
        label = \
            cv2.imread(os.path.join(self._base_dir, 'masks', case + '.png'), cv2.IMREAD_GRAYSCALE)[
                ..., None]
        # label = \
        #     cv2.imread(os.path.join(self._base_dir, 'masks', case + '_anno.bmp'), cv2.IMREAD_GRAYSCALE)[
        #         ..., None]
        #
        # label[label > 0] = 255
        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        # sample = {"image": image, "label": label, "idx": idx,"name":os.path.join('d3',case + '.png')}
        sample = {"image": image, "label": label, "idx": idx,}
        return sample
