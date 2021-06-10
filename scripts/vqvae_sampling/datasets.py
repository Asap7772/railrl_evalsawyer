import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from torch.utils.data import Dataset


class BlockDataset(Dataset):
    """
    Creates block dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading block data')
        data = np.array([cv2.resize(x[0][0][:, :, :3], dsize=(
            32, 32), interpolation=cv2.INTER_CUBIC) for x in data])

        n = data.shape[0]
        cutoff = n//10
        self.data = data[:-cutoff] if train else data[-cutoff:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset 
    """

    def __init__(self, file_path, train=True, transform=None, test_p=0.9):
        print('Loading latent block data')
        self.all_data = np.load(file_path, allow_pickle=True)
        data = np.load(file_path, allow_pickle=True).reshape(-1, 12, 12)
        print('Done loading latent block data')
        
        n = int(data.shape[0] * test_p)
        self.data = data[:n] if train else data[n:]
        self.transform = transform

    def get_closest_stats(self, latents):
        from scipy import spatial
        tree = spatial.KDTree(self.all_data)
        all_dists = []

        for i in range(latents.shape[0]):
            dist, index = tree.query(latents[i])
            all_dists.append(dist)
        all_dists = np.array(all_dists)
        print("Mean:", all_dists.mean())
        print("Std:", all_dists.std())
        print("Min:", all_dists.min())
        print("Max:", all_dists.mean())



    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)