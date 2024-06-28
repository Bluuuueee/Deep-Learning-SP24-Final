import socket
import numpy as np
import os
from scipy.ndimage import zoom
from tqdm import tqdm 
import random
from torchvision import datasets, transforms

# from: https://github.com/edenton/svg/blob/master/data/moving_mnist.py

class MaskDataset(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    
    def __init__(self, train, path="D:/GameCenter/DeepLearning/Final/dataset/train", seq_len=22, image_size=64, deterministic=True):
        self.path = path
        self.seq_len = seq_len
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False  # multi threaded loading
        self.channels = 9


        self.data = self.loadMasks()
        
        self.N = self.data.shape[0]
        print(self.N)

    def loadMasks(self):
        dir=self.path
        all_videos = os.listdir(dir)
        cnt = len(all_videos)
        data = np.zeros((cnt, self.seq_len, 40, 60, self.channels))
        #label = np.zeros((cnt, self.seq_len, 40, 60))
        for i in tqdm(range(cnt)):
            f = os.path.join(dir, all_videos[i], "mask.npy")
            mask = np.load(f).astype(int)
            #label[i, :, :, :] = mask
            mask_flat = mask.flatten()
            count = np.bincount(mask.flatten(), minlength=32)
            non_zero_indices = np.nonzero(count)[0]
            #print(non_zero_indices.shape[0])
            if non_zero_indices.shape[0] > 9:
                # print("!!!", i, "  !  ", non_zero_indices)
                non_zero_indices = non_zero_indices[0:9].copy()
            permutation = np.random.permutation(np.arange(1, 9))
            permutation = np.concatenate(([0], permutation))
            #print(permutation)

            value_to_index = {value: permutation[idx] for idx, value in enumerate(non_zero_indices)}
            #print(value_to_index)
            mask_p = np.array([value_to_index.get(item, 0) for item in mask_flat])
            mask_p = mask_p.reshape(mask.shape)
            one_hot = np.eye(9)[mask_p]
            resized = np.zeros((22, 40, 60, 9))
            for k in range(22):  # loop through each frame
                for j in range(9):  # loop through each channel
                    # Apply zoom for each channel of each frame
                    resized[k, :, :, j] = zoom(one_hot[k, :, :, j], (0.25, 0.25))
            data[i, :, :, :, :] = resized
        return data.astype(np.float32)
    
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        x = self.data[index, :, :, :, :]
        if random.uniform(0.0, 1.0) < 0.5:
            x = np.flip(x, axis=2).copy()
        return x


