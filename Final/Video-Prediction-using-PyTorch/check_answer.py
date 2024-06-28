import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lightning as pl
from lightning import Trainer
from multiprocessing import Process

from utils.start_tensorboard import run_tensorboard
from models.seq2seq_ConvLSTM import EncoderDecoderConvLSTM
from data.MovingMNIST import MovingMNIST
from data.MaskDataset import MaskDataset
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
import numpy as np
# from main import MovingMNIST, MovingMNISTLightning
from scipy.ndimage import zoom
from tqdm import tqdm 
import random
import matplotlib.pyplot as plt
import torchmetrics

def CheckAnswer(dir="D:/GameCenter/DeepLearning/Final/dataset/hidden", ans_flie="team_17.npy"):
    ans = np.load(ans_flie)
    print(ans.dtype)
    print(ans.shape)
    all_videos = os.listdir(dir)
    cnt = len(all_videos)
    for i in tqdm(range(cnt)):
        f = os.path.join(dir, all_videos[i], "mask_1.npy")
        mask_raw = np.load(f).astype(int)
        mask = mask_raw[0:11, :,:].astype(int)
        count = np.bincount(mask.flatten(), minlength=32)
        non_zero_indices = np.nonzero(count)[0]

        cur = ans[i,:,:].astype(int)
        count_2 =  np.bincount(cur.flatten(), minlength=32)
        non_zero_indices_2 = np.nonzero(count_2)[0]
        
        print("A: ",non_zero_indices)
        print("B: ",non_zero_indices_2)
        if i%30 == 0:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            plt.imshow(mask_raw[-1, :, :])
            plt.axis('off')

            # Plot the second image
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            plt.imshow(cur)
            plt.axis('off')

            plt.show()
  

  
if __name__ == '__main__':
    # ans_flie="team_17.npy"
    # ans = np.load(ans_flie)
    # print(ans.dtype)
    # ans = ans.astype(int).copy()
    # print(ans.dtype)
    # np.save("team_17_1", ans)

    CheckAnswer()
    # model = MovingMNISTLightning()
    # model = MovingMNISTLightning.load_from_checkpoint("epoch=164-step=13836.ckpt", model=conv_lstm_model)
    # model.eval()
    # testOnMasks(model=model)