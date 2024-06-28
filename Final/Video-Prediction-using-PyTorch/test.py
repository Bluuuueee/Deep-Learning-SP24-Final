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
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=96, help='number of hidden dim for ConvLSTM layers')

opt = parser.parse_args()


def testOnMasks(model, dir="D:/GameCenter/DeepLearning/Final/dataset/val",savefile="answer.npy"):
    all_videos = os.listdir(dir)
    cnt = len(all_videos)
    ans = np.zeros((cnt, 160,240))
    total_loss = 0
    total_loss_cnt = 0
    for i in tqdm(range(cnt)):
        f = os.path.join(dir, all_videos[i], "mask.npy")
        mask_raw = np.load(f)
        mask = mask_raw[0:11, :,:]
        mask_flat = mask.flatten()
        count = np.bincount(mask.flatten(), minlength=32)
        non_zero_indices = np.nonzero(count)[0]
        if non_zero_indices.shape[0] > 9:
            non_zero_indices = non_zero_indices[0:9].copy()
        permutation = np.random.permutation(np.arange(1, 9))
        permutation = np.concatenate(([0], permutation))

        value_to_index = {value: permutation[idx] for idx, value in enumerate(non_zero_indices)}
        index_to_value = {permutation[idx]: value for idx, value in enumerate(non_zero_indices)}
        mask_p = np.array([value_to_index.get(item, 0) for item in mask_flat])

        mask_p = mask_p.reshape(mask.shape)
        one_hot = np.eye(9)[mask_p]
        resized = np.zeros((11, 40, 60, 9))
        for k in range(11):
            for j in range(9):
                resized[k, :, :, j] = zoom(one_hot[k, :, :, j], (0.25, 0.25))
        resized = np.float32(resized)
        input = torch.tensor(resized).unsqueeze(0).permute(0, 1, 4, 2, 3)
        model.eval()
        answer = model(input, future_seq = 11)
        #answer = model.forward(input)
        # print(answer.shape)
        answer = answer.squeeze(0)[-1,:,:,:]
        # print(answer.shape)
        answer = answer.argmax(dim=-1).type(torch.IntTensor).numpy()
        answer_flat = answer.flatten()
        answer_r = np.array([index_to_value.get(item, 0) for item in answer_flat])
        answer_r = answer_r.reshape(answer.shape)
        answer_r = np.repeat(answer_r, repeats=4, axis=0)
        answer_r = np.repeat(answer_r, repeats=4, axis=1)
        ans[i,:,:] = answer_r
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        # loss = jaccard(torch.tensor(answer_r), torch.tensor(mask_raw[-1,:,:]))
        # total_loss += loss
        if True:
            loss = jaccard(torch.tensor(answer_r), torch.tensor(mask_raw[-1,:,:]))
            total_loss += loss
            total_loss_cnt += 1
            print("loss:",loss)
            print("avg loss:",total_loss/total_loss_cnt)
            print("loss_zeros:",jaccard(torch.tensor(np.zeros_like(answer_r)), torch.tensor(mask_raw[-1,:,:])))
            # plt.figure(figsize=(10, 5))
            
            # plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            # plt.imshow(answer_r)
            # plt.axis('off')

            # # Plot the second image
            # plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            # plt.imshow(mask_raw[-1,:,:])
            # plt.axis('off')

            # plt.show()
    np.save(savefile, ans)
  

  
if __name__ == '__main__':

    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=9)
    # torch.set_float32_matmul_precision('high')
    k = torch.load("epoch90_model_weights.pth")
    print(k.keys())
    conv_lstm_model.load_state_dict(k)
    conv_lstm_model.eval()
    testOnMasks(model=conv_lstm_model)
    # model = MovingMNISTLightning()
    # model = MovingMNISTLightning.load_from_checkpoint("epoch=164-step=13836.ckpt", model=conv_lstm_model)
    # model.eval()
    # testOnMasks(model=model)