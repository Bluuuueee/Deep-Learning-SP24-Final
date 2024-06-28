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

from models.seq2seq_ConvLSTM import EncoderDecoderConvLSTM
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
import cv2

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


def op(input): # 11, 160, 240, 9
    f, h, w, c = input.shape
    f_ahead = 11
    steps = 11
    output = np.zeros([h,w,c])
    for i in range(c):
        flows = []
        cur = (input[:,:,:,i] * 255).astype(np.uint8)
        for j in range(f-1):
            prev_frame = cur[j]
            next_frame = cur[j + 1]
            # if i == 3:
            #     plt.subplot(1, 1, 1)  # 1 row, 2 columns, 2nd subplot
            #     plt.imshow(next_frame)
            #     plt.axis('off')
            #     plt.show()
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
            
        frame = cur[-1,:,:].copy().astype(np.float32)
        for _ in range(steps):
            flow_map = flow

            # Create map coordinates from the flow
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = (x + flow_map[..., 0]).clip(0, w-1)
            y_new = (y + flow_map[..., 1]).clip(0, h-1)

            # Map the current frame to the next one using the flow
            frame = cv2.remap(frame, x_new.astype('float32'), y_new.astype('float32'), cv2.INTER_LINEAR)
            
            # if i == 3:
            #     plt.subplot(1, 1, 1)  # 1 row, 2 columns, 2nd subplot
            #     plt.imshow(frame)
            #     plt.axis('off')
            #     plt.show()
        # print("Last Frame: ", frame.shape)
        output[:,:,i] = frame
    return output

def testOnMasks(model, dir="D:/GameCenter/DeepLearning/Final/dataset/hidden",savefile="team_17_extra.npy", use_op = False):
    all_videos = os.listdir(dir)
    cnt = len(all_videos)
    ans = np.zeros((cnt, 160,240))
    total_loss = 0
    total_loss_cnt = 0
    for i in tqdm(range(cnt)):
        f = os.path.join(dir, all_videos[i], "mask_1.npy")
        # truth = np.load(os.path.join(dir, all_videos[i], "mask.npy")).astype(int)
        mask_raw = np.load(f).astype(int)
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
        if use_op:
            answer = op(one_hot)
            answer = answer.argmax(axis=-1)
        else:
            resized = np.zeros((11, 40, 60, 9))
            for k in range(11):
                for j in range(9):
                    resized[k, :, :, j] = zoom(one_hot[k, :, :, j], (0.25, 0.25))
            resized = np.float32(resized)
            input = torch.tensor(resized).unsqueeze(0).permute(0, 1, 4, 2, 3)
            model.eval()
            answer = model(input, future_seq = 11)
            answer = answer.squeeze(0)[-1,:,:,:]
            # print(answer.shape)
            answer = answer.argmax(dim=-1).type(torch.IntTensor).numpy()
        answer_flat = answer.flatten()
        answer_r = np.array([index_to_value.get(item, 0) for item in answer_flat])
        answer_r = answer_r.reshape(answer.shape)
        if use_op == False:
            answer_r = np.repeat(answer_r, repeats=4, axis=0)
            answer_r = np.repeat(answer_r, repeats=4, axis=1)
        ans[i,:,:] = answer_r
        # jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        # if True:
        #     loss = jaccard(torch.tensor(answer_r), torch.tensor(truth[-1,:,:]))
        #     total_loss += loss
        #     total_loss_cnt += 1
        #     print("loss:",loss)
        #     print("avg loss:",total_loss/total_loss_cnt)
        #     print("loss_zeros:",jaccard(torch.tensor(np.zeros_like(answer_r)), torch.tensor(truth[-1,:,:])))
            # plt.figure(figsize=(10, 5))
            
            # plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            # plt.imshow(answer_r)
            # plt.axis('off')

            # # Plot the second image
            # plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            # plt.imshow(mask_raw[-1,:,:])
            # plt.axis('off')

            # plt.show()
    np.save(savefile, ans.astype(int))
  

  
if __name__ == '__main__':

    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=9)
    # torch.set_float32_matmul_precision('high')
    k = torch.load("epoch32step37800_model_weights.pth")
    print(k.keys())
    conv_lstm_model.load_state_dict(k)
    conv_lstm_model.eval()
    testOnMasks(model=conv_lstm_model, use_op=False)