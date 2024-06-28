# import libraries
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
import test
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=96, help='number of hidden dim for ConvLSTM layers')

opt = parser.parse_args()


##########################
######### MODEL ##########
##########################

class MovingMNISTLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(MovingMNISTLightning, self).__init__()

        # default config
        self.path = os.getcwd() + '/data'
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        #self.criterion = torch.nn.MSELoss()
        k = 2.5
        class_weights = torch.tensor([1.0, k, k, k, k, k, k, k, k])
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.batch_size = opt.batch_size
        self.n_steps_past = 11
        self.n_steps_ahead = 11  # 4

    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
        x = x[0, :, :, :]
        y = y[0, :, :, :]
        y_hat = y_hat[0, :, :, :]
        x =torch.cat(tuple(x), dim=1)
        y= torch.cat(tuple(y), dim=1)
        y_hat= torch.cat(tuple(y_hat), dim=1)
        # print("X: ", x.shape)
        # print("Y: ",y.shape)
        # print("Y hat: ",y_hat.shape)
        preds = torch.cat([x.cpu(), y_hat.cpu()], dim=1)

        # entire input and ground truth
        y_plot = torch.cat([x.cpu(), y.cpu()], dim=1)

        # error (l2 norm) plot between pred and ground truth
        # difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        # zeros = torch.zeros(difference.shape)
        # difference_plot = torch.cat([zeros.cpu(), difference.cpu()], dim=1)[
        #     0].unsqueeze(1)

        # concat all images
        colors = torch.tensor([[0,0,0],[1,1,1],[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,0,1],[0,1,1],[0.5,0.5,0.5]])
        final_image = torch.cat([preds, y_plot], dim=0)#.unsqueeze(dim=0)
        #print(final_image.shape)
        colored = torch.zeros(final_image.shape[0], final_image.shape[1],3)
        for i in range(final_image.shape[0]):
            for j in range(final_image.shape[1]):
                #print(final_image[i,j].int().type())
                #print(final_image[i,j].int().shape)
                colored[i,j,:]=colors[final_image[i,j].int()]
        return colored.permute(2,0,1)

    def forward(self, x):
        x = x.to(device='cuda')

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x0 = x.permute(0, 1, 4, 2, 3)
        # print("in: ", x0.shape)
        y_hat = self.forward(x0)
        # print("out: ", y_hat.shape)

        loss = self.criterion(y_hat.permute(0,4,1,2,3), y.argmax(dim=-1))
        #loss =  basic_loss + yz
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 250 == 0:
                final_image = self.create_video(x.argmax(dim=-1), y_hat.argmax(dim=-1), y.argmax(dim=-1))

                self.logger.experiment.add_image(
                    'super_epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                
        if self.global_step % 200 == 0:
            torch.save(self.model.state_dict(), 'epoch' + str(self.current_epoch) + 'step' + str(self.global_step) + '_model_weights.pth')
        
        self.log('train_loss', loss)
        #self.log('learning_rate', lr_saved)
        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x0 = x.permute(0, 1, 4, 2, 3)
        y_hat = self.forward(x0)

        loss = self.criterion(y_hat.permute(0,4,1,2,3), y.argmax(dim=-1))
        #loss =  basic_loss + yz
        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        final_image = self.create_video(x.argmax(dim=-1), y_hat.argmax(dim=-1), y.argmax(dim=-1))

        self.logger.experiment.add_image(
            'Val_epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
            final_image, 0)
        
        self.log('val_loss', loss)
        
        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}
        
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}


    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))

    
    def train_dataloader(self):
        # train_data = MovingMNIST(
        #     train=True,
        #     data_root=self.path,
        #     seq_len=self.n_steps_past + self.n_steps_ahead,
        #     image_size=64,
        #     deterministic=True,
        #     num_digits=2)
        train_data = MaskDataset(
            train=True,
            path="D:/GameCenter/DeepLearning/Final/dataset/train",
            seq_len=22,
            image_size=64)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True)

        return train_loader

    def val_dataloader(self):
        
        val_data = MaskDataset(
            train=True,
            path="D:/GameCenter/DeepLearning/Final/dataset/val",
            seq_len=22,
            image_size=64)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=True)
        return val_loader
    
    def test_dataloader(self):
        test_data = MovingMNIST(
            train=False,
            data_root=self.path,
            seq_len=self.n_steps_past + self.n_steps_ahead,
            image_size=64,
            deterministic=True,
            num_digits=2)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=True)

        return test_loader


def run_trainer():
    
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=9)
    # checkpoint = torch.load("epoch=238-step=20076.ckpt") # ie, model_best.pth.tar
    # conv_lstm_model.load_state_dict(checkpoint['state_dict', 'model'])
    # model = MovingMNISTLightning.load_from_checkpoint("epoch=164-step=13836.ckpt", model=conv_lstm_model)
    model = MovingMNISTLightning(model=conv_lstm_model)

    # model = MovingMNISTLightning.load_from_checkpoint("epoch=238-step=20076.ckpt", model=conv_lstm_model)
    logger = TensorBoardLogger("tb_logs", name="Super_1")
    #logger = TensorBoardLogger("tb_logs", name="TEST1")
    trainer = Trainer(max_epochs=opt.epochs,
                      log_every_n_steps=2,
                      logger = logger,
                      default_root_dir="saved_model/",
                      val_check_interval=150,
                      limit_val_batches=2
                      # gpus=opt.n_gpus,
                      # distributed_backend='dp',
                      # early_stop_callback=False,
                      # use_amp=opt.use_amp
                      )

    trainer.fit(model)
    model.eval()
    conv_lstm_model.eval()
    torch.save(conv_lstm_model.state_dict(), 'super_final_model_weights.pth')
    test.testOnMasks(conv_lstm_model)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    run_trainer()

    # p1 = Process(target=run_trainer)                    # start trainer
    # p1.start()
    # #p2 = Process(target=run_tensorboard(new_run=True))  # start tensorboard
    # #p2.start()
    # p1.join()
    # #p2.join()



