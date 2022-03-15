import os
import copy
import itertools

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
from torchvision import transforms
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm
# from tqdm.notebook import tqdm

import config
from swinir import SwinIR
from vgg import Vgg16, GAN_D
import utils


class BasicCycleGAN(object):
    """"""
    def __init__(self):
        super(BasicCycleGAN, self).__init__()
        opt = config.get_cyclegan_options()
        self.train_file = opt.train_file
        self.valid_file = opt.valid_file
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        self.niter = opt.niter
        self.batch_size = opt.batch_size
        self.workers = opt.workers
        self.batch_scale = opt.batch_scale
        self.lr = opt.lr

        torch.backends.cudnn.benchmark = True

        self.target_fake = Variable(torch.rand(self.batch_size) * 0.3).to(self.device)
        self.target_real = Variable(torch.rand(self.batch_size) * 0.5 + 0.7).to(self.device)

        # cyclegan for bgan, init
        self.model_g_x2y = SwinIR(upscale=1, in_chans=3, out_chans=1, img_size=64, window_size=8,
                img_range=1., depths=[4, 4], embed_dim=60, num_heads=[4, 4],
                mlp_ratio=2, upsampler='', resi_connection='1conv').to(self.device)
        self.model_g_y2x = SwinIR(upscale=1, in_chans=1, out_chans=1, img_size=64, window_size=8,
                img_range=1., depths=[4, 4], embed_dim=60, num_heads=[4, 4],
                mlp_ratio=2, upsampler='', resi_connection='1conv').to(self.device)
        self.model_d_x = GAN_D(in_channels=1).to(self.device)
        self.model_d_y = GAN_D(in_channels=1).to(self.device)
        self.vgg = Vgg16().to(self.device)

        if os.path.exists("pretrain_x2y.pth"):
            bgan_params = torch.load("pretrain_y2x.pth")
            dbgan_params = torch.load("pretrain_x2y.pth")
            self.model_g_x2y.load_state_dict(bgan_params)
            self.model_g_y2x.load_state_dict(dbgan_params)
        else:
            self.model_g_x2y.apply(utils.weights_init)
            self.model_g_y2x.apply(utils.weights_init)

        self.model_d_x.apply(utils.weights_init)
        self.model_d_y.apply(utils.weights_init)

        # criterion init
        self.criterion_generate = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # dataset init
        description = {
            "low": "byte",
            "high": "byte",
            "h": "int",
        }
        train_dataset = TFRecordDataset(self.train_file, None, description, shuffle_queue_size=1024)
        self.train_dataloader = dataloader.DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True, 
            drop_last=True
        )
        self.data_length = 1902
        valid_dataset = TFRecordDataset(self.valid_file, None, description)
        self.valid_dataloader = dataloader.DataLoader(
            dataset=valid_dataset, 
            batch_size=1
        )

        # optim init
        self.optimizer_g = optim.Adam(
            itertools.chain(self.model_g_x2y.parameters(), self.model_g_y2x.parameters()), 
            lr=self.lr, betas=(0.75, 0.999)
        )
        self.optimizer_d_x = optim.Adam(
            self.model_d_x.parameters(), 
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_d_y = optim.Adam(
            self.model_d_y.parameters(), 
            lr=self.lr, betas=(0.5, 0.999)
        )

        # lr init
        self.model_scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=self.niter
        )
        self.model_scheduler_d_x = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d_x, T_max=self.niter
        )
        self.model_scheduler_d_y = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d_y, T_max=self.niter
        )

    def train_batch(self):
        print("-----------------train-----------------")
        cnt = 0
        for epoch in range(self.niter):
            epoch_losses_g_content = utils.AverageMeter()
            epoch_losses_g_style = utils.AverageMeter()
            epoch_losses_d_x = utils.AverageMeter()
            epoch_losses_d_y = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch+1, self.niter))

                for record in self.train_dataloader:
                    cnt += 1
                    blur = record["low"].reshape(
                        self.batch_size, 
                        1, 
                        record["h"][0],
                        record["h"][0]
                    ).float().to(self.device)
                    sharp = record["high"].reshape(
                        self.batch_size, 
                        1, 
                        record["h"][0],
                        record["h"][0]
                    ).float().to(self.device)
                    n, c, h, w = sharp.shape
                    blur_noise = utils.concat_noise(blur, (c + 1, h, w), n)
                    sharp_noise = utils.concat_noise(sharp, (c + 1, h, w), n)

                    # --------------------
                    # generator train(2 * model_g)
                    # --------------------
                    loss_content, blur_fake, sharp_fake = self._calc_loss_g(blur_noise, blur, sharp_noise, sharp)
                    loss_style = self._calc_loss_style(self.vgg(blur), self.vgg(blur_fake))
                    loss_total = 0.01 * loss_content + loss_style

                    self.model_g_x2y.train()
                    self.model_g_y2x.train()
                    if cnt % self.batch_scale == 0:
                        self.optimizer_g.zero_grad()
                        loss_total.backward()
                        epoch_losses_g_content.update(loss_content.item(), self.batch_size)
                        epoch_losses_g_style.update(loss_style.item(), self.batch_size)
                        self.optimizer_g.step()
                    self.model_g_x2y.eval()
                    self.model_g_y2x.eval()

                    # --------------------
                    # discriminator sharp train(model_d_x)
                    # -------------------- 
                    self.model_d_x.train()
                    loss_total_d_x = self._calc_loss_d(self.model_d_x, sharp_fake, sharp)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_d_x.zero_grad()
                        loss_total_d_x.backward()
                        epoch_losses_d_x.update(loss_total_d_x.item(), self.batch_size)
                        self.optimizer_d_x.step()
                    self.model_d_x.eval()

                    # --------------------
                    # discriminator blur train(model_d_y)
                    # -------------------- 
                    self.model_d_y.train()
                    loss_total_d_y = self._calc_loss_d(self.model_d_y, blur_fake, blur)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_d_y.zero_grad()
                        loss_total_d_y.backward()
                        epoch_losses_d_y.update(loss_total_d_y.item(), self.batch_size)
                        self.optimizer_d_y.step()
                    self.model_d_y.eval()

                    t.set_postfix(
                        loss_content='{:.6f}'.format(epoch_losses_g_content.avg), 
                        loss_style='{:.6f}'.format(epoch_losses_g_style.avg), 
                        loss_d_sharp='{:.6f}'.format(epoch_losses_d_x.avg), 
                        loss_d_blur='{:.6f}'.format(epoch_losses_d_y.avg)
                    )
                    t.update(self.batch_size)

                torch.save(self.model_g_x2y.state_dict(), "./models/x2y_epoch_{}.pth".format(epoch+1))
                torch.save(self.model_g_y2x.state_dict(), "./models/y2x_epoch_{}.pth".format(epoch+1))

            self.model_scheduler_g.step()
            self.model_scheduler_d_x.step()
            self.model_scheduler_d_y.step()

    def _calc_loss_g(self, blur_noise, blur_real, sharp_noise, sharp_real):
        # loss identity(ATTN!: `a_same = model_a2b(a_real)`)
        _, c, h, w = blur_real.shape
        blur_same = self.model_g_x2y(blur_noise)            # model_g_x2y: sharp --> blur
        loss_identity_blur = self.criterion_identity(blur_same, blur_real)

        sharp_fake = self.model_g_y2x(sharp_real)           # model_g_y2x: blur --> sharp
        loss_identity_sharp = self.criterion_identity(sharp_fake, sharp_real)

        # loss gan
        blur_fake = self.model_g_x2y(sharp_noise)
        blur_fake_pred = self.model_d_y(blur_fake)          # get blur features
        loss_gan_x2y = self.criterion_generate(blur_fake_pred, self.target_real)

        sharp_fake = self.model_g_y2x(blur_real)
        sharp_fake_pred = self.model_d_x(sharp_fake)        # get sharp features
        loss_gan_y2x = self.criterion_generate(sharp_fake_pred, self.target_real)

        sharp_fake_noise = utils.concat_noise(sharp_fake, (c + 1, h, w), blur_real.size()[0])

        # loss cycle
        blur_recover = self.model_g_x2y(sharp_fake_noise)   # recover the blur: blur->sharp->blur
        loss_cycle_x2y = self.criterion_cycle(blur_recover, blur_real) * 2

        sharp_recover = self.model_g_y2x(blur_fake)         # recover the sharp: sharp->blur->sharp
        loss_cycle_y2x = self.criterion_cycle(sharp_recover, sharp_real) * 2

        # loss total
        loss_total = loss_identity_blur + loss_identity_sharp + \
                     loss_gan_x2y + loss_gan_y2x + \
                     loss_cycle_x2y + loss_cycle_y2x

        return loss_total, blur_fake, sharp_fake
    
    def _calc_loss_style(self, features_fake, features_real, loss_style=0):
        for f_fake, f_real in zip(features_fake, features_real):
            gram_fake = utils.calc_gram(f_fake)
            gram_real = utils.calc_gram(f_real)
            loss_style += self.criterion_generate(gram_fake, gram_real)
        return loss_style

    def _calc_loss_d(self, model_d, fake, real):
        # loss real
        pred_real = torch.sigmoid(model_d(real))
        loss_real = self.criterion_generate(pred_real, self.target_real)

        # loss fake
        fake_ = fake.clone()
        with torch.no_grad():
            pred_fake = torch.sigmoid(model_d(fake_))
        loss_fake = self.criterion_generate(pred_fake, self.target_fake)

        # loss rbl
        loss_rbl = - torch.log(abs(loss_real - loss_fake)) - \
                     torch.log(abs(1 - loss_fake - loss_real))

        # loss total
        loss_total = (loss_real + loss_fake) * 0.5 + loss_rbl * 0.01

        return loss_total


if __name__ == "__main__":
    bgan = BasicCycleGAN()
    bgan.train_batch()
