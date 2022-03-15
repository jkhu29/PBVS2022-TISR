import os
import random
import warnings

import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm

import config
import utils
from loss import *

from rcan import *
from vgg import GAN_D
from swinir import SwinIR


model_dict = {
    "rcan": RCAN(scale=4, num_features=64, num_rg=10, num_rcab=20),
    "swin": SwinIR(
        upscale=4, in_chans=1, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )
}

criterion_dict = {
    "l1": nn.L1Loss(),
    "l2": nn.MSELoss(),
    "cl1": L1_Charbonnier_loss(),
}

# torch.autograd.set_detect_anomaly(True)


def train_init():
    # deveice init
    CUDA_ENABLE = torch.cuda.is_available()
    if CUDA_ENABLE and opt.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = False
    elif CUDA_ENABLE and not opt.cuda:
        warnings.warn("WARNING: You have CUDA device, so you should probably run with --cuda")
    elif not CUDA_ENABLE and opt.cuda:
        assert CUDA_ENABLE, "ERROR: You don't have a CUDA device"

    device = torch.device('cuda:0' if CUDA_ENABLE else 'cpu')

    # seed init
    manual_seed = opt.seed
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # dataset init, need .tfrecord
    description = {
        "low": "byte",
        "high": "byte",
        # "clean": "byte",
        "h": "int",
        "w": "int",
    }
    train_dataset = TFRecordDataset(opt.train_file, None, description, shuffle_queue_size=opt.batch_size)
    train_dataloader = dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    valid_dataset = TFRecordDataset(opt.valid_file, None, description)
    valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

    return device, train_dataloader, valid_dataloader, 951 * 2


def model_pretrain(opt, device, train_dataloader, valid_dataloader, length):
    weight_cliping_limit = 0.01

    # models init
    model = model_dict[opt.model_name].to(device)
    if os.path.exists("pretrain_" + opt.model_name + ".pth"):
        print("loading: " + "pretrain_" + opt.model_name + ".pth")
        params = torch.load("pretrain_" + opt.model_name + ".pth")
        model.load_state_dict(params)

    vgg = GAN_D().to(device)

    # criterion init
    criterion = criterion_dict["cl1"]

    # optim init
    if opt.adam:
        model_optimizer = optim.AdamW(
            model.parameters(),
            lr=opt.lr, eps=1e-8, weight_decay=0.01, betas=(0.5, 0.999)
        )
        vgg_optimizer = optim.AdamW(
            vgg.parameters(),
            lr=opt.lr, eps=1e-8, weight_decay=0.01, betas=(0.5, 0.999)
        )
    else:
        model_optimizer = optim.RMSprop(
            model.parameters(), lr=opt.lr
        )
        vgg_optimizer = optim.RMSprop(
            vgg.parameters(), lr=opt.lr
        )
    model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

    if opt.save_model_pdf:
        from torchviz import make_dot
        sample_data = torch.rand(1, 1, 32, 32).to(device)
        out = model(sample_data)
        d = make_dot(out)
        d.render('modelviz.pdf', view=False)

    # train
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    for epoch in range(opt.niter):
        epoch_losses = utils.AverageMeter()
        epoch_losses_tv = utils.AverageMeter()
        epoch_losses_vgg = utils.AverageMeter()
        epoch_losses_psnr = utils.AverageMeter()
        epoch_losses_ssim = utils.AverageMeter()
        # epoch_losses_feature = utils.AverageMeter()

        with tqdm(total=(length * 1 - length * 1 % opt.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))
            
            for data_loader in [train_dataloader]:
                for record in data_loader:
                    low_images = record["low"].reshape(
                        opt.batch_size,
                        1,
                        record["h"][0] // opt.scale,
                        record["w"][0] // opt.scale,
                    ).float().to(device) / 255
                    high_images = record["high"].reshape(
                        opt.batch_size,
                        1,
                        record["h"][0],
                        record["w"][0],
                    ).float().to(device) / 255

                    preds = model(low_images)

                    ### -------
                    # vgg train
                    ### -------
                    vgg.train()
                    for p in vgg.parameters():
                        p.data.clamp_(weight_cliping_limit, weight_cliping_limit)
                    for _ in range(4):
                        loss_real_d = vgg(high_images).mean()
                        loss_fake_d = vgg(Variable(preds)).mean()

                        vgg.zero_grad()
                        loss_real_d.backward(mone)
                        loss_fake_d.backward(one)

                        # train with gradient penalty
                        # loss_gradient_penalty = gradient_penalty(vgg, high_images, preds)
                        # loss_gradient_penalty.backward()

                        vgg_optimizer.step()

                        loss_d = loss_real_d - loss_fake_d
                        epoch_losses_vgg.update(loss_d.item(), opt.batch_size)
                    vgg.eval()

                    model.train()
                    model_optimizer.zero_grad()
                    loss = criterion(preds, high_images)
                    loss_tv = TVLoss()(preds)
                    # features_pred = vgg(preds)
                    # features_real = vgg(high_images)
                    # loss_feature = 0
                    # for f_fake, f_real in zip(features_pred, features_real):
                    #     gram_fake = utils.calc_gram(f_fake)
                    #     gram_real = utils.calc_gram(f_real)
                    #     loss_feature += criterion(gram_fake, gram_real)
                    total_loss = loss + (1 - utils.calc_ssim(preds, high_images)) * 1000 + loss_tv * 10
                    epoch_losses.update(loss.item(), opt.batch_size)
                    epoch_losses_tv.update(loss_tv.item(), opt.batch_size)
                    epoch_losses_psnr.update(utils.calc_psnr(preds, high_images), opt.batch_size)
                    epoch_losses_ssim.update(utils.calc_ssim(preds, high_images).item(), opt.batch_size)
                    # epoch_losses_feature.update(loss_feature.item(), opt.batch_size)
                    total_loss.backward()
                    model_optimizer.step()
                    model.eval()

                    t.set_postfix(
                        loss='{:.6f}'.format(epoch_losses.avg),
                        loss_tv='{:.6f}'.format(epoch_losses_tv.avg),
                        loss_vgg='{:.6f}'.format(epoch_losses_vgg.avg),
                        psnr='{:.6f}'.format(epoch_losses_psnr.avg),
                        ssim='{:.6f}'.format(epoch_losses_ssim.avg),
                        # loss_feature='{:.6f}'.format(epoch_losses_feature.avg),
                        # loss_denoise='{:.6f}'.format(epoch_losses_denoise.avg),
                    )
                    t.update(opt.batch_size)
                torch.cuda.empty_cache()

        model_scheduler.step()
        # model_denoise_scheduler.step()

        # test
        epoch_pnsr = utils.AverageMeter()
        epoch_ssim = utils.AverageMeter()
        
        with torch.no_grad():
            for cnt, record in enumerate(valid_dataloader):
                low_images_val = record["low"].reshape(
                    1,
                    1,
                    record["h"][0] // opt.scale,
                    record["w"][0] // opt.scale,
                ).float().to(device) / 255
                high_images_val = record["high"].reshape(
                    1,
                    1,
                    record["h"][0],
                    record["w"][0],
                ).float().to(device) / 255

                # clean_preds_val = low_images_val
                preds_val = model(low_images_val)
                epoch_pnsr.update(utils.calc_psnr(preds_val, high_images_val), 1)
                epoch_ssim.update(utils.calc_ssim(preds_val, high_images_val), 1)
                if cnt == 0:
                    cv2.imwrite("preds.png", preds_val.squeeze().cpu().numpy() * 255)
                    # cv2.imwrite("clean.png", clean_preds_val.squeeze().cpu().numpy() * 255)
                    cv2.imwrite("label.png", high_images_val.squeeze().cpu().numpy() * 255)

        print('epoch: {} eval psnr: {:.6f} eval ssim: {:.6f}'.format(epoch+1, epoch_pnsr.avg, epoch_ssim.avg))
        torch.save(model.state_dict(), "./models/epoch_{}.pth".format(epoch+1))
        # torch.save(denoise_model.state_dict(), "./models/denoise_epoch_{}.pth".format(epoch+1))
        torch.cuda.empty_cache()

    return model


if __name__ == "__main__":
    opt = config.get_options()
    device, train_dataloader, valid_dataloader, length = train_init()
    model = model_pretrain(opt, device, train_dataloader, valid_dataloader, length)
