import torch
import torch.nn.functional as F
from torch.utils.data import dataloader
import matplotlib.pyplot as plt
from tfrecord.torch.dataset import TFRecordDataset

import utils


description = {
    "low": "byte",
    "high": "byte",
    # "clean": "byte",
    "h": "int",
    "w": "int",
}
train_dataset = TFRecordDataset("valid_x2.tfrecord", None, description, shuffle_queue_size=256)
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

psnr = utils.AverageMeter()
ssim = utils.AverageMeter()
for cnt, record in enumerate(train_dataloader):
    low_images = record["low"].reshape(
        1,
        1,
        record["h"][0],
        record["w"][0],
    ).float()
    # clean_images = record["clean"].reshape(
    #     1,
    #     1,
    #     record["h"][0] // 4,
    #     record["w"][0] // 4,
    # ).float()
    high_images = record["high"].reshape(
        1,
        1,
        record["h"][0] * 2,
        record["w"][0] * 2,
    ).float()

    high_images_pred = F.interpolate(
        low_images, scale_factor=2, mode="bilinear", align_corners=False
    )
    # NOTE: gamma, no more need
    high_images_pred = torch.pow(high_images_pred, 1.005)

    plt.subplot(1, 3, 1)
    plt.imshow(low_images.squeeze().numpy(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(high_images_pred.squeeze().numpy(), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(high_images.squeeze().numpy(), cmap="gray")

    psnr.update(utils.calc_psnr(high_images_pred, high_images))
    ssim.update(utils.calc_ssim(high_images_pred, high_images).item())

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())  # NOTE: only for tkinter backend
    # mng.full_screen_toggle()
    plt.show()

    if cnt == 9:
        break

# print(psnr.avg, ssim.avg)
