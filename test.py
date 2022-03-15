import os
import cv2
import torch

from swinir import SwinIR


model = {
    "x4": SwinIR(
        upscale=4, in_chans=1, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    ),
}
params_path = {
    "x4": "pretrain_swin.pth",
}

test_data_path = {
    "x4": "./testingSetInput/evaluation1/hr_x4",
    "x2": "./testingSetInput/evaluation2/mr_real",
}
output_path = {
    "x4": "submitStructure/evaluation1/x4",
    "x2": "submitStructure/evaluation2/x2",
}

# NOTE: for x4
model_x4 = model["x4"].cuda().eval()
model_x4_params = torch.load(params_path["x4"])
model_x4.load_state_dict(model_x4_params)
with torch.no_grad():
    for i, img_path in enumerate(os.listdir(test_data_path["x4"])):
        img_file_path = os.path.join(test_data_path["x4"], img_path)
        img = cv2.imread(img_file_path, 0)
        img_input = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda() / 255
        img_output = model_x4(img_input).cpu().squeeze().numpy() * 255
        cv2.imwrite(os.path.join(output_path["x4"], "ev1_{}.jpg".format(img_path.split(".")[0][-3:])), img_output)

# NOTE: for x2
with torch.no_grad():
    for i, img_path in enumerate(os.listdir(test_data_path["x2"])):
        img_file_path = os.path.join(test_data_path["x2"], img_path)
        img = cv2.imread(img_file_path, 0)
        h, w = img.shape
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        img_input = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda() / 255
        img_output = torch.pow(model_x4(img_input).clamp(0, 1), 0.9).cpu().squeeze().numpy() * 255
        cv2.imwrite(os.path.join(output_path["x2"], "ev2_{}.jpg".format(img_path.split(".")[0][-3:])), img_output)
