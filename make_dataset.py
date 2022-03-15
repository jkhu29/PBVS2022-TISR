import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

import tfrecord


cnt = 0
writer = tfrecord.TFRecordWriter("train_x2.tfrecord")
low_path = "./data/train/320_axis_mr"
high_path = "./data/train/640_flir_hr"

# NOTE: code for sr_x4, writer should be "train_x4.tfrecord"
# for file_name in os.listdir(low_path):
#     # low_image_path = os.path.join(low_path, file_name)
#     high_image_path = os.path.join(high_path, file_name)

#     images = []
#     # image = cv2.imread(low_image_path, 0)
#     high_image = cv2.imread(high_image_path, 0)
#     h, w = high_image.shape
#     images.append(high_image)
#     # images.append(high_image[0:h, 0:w])
#     # images.append(high_image[h-1:-1, w-1:-1])
#     # images.append(high_image[h-1:-1, 0:w])
#     # images.append(high_image[0:h, w-1:-1])
#     # for i in [45, 90, 135, 180, 225, 270, 315]:
#     #     scale = 2
#     #     rotate = cv2.getRotationMatrix2D((h * 0.5, w * 0.5), i, scale)
#     #     res = cv2.warpAffine(image, rotate, (w, h))
#     #     images.append(res)
#     for i in range(len(images)):
#         images.append(cv2.flip(images[i], 1))

#     for img in images:
#         noisy_image = img + np.random.normal(0, 10**0.5, img.shape)
#         noise_image = cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1).astype(np.uint8)
#         lower_image = cv2.resize(noise_image, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)

#         writer.write({
#             # "lower": (lower_image.tobytes(), "byte"),
#             "low": (lower_image.tobytes(), "byte"),
#             "high": (img.tobytes(), "byte"),
#             "h": (h, "int"),
#             "w": (w, "int"),
#         })
#         cnt += 1

# NOTE: code for paried sr_x2, writer should be "train_x2.tfrecord"
# orb = cv2.ORB_create()
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# for file_name in os.listdir(low_path):
#     low_image_path = os.path.join(low_path, file_name)
#     high_image_path = os.path.join(high_path, file_name)

#     img1 = cv2.imread(low_image_path, 0)
#     img2 = cv2.imread(high_image_path, 0)

#     # find the keypoints and descriptors with ORB
#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)

#     # Match descriptors.
#     matches = bf.match(des1, des2)
#     # Sort them in the order of their distance.
#     matches = sorted(matches, key=lambda x: x.distance)

#     goodMatch = matches[:100]
#     ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
#     ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
#     ransacReprojThreshold = 4
#     H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
#     img_out = cv2.warpPerspective(
#         img1, H, 
#         (img2.shape[1], img2.shape[0]), 
#         flags=cv2.INTER_LINEAR
#     )
#     img_out = cv2.resize(img_out, (img1.shape[1], img1.shape[0]))

#     # img_out is low image, img2 is high image
#     low_images = []
#     high_images = []
#     h, w = img_out.shape
#     low_images.append(img_out)
#     high_images.append(img2)
#     # for i in [45, 90, 135, 180, 225, 270, 315]:
#     #     scale = 2
#     #     low_rotate = cv2.getRotationMatrix2D((h // 2, w // 2), i, scale)
#     #     res = cv2.warpAffine(img_out, low_rotate, (w, h))
#     #     low_images.append(res)
#     #     high_rotate = cv2.getRotationMatrix2D((h, w), i, scale)
#     #     high_res = cv2.warpAffine(img2, high_rotate, (w * 2, h * 2))
#     #     high_images.append(high_res)
#     low_images.append(cv2.flip(img_out, 1))
#     high_images.append(cv2.flip(img2, 1))

#     for low_img, high_img in zip(low_images, high_images):
#         writer.write({
#             "low": (low_img.tobytes(), "byte"),
#             "high": (high_img.tobytes(), "byte"),
#             "h": (h, "int"),
#             "w": (w, "int"),
#         })
#         cnt += 1

# NOTE: code for unparied sr_x2, writer should be "train_x2.tfrecord"
# img_out is low image, img2 is high image
# for file_name in os.listdir(low_path):
#     low_image_path = os.path.join(low_path, file_name)
#     high_image_path = os.path.join(high_path, file_name)

#     img1 = cv2.imread(low_image_path, 0)
#     img2 = cv2.imread(high_image_path, 0)

#     low_images = []
#     high_images = []
#     h, w = img1.shape
#     low_images.append(img1)
#     high_images.append(img2)
#     # for i in [45, 90, 135, 180, 225, 270, 315]:
#     #     scale = 2
#     #     low_rotate = cv2.getRotationMatrix2D((h // 2, w // 2), i, scale)
#     #     res = cv2.warpAffine(img_out, low_rotate, (w, h))
#     #     low_images.append(res)
#     #     high_rotate = cv2.getRotationMatrix2D((h, w), i, scale)
#     #     high_res = cv2.warpAffine(img2, high_rotate, (w * 2, h * 2))
#     #     high_images.append(high_res)
#     low_images.append(cv2.flip(img1, 1))
#     high_images.append(cv2.flip(img2, 1))

#     for low_img, high_img in zip(low_images, high_images):
#         writer.write({
#             "low": (low_img.tobytes(), "byte"),
#             "high": (high_img.tobytes(), "byte"),
#             "h": (h, "int"),
#             "w": (w, "int"),
#         })
#         cnt += 1

writer.close()
print(cnt)
