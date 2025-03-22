import torch
import numpy as np
import cv2
from utils.plot import *


def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0].append([y, x])


def visualize_image(name, img, scale=1.0, write_image=[True, True], gamma=1.0, show_window=True):
    h, w, c = img.shape
    img_np = img

    if torch.is_tensor(img):
        img_np = img.detach().cpu().numpy()

    img_np = img_np ** gamma

    if show_window:
        cv2.imshow(name, cv2.resize(img_np[:, :, ::-1], dsize=(0, 0), fx=scale, fy=scale))
        cv2.waitKey(0)

    if c > 3:
        if write_image[0]: 
            cv2.imwrite(name + '_0.exr', img_np[:, :, :3][:, :, ::-1])
            cv2.imwrite(name + '_1.exr', img_np[:, :, 3:6][:, :, ::-1])
        if write_image[1]: 
            cv2.imwrite(name + '_0.png', img_np[:, :, :3][:, :, ::-1] * 255.0)
            cv2.imwrite(name + '_1.png', img_np[:, :, 3:6][:, :, ::-1] * 255.0)
    else:
        if write_image[0]: cv2.imwrite(name + '.exr', img_np[:, :, ::-1])
        if write_image[1]: cv2.imwrite(name + '.png', img_np[:, :, ::-1] * 255.0)


def visualize_images(name, img1, img2, is_vertical=True, scale=1.0, write_image=[True, True], gamma=1.0, show_window=True):
    if len(img1.shape) == 2:
        img1 = img1.unsqueeze(-1)
    if len(img2.shape) == 2:
        img2 = img2.unsqueeze(-1)

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    if c1 == 1:
        img1 = img1.repeat(1, 1, 3)
    if c2 == 1:
        img2 = img2.repeat(1, 1, 3)

    h = max(h1, h2)
    w = max(w1, w2)

    if is_vertical:
        img = torch.zeros([h * 2, w, 3])
        img[:h1, :w1, :] = img1[:, :, :3].clone()
        img[h:h+h2, :w2, :] = img2[:, :, :3].clone()
        visualize_image(name, img, scale, write_image, gamma, show_window)

    else:
        img = torch.zeros([h, w * 2, 3])
        img[:h1, :w1, :] = img1[:, :, :3].clone()
        img[:h2, w:w+w2, :] = img2[:, :, :3].clone()
        visualize_image(name, img, scale, write_image, gamma, show_window)


def save_imgs(name, img, scale=1.0, write_image=[True, True], gamma=1.0):
    if len(img.shape) == 2:
        img = img.unsqueeze(-1)

    img_np = img
    img_np = img_np ** gamma

    if torch.is_tensor(img):
        img_np = img.detach().cpu().numpy()

    if 'dop' in name:
        if write_image[0]: cv2.imwrite(name + '.exr', img_np)
        elif write_image[1]: 
            img_np = np.repeat(img_np, 3, axis=-1)
            img_np[:, :, :2] = 0.0
            cv2.imwrite(name + '.png', img_np * 255.0)
    elif 'aolp' in name:
        if write_image[0]: cv2.imwrite(name + '.exr', img_np)
        elif write_image[1]: cv2.imwrite(name + '.png', img_np[:, :, ::-1] * 255.0)



def visualize_plotting(name, v_image, p_image, sample_p, patch_size, scale=1.0):
    point = list()

    if torch.is_tensor(sample_p):
        sample_p = sample_p.detach().numpy()

    key = 0
    visualize_image(name, v_image, scale)
    cv2.setMouseCallback(name, mouse, [point])

    while key != 27:
        while len(point) < 1:
            key = cv2.waitKey(1)
            if key == 27:
                break
        if len(point) == 1:
            target_point = np.array(point).ravel() * scale - sample_p
            if 0 <= target_point[0] < patch_size and 0 <= target_point[1] < patch_size:
                target_point = target_point.astype(np.int64)
                print(target_point)
                idx = target_point[0] * patch_size + target_point[1]
                plotting_scatter_2d(p_image[:, idx, :3], 'DG', 'angle', 'DG', '{0}'.format(target_point), x=p_image[:, idx, 3])

            point.clear()
    cv2.destroyWindow(name)


def main():
    a = np.zeros([128, 128, 3])
    sample_p = np.array([10, 10])
    visualize_plotting('test', a, None, sample_p, 256)


if __name__ == '__main__':
    main()

