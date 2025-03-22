import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.umask(0)

from joblib import Parallel, delayed
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import torch
import glob
import pathlib


def load_images_parallel(path, h, w, white_balance, num_cams=0, num_frames=0, selected_index=None):
    filenames = os.listdir(path)
    filenames.sort()
    img_filenames = np.array(filenames[:len(filenames) // 2])
    visibility_filenames = np.array(filenames[len(filenames) // 2:])

    if num_cams != 0 and num_frames != 0:
        cam_idx = [name.split('_')[0] for name in img_filenames]
        num_full_cams = len(set(cam_idx))
        num_full_frames = len(img_filenames) // num_full_cams

        if selected_index is None:
            select_idx = np.linspace(0, num_full_frames - 1, num_frames).astype(np.int64)
            print(select_idx)
        else:
            select_idx = selected_index

        tmp_img_name = []
        tmp_visibility_name = []
        for cam_idx in range(num_cams):
            tmp_img_name.extend(img_filenames[(cam_idx * num_full_frames + select_idx)])
            tmp_visibility_name.extend(visibility_filenames[(cam_idx * num_full_frames + select_idx)])
        img_filenames = tmp_img_name
        visibility_filenames = tmp_visibility_name


    N = len(img_filenames)
    imgs = torch.zeros([N, h, w, 3])
    visibilities = torch.zeros([N, h, w, 1])

    def load_image(idx, img_name, visibility_name):
        img = cv2.imread(os.path.join(path, img_name), -1)[:, :, ::-1].copy()
        imgs[idx, :, :, :] = torch.Tensor(img)
        visibility = cv2.imread(os.path.join(path, visibility_name), -1)[:, :, ::-1].copy()
        visibilities[idx, :, :, :] = torch.Tensor(visibility).mean(dim=-1, keepdim=True)

    Parallel(n_jobs=16, backend="threading")(
        delayed(load_image)(idx, img_name, visibility_name)
        for idx, [img_name, visibility_name] in enumerate(zip(img_filenames, visibility_filenames)))

    print('loading {0} done'.format(path))

    cam1_white_balance = white_balance[0, 1:][None, None, None, :]
    cam2_white_balance = white_balance[1, 1:][None, None, None, :]

    imgs[:num_frames] = imgs[:num_frames] * cam1_white_balance
    imgs[num_frames:2*num_frames] = imgs[num_frames:2*num_frames] * cam2_white_balance

    return imgs, visibilities, num_cams, num_frames, select_idx


def load_cam_images_parallel(path, angle, start_idx, white_balance, d65_mat, num_cams=0, num_frames=0, selected_index=None):
    img1_filenames = np.array(sorted(glob.glob(f'{path}/cam1/undistorted/{angle}/*.png')))
    img2_filenames = np.array(sorted(glob.glob(f'{path}/cam2/undistorted/{angle}/*.png')))
    img3_filenames = np.array(sorted(glob.glob(f'{path}/cam3/undistorted/{angle}/*.png')))
    img_filenames = np.concatenate([img1_filenames, img2_filenames, img3_filenames], axis=-1)

    if num_cams != 0 and num_frames != 0:
        num_full_frames = len(img2_filenames)

        if selected_index is None:
            select_idx = np.linspace(start_idx, start_idx + num_frames - 1, num_frames).astype(np.int64)
        else:
            select_idx = selected_index

        tmp_img_name = []
        for cam_idx in range(num_cams):
            tmp_img_name.extend(img_filenames[(cam_idx * num_full_frames + select_idx)])
        img_filenames = tmp_img_name


    tmp_img = cv2.imread(img_filenames[0], -1)
    h, w, _ = tmp_img.shape

    N = len(img_filenames)
    imgs = torch.zeros([N, h, w, 3])

    def load_image(idx, img_name):
        img = cv2.imread(os.path.join(path, img_name), -1)[:, :, ::-1].copy()
        imgs[idx, :, :, :] = (torch.Tensor(img.astype(np.float32)) / 65535.0) ** 2.2

    Parallel(n_jobs=16, backend="threading")(
        delayed(load_image)(idx, img_name)
        for idx, img_name in enumerate(img_filenames))

    print(f'loading {path}/{angle} frame {start_idx} to {start_idx + num_frames - 1} done.')

    cam1_white_balance = white_balance[0, 1:][None, None, None, :]
    cam2_white_balance = white_balance[1, 1:][None, None, None, :]
    cam3_white_balance = white_balance[2, 1:][None, None, None, :]

    imgs[:num_frames] = imgs[:num_frames] * cam1_white_balance
    imgs[num_frames:2*num_frames] = imgs[num_frames:2*num_frames] * cam2_white_balance

    d65_mat3_inv = torch.linalg.inv(d65_mat[2])
    imgs3 = imgs[2*num_frames:3*num_frames]
    imgs3 = torch.einsum('lk,nijk->nijl', d65_mat3_inv, imgs3)
    imgs[2*num_frames:3*num_frames] = imgs3 * cam3_white_balance

    return imgs, num_cams, num_frames, select_idx


def load_images_sequence_parallel(path, h, w, white_balance, start, end, num_cams, selected_index=None):
    filenames = os.listdir(path)
    filenames.sort()
    img_filenames = np.array(filenames[:len(filenames) // 2])
    visibility_filenames = np.array(filenames[len(filenames) // 2:])

    cam_idx = [name.split('_')[0] for name in img_filenames]
    num_full_cams = len(set(cam_idx))
    num_full_frames = len(img_filenames) // num_full_cams

    num_frames = end - start
    if selected_index is None:
        select_idx = np.arange(start, end).astype(np.int64)
    else:
        select_idx = selected_index

    tmp_img_name = []
    tmp_visibility_name = []
    for cam_idx in range(num_cams):
        tmp_img_name.extend(img_filenames[(cam_idx * num_full_frames + select_idx)])
        tmp_visibility_name.extend(visibility_filenames[(cam_idx * num_full_frames + select_idx)])
    img_filenames = tmp_img_name
    visibility_filenames = tmp_visibility_name


    N = len(img_filenames)
    imgs = torch.zeros([N, h, w, 3])
    visibilities = torch.zeros([N, h, w, 1])

    def load_image(idx, img_name, visibility_name):
        img = cv2.imread(os.path.join(path, img_name), -1)[:, :, ::-1].copy()
        imgs[idx, :, :, :] = torch.Tensor(img)
        visibility = cv2.imread(os.path.join(path, visibility_name), -1)[:, :, ::-1].copy()
        visibilities[idx, :, :, :] = torch.Tensor(visibility).mean(dim=-1, keepdim=True)

    Parallel(n_jobs=16, backend="threading")(
        delayed(load_image)(idx, img_name, visibility_name)
        for idx, [img_name, visibility_name] in enumerate(zip(img_filenames, visibility_filenames)))

    print('loading {0} done'.format(path))

    cam1_white_balance = white_balance[0, 1:][None, None, None, :]
    cam2_white_balance = white_balance[1, 1:][None, None, None, :]

    imgs[:num_frames] = imgs[:num_frames] * cam1_white_balance
    imgs[num_frames:2*num_frames] = imgs[num_frames:2*num_frames] * cam2_white_balance

    return imgs, visibilities, num_cams, num_frames, select_idx



def load_images_glob_parallel(path, name, h, w, num_frames, selected_index):
    filenames_full = np.array(sorted(pathlib.Path(path).glob(f'{name}*.exr')))
    filenames = filenames_full[selected_index]

    imgs = torch.zeros([num_frames, h, w, 3])

    def load_image(idx, img_name):
        img = cv2.imread(str(img_name), -1)[:, :, ::-1].copy()
        imgs[idx, :, :, :] = torch.Tensor(img)

    Parallel(n_jobs=16, backend="threading")(
        delayed(load_image)(idx, img_name)
        for idx, img_name in enumerate(filenames))

    print(f'loading {path}: {name} done')

    return imgs


def load_ref_images_parallel(path, h, w, white_balance, d65_mat_inv, num_cams=0, num_frames=0, selected_index=None):
    img_filenames = np.array(sorted(glob.glob(f'{path}/img3_*.exr')))
    visibility_filenames = np.array(sorted(glob.glob(f'{path}/visible_img3_*.exr')))

    num_full_frames = len(img_filenames)

    selected_index = None

    if selected_index is None:
        select_idx = np.arange(0, num_frames).astype(np.int64)
    else:
        select_idx = selected_index
    img_filenames = img_filenames[select_idx]
    visibility_filenames = visibility_filenames[select_idx]


    imgs = torch.zeros([num_frames, h, w, 3])
    visibilities = torch.zeros([num_frames, h, w, 1])

    def load_image(idx, img_name, visibility_name):
        img = cv2.imread(os.path.join(path, img_name), -1)[:, :, ::-1].copy()
        img = np.einsum('lk,ijk->ijl', d65_mat_inv, img)
        imgs[idx, :, :, :] = torch.Tensor(img)
        visibility = cv2.imread(os.path.join(path, visibility_name), -1)[:, :, ::-1].copy()
        visibilities[idx, :, :, :] = torch.Tensor(visibility).mean(dim=-1, keepdim=True)

    Parallel(n_jobs=16, backend="threading")(
        delayed(load_image)(idx, img_name, visibility_name)
        for idx, [img_name, visibility_name] in enumerate(zip(img_filenames, visibility_filenames)))

    print('loading {0} done'.format(path))

    cam3_white_balance = white_balance[2, 1:][None, None, None, :]
    imgs = imgs * cam3_white_balance

    return imgs, visibilities, num_frames, select_idx


def get_camera_params(path, num_cams):
    f = open(os.path.join(path, 'images.txt'))
    world_to_cam = torch.eye(4).unsqueeze(0).repeat_interleave(num_cams, dim=0)
    cam_to_world = torch.eye(4).unsqueeze(0).repeat_interleave(num_cams, dim=0)
    for line in f:
        lst = line.strip().split()
        if len(lst) > 0:
            if not lst[0].isdigit(): continue
            idx = int(lst[0]) - 1
            if idx >= num_cams:
                continue
            q = np.array(list(map(float, lst[1:5]))) 
            t = torch.Tensor(list(map(float, lst[5:8])))
            world_to_cam[idx, :3, :3] = torch.Tensor(R.from_quat(np.roll(q, -1)).as_matrix())
            world_to_cam[idx, :3, 3] = t[:3]
            cam_to_world[idx, :, :] = torch.Tensor(np.linalg.inv(world_to_cam[idx, :, :]))
    f.close()

    f = open(os.path.join(path, 'cameras.txt'))
    intrinsic = torch.zeros([num_cams, 3, 3])
    for line in f:
        lst = line.strip().split()
        if len(lst) <= 0: continue
        if not lst[0].isdigit(): continue
        idx = int(lst[0]) - 1
        if idx >= num_cams:
            continue
        tup = tuple(map(float, lst[4:]))
        fx, fy, cx, cy, k1, k2, p1, p2 = tup
        intrinsic[idx, :, :] = torch.Tensor([[fx, 0, cx],
                                            [0, fy, cy],
                                            [0, 0, 1]])
    f.close()

    return intrinsic, world_to_cam, cam_to_world