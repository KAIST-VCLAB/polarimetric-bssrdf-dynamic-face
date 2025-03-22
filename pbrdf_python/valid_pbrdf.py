import os
os.umask(0)
import json
import argparse

from easydict import EasyDict as edict
from joblib import Parallel, delayed
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader

from dataset import PbrdfDatasetValidation, PbrdfDynamicDatasetTest
from renderer.patch import PbrdfRendererPatch
from renderer.batch import PbrdfRendererBatch
from utils.viewer import visualize_images, save_imgs
from utils.polar_util import *


@torch.no_grad()
def test(args:edict):
    assert args.general.test
    
    torch.manual_seed(args.general.seed)
    np.random.seed(args.general.seed)

    torch.cuda.set_device(0)

    dataset = PbrdfDatasetValidation(args)

    res_dir = f'{args.general.out_dir_root}'
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(f'{res_dir}/png', exist_ok=True)
    os.makedirs(f'{res_dir}/exr', exist_ok=True)

    num_worker = 2
    dataloader = DataLoader(dataset, batch_size=args.data.batch_size, shuffle=False, drop_last=False,
                            num_workers=num_worker, pin_memory=True)
    
    for batch in dataloader:
        for k, v in batch.items(): batch[k] = v.cuda()

        def t2a(x:torch.Tensor): return x.cpu().numpy()
        index = t2a(batch.index)

        cam_frame_index = np.mod(index, dataset.num_frames)

        dop = compute_dop(batch, rear_str='_cam')
        aolp = compute_aolp(batch, rear_str='_cam')
        color_aolp = applyColorToAoLP(aolp.cpu().numpy())

        Parallel(n_jobs=-1, backend="threading")(
            delayed(save_imgs)(f'{res_dir}/png/aolp_cam{batch.cam_idx[i]}_{dataset.select_index[cam_frame_index[i]] + 1:04d}',
                                color_aolp[i], scale=1.0, gamma=1.0, write_image=[False, True])
            for i in range(dop.shape[0]))

        Parallel(n_jobs=-1, backend="threading")(
            delayed(save_imgs)(f'{res_dir}/png/dop_cam{batch.cam_idx[i]}_{dataset.select_index[cam_frame_index[i]] + 1:04d}',
                                dop[i], scale=1.0, gamma=1.0, write_image=[False, True])
            for i in range(dop.shape[0]))

        Parallel(n_jobs=-1, backend="threading")(
            delayed(save_imgs)(f'{res_dir}/exr/aolp_cam{batch.cam_idx[i]}_{dataset.select_index[cam_frame_index[i]] + 1:04d}',
                                aolp[i], scale=1.0, gamma=1.0, write_image=[True, False])
            for i in range(dop.shape[0]))

        Parallel(n_jobs=-1, backend="threading")(
            delayed(save_imgs)(f'{res_dir}/exr/dop_cam{batch.cam_idx[i]}_{dataset.select_index[cam_frame_index[i]] + 1:04d}',
                                dop[i], scale=1.0, gamma=1.0, write_image=[True, False])
            for i in range(dop.shape[0]))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, default='config/test_valid.json')
    parser.add_argument('--module_name', '-m', type=str, default='')
    parser.add_argument('--participants_name', '-p', type=str, default='')
    parser.add_argument('--dir_root', '-d',type=str, default='')
    parser.add_argument('--out_dir_name', '-o', type=str, default='')
    parser.add_argument('--num_frames', '-n', type=int, default=-1)
    parser.add_argument('--input_texture_root', '-i', type=str, default='')
    _args = parser.parse_args()
    with open(_args.config_path, 'r') as f:
        cfg = json.load(f)
    args = edict(cfg)

    if not _args.participants_name == '':
        args.general.participants_name = _args.participants_name

    if not _args.module_name == '':
        args.general.module_name = _args.module_name

    if not _args.dir_root == '':
        args.general.dir_root = _args.dir_root

    if not _args.out_dir_name == '':
        args.general.out_dir_name = _args.out_dir_name

    if not _args.input_texture_root == '':
        args.general.input_texture_root = _args.input_texture_root

    args.general.out_dir_root = args.general.out_dir_root.replace('(PARTICIPANTS_NAME)', args.general.participants_name)
    args.data.path = args.data.path.replace('(PARTICIPANTS_NAME)', args.general.participants_name)
    args.data.input_img_path = args.data.input_img_path.replace('(PARTICIPANTS_NAME)', args.general.participants_name)

    args.general.out_dir_root = args.general.out_dir_root.replace('(MODULE_NAME)', args.general.module_name)
    args.data.path = args.data.path.replace('(MODULE_NAME)', args.general.module_name)
    args.data.input_img_path = args.data.input_img_path.replace('(MODULE_NAME)', args.general.module_name)
    
    args.general.out_dir_root = args.general.out_dir_root.replace('(DIR_ROOT)', args.general.dir_root)
    args.data.path = args.data.path.replace('(DIR_ROOT)', args.general.dir_root)
    args.data.input_img_path = args.data.input_img_path.replace('(INPUT_DIR_ROOT)', args.general.input_texture_root)
    
    args.general.out_dir_root = args.general.out_dir_root.replace('(OUT_DIR_NAME)', args.general.out_dir_name)
    args.data.path = args.data.path.replace('(OUT_DIR_NAME)', args.general.out_dir_name)
    args.data.input_img_path = args.data.input_img_path.replace('(OUT_DIR_NAME)', args.general.out_dir_name)

    total_frame = len(glob.glob(f'{args.data.path}/texture/0/exr/visible_img1*.exr'))
    if _args.num_frames >= 0:
        args.data.num_frames = _args.num_frames
    if args.data.num_frames == 0:
        args.data.num_frames = total_frame

    print('################################################################')
    start_idx = args.data.start_idx
    end_idx = min(args.data.num_frames + start_idx, total_frame)
    print(f'Frame {start_idx:03d} to {end_idx:03d} with batch size {args.data.batch_size}')
    for i in range(start_idx, end_idx, args.data.batch_size):
        args.data.start_idx = i
        if i + args.data.batch_size > end_idx:
            args.data.batch_size = args.data.num_frames + start_idx - i
        test(args)
    
    print('################################################################')
    print(' Complete')
    print('################################################################')
