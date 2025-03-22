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

from dataset import PbrdfDatasetBase, PbrdfDynamicDatasetTest
from renderer.patch import PbrdfRendererPatch
from renderer.batch import PbrdfRendererBatch
from utils.viewer import visualize_images, visualize_image


@torch.no_grad()
def test(args:edict):
    assert args.general.test
    
    torch.manual_seed(args.general.seed)
    np.random.seed(args.general.seed)


    torch.cuda.set_device(0)

    if args.opt.stage == 1:
        dataset = PbrdfDatasetBase(args)
        renderer = PbrdfRendererPatch(args).cuda()
        res_dir = f'{args.general.out_dir_root}/results/s1'
    elif args.opt.stage == 6:
        dataset = PbrdfDynamicDatasetTest(args)
        renderer = PbrdfRendererBatch(args, dataset).cuda()
        res_dir = f'{args.general.out_dir_root}/results/s6'

    os.makedirs(res_dir, exist_ok=True)

    num_worker = 2
    dataloader = DataLoader(dataset, batch_size=args.data.batch_size, shuffle=False, drop_last=False,
                            num_workers=num_worker, pin_memory=True)

    renderer.load_ckpt(args)
    
    if args.opt.stage != 6:
        renderer.to_render()
        textures = renderer.get_textures()
        vertex_map, normal_map = renderer.compute_vertex_normal_map(dataset.I_vertex, dataset.I_normal)
        visualize_image(f'{res_dir}/tex_vertmap', vertex_map[:, :, :3], write_image=[True, False], show_window=False)
        visualize_image(f'{res_dir}/tex_normalmap', normal_map[:, :, :3], write_image=[True, False], show_window=False)

        for k, v in textures.items():
            visualize_image(f'{res_dir}/{k}', textures[f'{k}'], write_image=[True, False], show_window=False)

        rho_d = renderer.update_rho_d(args, vertex_map, normal_map, dataset.dataset.I_diffuse, dataset.dataset.I_mask, 
                                      dataset.dataset)
        rho_d_ref = renderer.update_rho_d_ref(args, vertex_map, normal_map, dataset.I_diffuse_ref, dataset.I_mask_ref, 
                                            dataset.dataset)
        visualize_image(f'{res_dir}/tex_rho_d_2', rho_d_ref, write_image=[True, False], show_window=False)
        visualize_image(f'{res_dir}/tex_rho_d', rho_d, write_image=[True, False], show_window=False)

    else:
        textures = renderer.get_textures()
        for k, v in textures.items():
            Parallel(n_jobs=16, backend="threading")(
                delayed(visualize_image)(f'{res_dir}/{k}_{i + 1:04d}', v[i], write_image=[True, False], show_window=False)
            for i in range(v.shape[0]))
        for batch in dataloader:
            for k, v in batch.items(): batch[k] = v.cuda()
            vertex_map, normal_map = renderer.compute_vertex_normal_maps(batch.I_vertex, batch.I_normal, batch.frame_idx)
            Parallel(n_jobs=16, backend="threading")(
                    delayed(visualize_image)(f'{res_dir}/tex_vertmap_{batch.frame_idx[i] + 1:04d}', vertex_map[i, :, :, :3], write_image=[True, False], show_window=False)
                for i in range(vertex_map.shape[0]))
            Parallel(n_jobs=16, backend="threading")(
                    delayed(visualize_image)(f'{res_dir}/tex_normalmap_{batch.frame_idx[i] + 1:04d}', normal_map[i, :, :, :3], write_image=[True, False], show_window=False)
                for i in range(normal_map.shape[0]))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', type=str, default='config/hhha_local_test.json')
    parser.add_argument('-root_dir_path', type=str, default='')
    parser.add_argument('-module_name', type=str, default='')
    parser.add_argument('-participants_name', type=str, default='')
    parser.add_argument('-num_frames', type=int, default=-1)
    _args = parser.parse_args()
    with open(_args.config_path, 'r') as f:
        cfg = json.load(f)
    args = edict(cfg)

    if not _args.root_dir_path == '':
        args.general.root_dir_path = _args.root_dir_path

    if not _args.participants_name == '':
        args.general.participants_name = _args.participants_name

    if not _args.module_name == '':
        args.general.module_name = _args.module_name
    
    args.general.out_dir_root = args.general.out_dir_root.replace('(ROOT_DIR_PATH)', args.general.root_dir_path)
    args.data.path = args.data.path.replace('(ROOT_DIR_PATH)', args.general.root_dir_path)
    for i in range(len(args.opt.ckpt_path)):
        args.opt.ckpt_path[i] = args.opt.ckpt_path[i].replace('(ROOT_DIR_PATH)', args.general.root_dir_path)

    args.general.out_dir_root = args.general.out_dir_root.replace('(PARTICIPANTS_NAME)', args.general.participants_name)
    args.data.path = args.data.path.replace('(PARTICIPANTS_NAME)', args.general.participants_name)
    for i in range(len(args.opt.ckpt_path)):
        args.opt.ckpt_path[i] = args.opt.ckpt_path[i].replace('(PARTICIPANTS_NAME)', args.general.participants_name)

    args.general.out_dir_root = args.general.out_dir_root.replace('(MODULE_NAME)', args.general.module_name)
    args.data.path = args.data.path.replace('(MODULE_NAME)', args.general.module_name)
    for i in range(len(args.opt.ckpt_path)):
        args.opt.ckpt_path[i] = args.opt.ckpt_path[i].replace('(MODULE_NAME)', args.general.module_name)

    if _args.num_frames >= 0:
        args.data.num_frames = _args.num_frames
    if args.data.num_frames == 0:
        args.data.num_frames = len(glob.glob(f'{args.data.path}/texture/0/exr/visible_img1*.exr'))

    if args.opt.stage == 6:
        if _args.num_frames >= 0:
            args.data.num_frames = _args.num_frames
        if args.data.num_frames == 0:
            args.data.num_frames = len(glob.glob(f'{args.data.path}/texture/0/exr/visible_img1*.exr'))

    print('################################################################')
    test(args)
    
    print('################################################################')
    print(' Complete')
    print('################################################################')
