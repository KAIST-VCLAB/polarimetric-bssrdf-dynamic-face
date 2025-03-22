import os
os.umask(0)
import datetime
import json
import argparse
import shutil
import glob

from easydict import EasyDict as edict
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

from logger import Logger0
from dataset import PbrdfDatasetPatch, PbrdfDynamicDatasetBase
from renderer.patch import PbrdfRendererPatch
from renderer.batch import PbrdfRendererBatch
from loss import *



def init_worker(rank, world_size, args, logger):
    torch.manual_seed(args.general.seed)
    np.random.seed(args.general.seed)

    if args.general.ddp:
        assert world_size > 1, \
            '(args.general.ddp = true) not matches (world_size > 1)'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:7777',
            world_size=world_size,
            rank=rank)
        logger.print(f'DDP, Process {rank} initialized.')
    else:
        assert (rank == 0) and (world_size == 1), \
            '(args.general.ddp = false) not matches ((rank == 0) && (world_size == 1))'
        logger.print(f'Process initialized.')
        



def init_dataloader(world_size, args, logger, dataset):
    num_worker = 2
    if args.general.ddp:
        assert args.data.batch_size % world_size == 0, f'Batch size:{args.opt.batch_size} should be divisable by world size:{world_size}'
        batch_size_sub = int(args.data.batch_size / world_size)
        args.data.batch_size = batch_size_sub * world_size
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=batch_size_sub, shuffle=False, drop_last=True,
                                num_workers=num_worker, pin_memory=True, sampler=sampler)
        logger.print('--------'*4, write_to_log=True)
        logger.print(f'Batch size / 1 world  : {batch_size_sub}', write_to_log=True)
        logger.print(f'Total train epoch : {args.opt.max_epoch}', write_to_log=True)
        logger.print(f'Total train iteration : {dataset.__len__() * args.opt.max_epoch / args.data.batch_size}', write_to_log=True)
        logger.print('--------'*4, write_to_log=True)
    
    else:
        dataloader = DataLoader(dataset, batch_size=args.data.batch_size, shuffle=True, drop_last=False,
                                num_workers=num_worker, pin_memory=True)
        logger.print('--------'*4, write_to_log=True)
        logger.print(f'Total train epoch : {args.opt.max_epoch}', write_to_log=True)
        logger.print(f'Total train iteration : {dataset.__len__() * args.opt.max_epoch / args.data.batch_size}', write_to_log=True)
        logger.print('--------'*4, write_to_log=True)

    return dataloader

def opt(rank, world_size, args:edict):
    assert not args.general.test
    assert args.general.out_dir is not None
    logger = Logger0(args, rank, world_size, f'{args.general.out_dir}/run.log')
    
    init_worker(rank, world_size, args, logger)
    torch.cuda.set_device(rank)

    if args.opt.stage == 1:
        dataset = PbrdfDatasetPatch(args)
        renderer = PbrdfRendererPatch(args).cuda(rank)
        loss_function = pbrdf_loss
    elif args.opt.stage == 6:
        dataset = PbrdfDynamicDatasetBase(args)
        renderer = PbrdfRendererBatch(args, dataset).cuda(rank)
        loss_function = pbrdf_dynamic_loss


    dataloader = init_dataloader(world_size, args, logger, dataset)
    if args.general.ddp:
        renderer = DDP(renderer, device_ids=[rank])
    renderer.load_ckpt(args)
            

    optimizer = optim.RMSprop(renderer.parameters(), lr=args.opt.lr)

    if os.path.exists(f'{args.general.out_dir}/tb{rank}'):
        shutil.rmtree(f'{args.general.out_dir}/tb{rank}')
    logger.init_tensorboard(f'{args.general.out_dir}/tb{rank}')
    if args.general.ddp: dist.barrier()

    it = 0
    for epoch in range(args.opt.max_epoch):
        for batch in dataloader:
            optimizer.zero_grad()

            for k, v in batch.items(): batch[k] = v.cuda(rank, non_blocking=True)

            predictions = renderer(batch)
            if predictions.mask_all.sum() == 0:
                continue
            loss, loss_items = loss_function(predictions, batch, args.loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(renderer.parameters(), args.opt.max_grad_norm)
            optimizer.step()

            renderer.clamp_paramters(args.clamp)

            it += 1

            if args.general.ddp:
                for k, v in loss_items.items():
                    dist.all_reduce(v)
                    v /= args.data.batch_size

            if it % args.opt.LOG_FREQ_ITER == 0:
                logger.add_scalar('params/lr', optimizer.param_groups[0]['lr'], it)
                logger.train_log(it, loss_items)
            if it % args.opt.VIS_FREQ_ITER == 0:
                normal_map = None
                renderer.to_render()
                vertex_map, normal_map = renderer.compute_vertex_normal_map(dataset.I_vertex, dataset.I_normal)
                if args.opt.stage == 6:
                    rho_d = renderer.rho_d[renderer.visualize_idx].detach().clone().cpu()
                    logger.train_vis_d(it, normal_map, rho_d)
                else:
                    textures = renderer.get_textures()
                    logger.train_vis(it, normal_map, textures)
                

    logger.print('--------'*4, write_to_log=True)
    logger.save_ckpt(renderer, f'{args.general.out_dir}/ckpt/s{args.opt.stage}_last.pt')
    
    if args.general.ddp:
        dist.barrier()
        dist.destroy_process_group()


def make_train_result_dir(dir_root, out_name):
    if out_name is None:
        dir_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print(f'No output directory name given. Use current time: {dir_name}')
    else:
        print(f'Output directory: {out_name}')
        dir_name = out_name
    dirname = f'{dir_root}/{dir_name}'
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(f'{dirname}/ckpt', exist_ok=True)
    return dirname


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', type=str, default='config/hhha_local.json')
    parser.add_argument('-module_name', type=str, default='')
    parser.add_argument('-root_dir_path', type=str, default='')
    parser.add_argument('-participants_name', type=str, default='')
    parser.add_argument('-static_module_name', type=str, default='')
    parser.add_argument('-num_frames', type=int, default=-1)
    _args = parser.parse_args()
    with open(_args.config_path, 'r') as f:
        cfg = json.load(f)
    args = edict(cfg)
    
    print('################################################################')
    if not _args.root_dir_path == '':
        args.general.root_dir_path = _args.root_dir_path

    if not _args.participants_name == '':
        args.general.participants_name = _args.participants_name

    if not _args.module_name == '':
        args.general.module_name = _args.module_name

    if not _args.static_module_name == '':
        args.data.static_module_name = _args.static_module_name
        
    print(f'pBRDF {args.general.participants_name}/{args.general.module_name} for stage {args.opt.stage} is start.')
    
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

    if args.opt.stage == 6:
        print('Generate dynamic pbrdf dataset from static pbrdf results')
        args.data.static_path = args.data.path.replace(args.general.module_name, args.data.static_module_name)
        input_polar_textures = glob.glob(f'{args.data.static_path}/pbrdf/results/s1/tex*.exr')
        assert len(input_polar_textures) > 0, 'Generate static result using test method in pBRDF. Terminate.'
            
        for name in input_polar_textures:
            out_polar_texture = name.replace(f'{args.data.static_path}/pbrdf/results/s1', f'{args.data.path}/texture')
            shutil.copy(name, out_polar_texture)
        shutil.copy(f'{args.data.static_path}/texture/mask.exr', f'{args.data.path}/texture/mask.exr')

    if _args.num_frames >= 0:
        args.data.num_frames = _args.num_frames
    if args.data.num_frames == 0:
        args.data.num_frames = len(glob.glob(f'{args.data.path}/texture/0/exr/visible_img1*.exr'))

    args.general.out_dir = make_train_result_dir(args.general.out_dir_root, args.general.out_name)
    shutil.copy(_args.config_path, f'{args.general.out_dir}/config.json')

    assert torch.cuda.is_available(), 'GPU required. Terminate.'
    world_size = torch.cuda.device_count()
    print(f'Number of GPU = {world_size}')

    if args.general.ddp:
        mp.spawn(opt, nprocs=world_size, args=(world_size, cfg))
    else:
        opt(0, 1, args)
    
    print('################################################################')
    print(' Complete')
    print('################################################################')
