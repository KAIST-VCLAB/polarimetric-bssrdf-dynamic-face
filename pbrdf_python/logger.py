from easydict import EasyDict as edict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter



class Logger0:
    def __init__(self,
                 args:edict,
                 rank:int,
                 world_size:int,
                 log_out_path=None):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.tb = None
        if (rank == 0) and bool(log_out_path):
            self.log_out = open(log_out_path, 'w')
            self.log_out.write(f'World size (# GPU): {self.world_size}\n')
            self.log_out.flush()
    

    def __del__(self):
        if (self.rank == 0):
            self.log_out.close()
            if self.tb is not None:
                self.tb.close()
        self.print('Logger destructed')


    def run_only_rank0(function):
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.rank == 0: function(*args, **kwargs)
        return wrapper


    @run_only_rank0
    def print(self, obj, write_to_log=False):
        print(obj)
        if write_to_log:
            self.log_out.write(f'{obj}\n')
            self.log_out.flush()

    @run_only_rank0
    def init_tensorboard(self, log_dir): self.tb = SummaryWriter(log_dir=log_dir)
    
    @run_only_rank0
    def add_scalar(self, tag, scalar_value, global_step):
        self.tb.add_scalar(tag, scalar_value, global_step)



    @run_only_rank0
    def save_ckpt(self, net, ckpt_path, suffix=""):
        if self.args.general.ddp:
            torch.save(net.module.state_dict(), ckpt_path)
        else:
            torch.save(net.state_dict(), ckpt_path)
        print(f'# Checkpoint saved! {suffix}')


    @run_only_rank0
    def train_log(self, iteration, loss_dict):
        for k,v in loss_dict.items():
            if k == 'all':
                self.tb.add_scalar(f'loss_all', v.item(), global_step=iteration)
            else:
                self.tb.add_scalar(f'loss/{k}', v.item(), global_step=iteration)
            
    @run_only_rank0
    def train_vis(self, iteration, normal_map, textures):        
        if normal_map is not None:
            normal_map_ = (normal_map[:, :, :3].detach().clone().cpu() + 1) * 0.5 # H W C
            self.tb.add_image('normal_map', normal_map_, iteration, dataformats='HWC')
        alpha_s_map_ = textures.tex_alpha_s.cpu()
        self.tb.add_image('/textures/alpha_s_map', alpha_s_map_, iteration, dataformats='HWC')
        alpha_ss_map_ = textures.tex_alpha_ss.cpu()
        self.tb.add_image('/textures/alpha_ss_map', alpha_ss_map_, iteration, dataformats='HWC')
        rho_d_map_ = textures.tex_rho_d.cpu() ** (1.0 / 2.2)
        self.tb.add_image('/textures/rho_d1_map', rho_d_map_[:, :, :3], iteration, dataformats='HWC')
        self.tb.add_image('/textures/rho_d2_map', rho_d_map_[:, :, 3:6], iteration, dataformats='HWC')


    @run_only_rank0
    def train_vis_d(self, iteration, normal_map, diffuse_map=None):        
        if normal_map is not None:
            normal_map_ = (normal_map[:, :, :3].detach().clone().cpu() + 1) * 0.5 # H W C
            self.tb.add_image('normal_map', normal_map_, iteration, dataformats='HWC')
            diffuse_map_1_ = (diffuse_map[:, :, :3].detach().clone().cpu()) ** (1.0 / 2.2) # H W C
            self.tb.add_image('diffuse_map/diffuse_map1', diffuse_map_1_, iteration, dataformats='HWC')
            diffuse_map_2_ = (diffuse_map[:, :, 3:6].detach().clone().cpu()) ** (1.0 / 2.2) # H W C
            self.tb.add_image('diffuse_map/diffuse_map2', diffuse_map_2_, iteration, dataformats='HWC')
    









if __name__=='__main__':
    logger0 = Logger0(0, 0)
    logger0.print('print on rank0')

    logger1 = Logger0(0, 1)
    logger1.print('print on rank1')