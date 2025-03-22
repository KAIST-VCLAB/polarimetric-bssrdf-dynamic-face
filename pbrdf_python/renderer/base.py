from easydict import EasyDict as edict
import torch
import torch.nn as nn
import numpy as np

from renderer.utils import *


class PbrdfRenderer(nn.Module):
    def __init__(self, args:edict) -> None:
        super().__init__()

        self.N = args.data.num_cams * args.data.num_frames
        self.w = args.data.w
        self.h = args.data.h
        self.c = args.data.c
        self.batch_size = args.data.batch_size
        self.num_cams = args.data.num_cams
        self.num_frames = args.data.num_frames
        self.patch_size = args.data.patch_size
        self.num_frame_patch = args.data.num_frame_patch

        self.threshold = args.threshold

        self.tex_refrac_idx = torch.zeros([args.data.h, args.data.w, 1])
        self.tex_rho_s = torch.zeros([args.data.h, args.data.w, 1])
        self.tex_rho_ss = torch.zeros([args.data.h, args.data.w, args.data.c])
        self.tex_alpha_s = torch.zeros([args.data.h, args.data.w, 1])
        self.tex_alpha_ss = torch.zeros([args.data.h, args.data.w, 1])
        self.tex_height = torch.zeros([args.data.h, args.data.w, 1])
        self.tex_rho_d = torch.zeros([args.data.h, args.data.w, args.data.c])

    
    def get_textures(self):
        tex = edict()

        tex.tex_refrac_idx = self.tex_refrac_idx.clone()
        tex.tex_rho_s = self.tex_rho_s.clone()
        tex.tex_rho_ss = self.tex_rho_ss.clone()
        tex.tex_alpha_s = self.tex_alpha_s.clone()
        tex.tex_alpha_ss = self.tex_alpha_ss.clone()
        tex.tex_height = self.tex_height.clone()
        tex.tex_rho_d = self.tex_rho_d.clone()

        return tex

    def to_render(self):
        pass
    
    def clamp_paramters(self, args):
        for k, v in self.named_parameters():
            if k in args.keys():
                v.data.clamp_(args[f'{k}'][0], args[f'{k}'][1])


    def _load_ckpt(self, args, ckpt_path, msg=''):
        device = next(self.parameters()).device
        ckpt = torch.load(ckpt_path, map_location=device)
        if args.general.ddp:
            ckpt = {f'module.{k}': v for k,v in ckpt.items()}
        self.load_state_dict(ckpt)
        print(f'# Checkpoint loaded! {msg}')

    
    @torch.no_grad()
    def compute_vertex_normal_map(self, I_vertex, I_normal):
        device = self.tex_height.device
        vertex = I_vertex.to(device)
        normal = I_normal.to(device)
        normalmap = compute_height_normal(vertex, normal, self.tex_height.detach().clone())
        vertmap = compute_height_vertex(vertex, normal, self.tex_height.detach().clone())

        one_tensor = torch.ones(vertmap.shape[:-1],
                                dtype=vertmap.dtype, device=vertmap.device)[..., None]
        zero_tensor = torch.zeros(normalmap.shape[:-1],
                                  dtype=normalmap.dtype, device=normalmap.device)[..., None]

        vert4map = torch.cat([vertmap, one_tensor], dim=-1)
        normal4map = torch.cat([normalmap, zero_tensor], dim=-1)

        return vert4map, normal4map
    

    @torch.no_grad()
    def update_rho_d_ref(self, args, vertex_map, normal_map, I_diffuse_ref, I_mask_ref, data):
        h = self.h
        w = self.w
        c = I_diffuse_ref.shape[-1]
        N = I_diffuse_ref.shape[0]
        
        I_diffuse_ref = I_diffuse_ref.cuda()
        ref_to_frame_mat = data.ref_to_frame_full.cuda()
        cam_pos = data.cam_to_world_ref[:, :3, 3].cuda() 
        I_mask_ref = I_mask_ref.cuda()

        vertmap = vertex_map.cuda()
        normalmap = normal_map.cuda()

        rho_d_ref = torch.zeros([h, w, c]).cuda()
        for y in range(0, self.h, self.patch_size):
            for x in range(0, self.w, self.patch_size):
                ey = y + self.patch_size
                ex = x + self.patch_size
                if ey > self.h: ey = self.h
                if ex > self.w: ex = self.w

                vert4map = vertmap[y:ey, x:ex, :]
                normal4map = normalmap[y:ey, x:ex, :]

                I_mask_ref_patch = I_mask_ref[:, y:ey, x:ex]
                if I_mask_ref_patch.sum() == 0: 
                    continue
                I_diffuse_ref_patch = I_diffuse_ref[:, y:ey, x:ex, :]
                refrac_idx_patch = self.tex_refrac_idx.unsqueeze(0)[:, y:ey, x:ex, :]

                n_t = (ref_to_frame_mat[:, None, None, :, :] *
                    normal4map[None, :, :, None, :]).sum(axis=-1)[:, :, :, :3]
                v_t = (ref_to_frame_mat[:, None, None, :, :] *
                    vert4map[None, :, :, None, :]).sum(axis=-1)[:, :, :, :3]

                n_t = F.normalize(n_t, dim=-1) 
                distance_t = torch.linalg.norm(cam_pos[:, None, None, :] - v_t, dim=-1, keepdim=True) / 1000.0
                attenuation = 1.0 / (distance_t * distance_t)
                v_dirs_t = F.normalize(cam_pos[:, None, None, :] - v_t, dim=-1) 
                l_dirs_t = v_dirs_t.clone() 

                ndotl = torch.clamp((n_t * l_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0)
                ndotv = torch.clamp((n_t * v_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0)

                _, _, Ts_i, Tp_i = compute_Fresnel(ndotl, 1.0, refrac_idx_patch, False)
                _, _, Ts_o, Tp_o = compute_Fresnel(ndotv, refrac_idx_patch, 1.0, True)
                Tpos_o = (Ts_o + Tp_o) / 2.0
                Tpos_i = (Ts_i + Tp_i) / 2.0
                TpoTpi = Tpos_o * Tpos_i

                shade_mask = ndotl > args.threshold.ndotl
                I_mask_ref_patch[~(shade_mask.squeeze())] = False

                rho_d_weights = ndotl.clone()
                rho_d_weights[~I_mask_ref_patch] = 0
                thres_mask = (rho_d_weights >= 0.72)
                rho_d_weights[~thres_mask] = 0

                bad_region = ~torch.any(thres_mask, dim=0, keepdim=True)
                bad_region = bad_region.repeat_interleave(N, dim=0)
                rho_d_weights[bad_region] = ndotl[bad_region]
                rho_d_weights[~I_mask_ref_patch] = 0

                rho_d_weights = rho_d_weights * rho_d_weights

                rendered_shade = torch.ones([N, ey-y, ex-x, 1], dtype=self.tex_rho_d.dtype, device=self.tex_rho_d.device)
                rho_d_image = torch.zeros([N, ey-y, ex-x, c], dtype=self.tex_rho_d.dtype, device=self.tex_rho_d.device)

                rendered_shade[I_mask_ref_patch] = (TpoTpi * ndotl * attenuation)[I_mask_ref_patch]
                rendered_shade = rendered_shade.repeat_interleave(c, dim=-1)
                rho_d_image[I_mask_ref_patch] = I_diffuse_ref_patch[I_mask_ref_patch] / rendered_shade[I_mask_ref_patch]

                sum_weights = torch.sum(rho_d_weights, dim=0)
                rho_d_ref_patch = torch.sum(rho_d_image * rho_d_weights, dim=0) / sum_weights
                rho_d_ref_patch[~(sum_weights.squeeze() > 0.025)] = 0
                rho_d_ref[y:ey, x:ex, :] = rho_d_ref_patch

        return rho_d_ref

    @torch.no_grad()
    def render(self, data:edict):
        self.to_render()
        h = self.h
        w = self.w
        N = data.I_diffuse.shape[0]
        
        ref_to_frame_mat = data.ref_to_frame
        cam_pos = data.cam_to_world[:, :3, 3] 
        db_vertex = data.I_vertex[0]
        db_normal = data.I_normal[0]
        mask = data.I_mask
        I_alpha = data.I_alpha
        I_spec = data.I_specular
        I_diffuse = data.I_diffuse

        rho_d = torch.zeros_like(I_diffuse)
        rho_ss = torch.zeros_like(I_spec)

        cam_idx_mask = (data.cam_idx == 0)
        rho_d[cam_idx_mask] = self.tex_rho_d[:, :, :3].unsqueeze(0).repeat_interleave(cam_idx_mask.sum(), dim=0)
        rho_ss[cam_idx_mask] = self.tex_rho_ss[:, :, :3].unsqueeze(0).repeat_interleave(cam_idx_mask.sum(), dim=0)
        
        cam_idx_mask = (data.cam_idx == 1)
        rho_d[cam_idx_mask] = self.tex_rho_d[:, :, 3:6].unsqueeze(0).repeat_interleave(cam_idx_mask.sum(), dim=0)
        rho_ss[cam_idx_mask] = self.tex_rho_ss[:, :, 3:6].unsqueeze(0).repeat_interleave(cam_idx_mask.sum(), dim=0)

        rho_s = self.tex_rho_s

        height_vertex = compute_height_vertex(db_vertex, db_normal, self.tex_height, mask)
        height_normal = compute_height_normal(db_vertex, db_normal, self.tex_height, mask)

        one_tensor = torch.ones(db_vertex.shape[:-1],
                                dtype=height_vertex.dtype, device=height_vertex.device)[..., None]
        zero_tensor = torch.zeros(height_normal.shape[:-1],
                                  dtype=height_normal.dtype, device=height_normal.device)[..., None]

        vert4map = torch.cat([height_vertex, one_tensor], dim=-1)
        normal4map = torch.cat([height_normal, zero_tensor], dim=-1)

        n_t = (ref_to_frame_mat[:, None, None, :, :] *
               normal4map[None, :, :, None, :]).sum(axis=-1)[:, :, :, :3]
        v_t = (ref_to_frame_mat[:, None, None, :, :] *
               vert4map[None, :, :, None, :]).sum(axis=-1)[:, :, :, :3]

        n_t = F.normalize(n_t, dim=-1) 
        distance_t = torch.linalg.norm(cam_pos[:, None, None, :] - v_t, dim=-1, keepdim=True) / 1000.0
        attenuation = 1.0 / (distance_t * distance_t)
        v_dirs_t = F.normalize(cam_pos[:, None, None, :] - v_t, dim=-1) 
        l_dirs_t = v_dirs_t.clone() 
        h_dirs_t = F.normalize(v_dirs_t + l_dirs_t, dim=-1)

        hdotl = torch.clamp((h_dirs_t * l_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0) 
        ndotl = torch.clamp((n_t * l_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0) 
        ndoth = torch.clamp((n_t * h_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0) 
        ndotv = torch.clamp((n_t * v_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0) 

        theta_h = torch.acos(ndoth) 
        theta_i = torch.acos(ndotl) 
        theta_o = torch.acos(ndotv) 

        Rs_i, Rp_i, _, _ = compute_Fresnel(hdotl, 1.0, self.tex_refrac_idx.unsqueeze(0), False)
        Rpos = (Rs_i + Rp_i) / 2.0

        D_s = compute_D_GGX(self.tex_alpha_s.unsqueeze(0), theta_h) 
        D_ss = compute_D_GGX(self.tex_alpha_ss.unsqueeze(0), theta_h) 

        G_s = compute_G_smith(self.tex_alpha_s.unsqueeze(0), theta_i, theta_o) 
        G_ss = compute_G_smith(self.tex_alpha_ss.unsqueeze(0), theta_i, theta_o) 

        DG_s = rho_s * D_s * G_s 
        DG_ss = rho_ss * D_ss * G_ss 

        _, _, Ts_i, Tp_i = compute_Fresnel(ndotl, 1.0, self.tex_refrac_idx.unsqueeze(0), False)
        _, _, Ts_o, Tp_o = compute_Fresnel(ndotv, self.tex_refrac_idx.unsqueeze(0), 1.0, True)
        Tpos_o = (Ts_o + Tp_o) / 2.0
        Tpos_i = (Ts_i + Tp_i) / 2.0
        TpoTpi = Tpos_o * Tpos_i

        rendered_d = torch.zeros([N, h, w, 3], dtype=self.tex_rho_d.dtype, device=self.tex_rho_d.device)
        rendered_s = torch.zeros([N, h, w, 1], dtype=self.tex_rho_s.dtype, device=self.tex_rho_s.device)
        rendered_ss = torch.zeros([N, h, w, 3], dtype=self.tex_rho_ss.dtype, device=self.tex_rho_ss.device)

        rendered_d[mask] = \
            (rho_d * TpoTpi * ndotl * attenuation)[mask]
        rendered_s[mask] = \
            (Rpos * DG_s / (4.0 * ndotv) * attenuation)[mask]  
        rendered_ss[mask] = \
            (Rpos * DG_ss / (4.0 * ndotv) * attenuation)[mask] 

        return rendered_d, rendered_s, rendered_ss

