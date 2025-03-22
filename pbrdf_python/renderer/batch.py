from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn

from renderer.utils import *
from renderer.base import PbrdfRenderer


class PbrdfRendererBatch(PbrdfRenderer):
    def __init__(self, args:edict, dataset:edict) -> None:
        super().__init__(args)
        self.N = args.data.num_frames 
        self.refrac_idx = nn.Parameter(dataset.dataset.tex_refrac_idx.unsqueeze(0).clone(), requires_grad=False)
        self.alpha_s = nn.Parameter(dataset.dataset.tex_alpha_s.unsqueeze(0).clone(), requires_grad=False)
        self.alpha_ss = nn.Parameter(dataset.dataset.tex_alpha_ss.unsqueeze(0).clone(), requires_grad=False)

        tex_rho_s = dataset.dataset.tex_rho_s.unsqueeze(0).repeat_interleave(self.N, dim=0)
        tex_rho_ss = dataset.dataset.tex_rho_ss.unsqueeze(0).repeat_interleave(self.N, dim=0)
        tex_height = dataset.dataset.tex_height.unsqueeze(0).repeat_interleave(self.N, dim=0)
        tex_rho_d = dataset.dataset.tex_rho_d.unsqueeze(0).repeat_interleave(self.N, dim=0)

        self.rho_s = nn.Parameter(tex_rho_s.clone())
        self.rho_ss = nn.Parameter(tex_rho_ss.clone())
        self.height = nn.Parameter(tex_height.clone())
        self.rho_d = nn.Parameter(tex_rho_d.clone())

        self.visualize_idx = self.N // 2
        

    def to_render(self):
        idx = self.visualize_idx
        self.tex_refrac_idx = self.refrac_idx[0].detach().clone()
        self.tex_rho_s = self.rho_s[idx].detach().clone()
        self.tex_rho_ss = self.rho_ss[idx].detach().clone()
        self.tex_alpha_s = self.alpha_s[0].detach().clone()
        self.tex_alpha_ss = self.alpha_ss[0].detach().clone()
        self.tex_height = self.height[idx].detach().clone()
        self.tex_rho_d = self.rho_d[idx].detach().clone()

    def get_textures(self):
        tex = edict()
        self.tex_refrac_idx = self.refrac_idx.detach().clone()
        self.tex_rho_s = self.rho_s.detach().clone()
        self.tex_rho_ss = self.rho_ss.detach().clone()
        self.tex_alpha_s = self.alpha_s.detach().clone()
        self.tex_alpha_ss = self.alpha_ss.detach().clone()
        self.tex_height = self.height.detach().clone()
        self.tex_rho_d = self.rho_d.detach().clone()

        tex.tex_rho_s = self.tex_rho_s.clone()
        tex.tex_rho_ss = self.tex_rho_ss.clone()
        tex.tex_height = self.tex_height.clone()
        tex.tex_rho_d = self.tex_rho_d.clone()

        return tex
    
    @torch.no_grad()
    def compute_vertex_normal_map(self, I_vertex, I_normal):
        device = self.tex_height.device
        idx = self.visualize_idx
        vertex = I_vertex[idx].to(device)
        normal = I_normal[idx].to(device)
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
    def compute_vertex_normal_maps(self, I_vertex, I_normal, idx):
        device = self.tex_height.device
        vertex = I_vertex.to(device)
        normal = I_normal.to(device)
        normalmap = compute_batch_height_normal(vertex, normal, self.tex_height[idx].detach().clone())
        vertmap = compute_height_vertex(vertex, normal, self.tex_height[idx].detach().clone())

        one_tensor = torch.ones(vertmap.shape[:-1],
                                dtype=vertmap.dtype, device=vertmap.device)[..., None]
        zero_tensor = torch.zeros(normalmap.shape[:-1],
                                  dtype=normalmap.dtype, device=normalmap.device)[..., None]

        vert4map = torch.cat([vertmap, one_tensor], dim=-1)
        normal4map = torch.cat([normalmap, zero_tensor], dim=-1)

        return vert4map, normal4map
       
    def load_ckpt(self, args):
        if args.opt.restore_ckpt[5]:
            assert os.path.exists(args.opt.ckpt_path[5]),'No checkpoint exists'
            self._load_ckpt(args, args.opt.ckpt_path[5], 'stage 6 ckpt -> stage 6 renderer')
    
    def load_ckpt_path(self, args, path):
        if args.opt.restore_ckpt[5]:
            assert os.path.exists(path),'No checkpoint exists'
            self._load_ckpt(args, path, 'stage 6 ckpt -> stage 6 renderer')

    @torch.no_grad()
    def update_rho_d_patch(self, I_diffuse, ndotl, mask_all, mask_patch, TpoTpi, attenuation):
        rho_d_weights = ndotl.clone()
        rho_d_weights = rho_d_weights * rho_d_weights
        rho_d_weights[~mask_all] = 0
        N = mask_all.shape[0]

        rendered_shade = torch.ones([N, self.patch_size, self.patch_size, 1], dtype=ndotl.dtype, device=ndotl.device)
        rho_d_image = torch.zeros([N, self.patch_size, self.patch_size, 3], dtype=I_diffuse.dtype, device=I_diffuse.device)

        rendered_shade[mask_all] = (TpoTpi * ndotl * attenuation)[mask_all]
        rendered_shade = rendered_shade.repeat_interleave(3, dim=-1)
        rho_d_image[mask_all] = I_diffuse[mask_all] / rendered_shade[mask_all]

        rho_d = \
            torch.sum(rho_d_image[:self.num_frames] * rho_d_weights[:self.num_frames], dim=0) / \
            torch.sum(rho_d_weights[:self.num_frames], dim=0)
        weight_mask = (torch.sum(rho_d_weights[:self.num_frames], dim=0) > 0.5).squeeze(-1)
        region_weight_mask = mask_patch.clone()
        temp_mask = region_weight_mask[mask_patch]
        temp_mask[~(weight_mask.ravel())] = False
        region_weight_mask[mask_patch] = temp_mask
        self.rho_d[:, :, :3][region_weight_mask] = rho_d[weight_mask]

        rho_d = \
            torch.sum(rho_d_image[self.num_frames:2 * self.num_frames] *
                        rho_d_weights[self.num_frames:2 * self.num_frames], dim=0) / \
            torch.sum(rho_d_weights[self.num_frames:2 * self.num_frames], dim=0)
        weight_mask = (torch.sum(rho_d_weights[self.num_frames:2 * self.num_frames], dim=0) > 0.5).squeeze(-1)
        region_weight_mask = mask_patch.clone()
        temp_mask = region_weight_mask[mask_patch]
        temp_mask[~(weight_mask.ravel())] = False
        region_weight_mask[mask_patch] = temp_mask
        self.rho_d[:, :, 3:6][region_weight_mask] = rho_d[weight_mask]


    @torch.no_grad()
    def render(self, data:edict):
        self.tex_refrac_idx = self.refrac_idx.detach().clone()
        self.tex_rho_s = self.rho_s.detach().clone()
        self.tex_rho_ss = self.rho_ss.detach().clone()
        self.tex_alpha_s = self.alpha_s.detach().clone()
        self.tex_alpha_ss = self.alpha_ss.detach().clone()
        self.tex_height = self.height.detach().clone()
        self.tex_rho_d = self.rho_d.detach().clone()
        h = self.h
        w = self.w
        N = data.I_diffuse.shape[0]

        ref_to_frame_mat = data.ref_to_frame
        cam_pos = data.cam_to_world[:, :3, 3] 
        db_vertex = data.I_vertex
        db_normal = data.I_normal
        mask = data.I_mask
        I_alpha = data.I_alpha
        I_spec = data.I_specular
        I_diffuse = data.I_diffuse

        rho_d = torch.zeros_like(I_diffuse)
        rho_ss = torch.zeros_like(I_spec)

        cam_idx_mask = (data.cam_idx == 0)
        rho_d[cam_idx_mask] = self.tex_rho_d[data.frame_idx][cam_idx_mask][:, :, :, :3]
        rho_ss[cam_idx_mask] = self.tex_rho_ss[data.frame_idx][cam_idx_mask][:, :, :, :3]
        
        cam_idx_mask = (data.cam_idx == 1)
        rho_d[cam_idx_mask] = self.tex_rho_d[data.frame_idx][cam_idx_mask][:, :, :, 3:6]
        rho_ss[cam_idx_mask] = self.tex_rho_ss[data.frame_idx][cam_idx_mask][:, :, :, 3:6]

        rho_s = self.tex_rho_s[data.frame_idx]
        height = self.tex_height[data.frame_idx]

        height_vertex = compute_height_vertex(db_vertex, db_normal, height)
        height_normal = compute_batch_height_normal(db_vertex, db_normal, height)

        one_tensor = torch.ones(db_vertex.shape[:-1],
                                dtype=height_vertex.dtype, device=height_vertex.device)[..., None]
        zero_tensor = torch.zeros(height_normal.shape[:-1],
                                  dtype=height_normal.dtype, device=height_normal.device)[..., None]

        vert4map = torch.cat([height_vertex, one_tensor], dim=-1)
        normal4map = torch.cat([height_normal, zero_tensor], dim=-1)

        n_t = (ref_to_frame_mat[:, None, None, :, :] *
               normal4map[:, :, :, None, :]).sum(axis=-1)[:, :, :, :3]
        v_t = (ref_to_frame_mat[:, None, None, :, :] *
               vert4map[:, :, :, None, :]).sum(axis=-1)[:, :, :, :3]

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

        Rs_i, Rp_i, _, _ = compute_Fresnel(hdotl, 1.0, self.tex_refrac_idx, False)
        Rpos = (Rs_i + Rp_i) / 2.0

        D_s = compute_D_GGX(self.tex_alpha_s, theta_h) 
        D_ss = compute_D_GGX(self.tex_alpha_ss, theta_h) 

        G_s = compute_G_smith(self.tex_alpha_s, theta_i, theta_o) 
        G_ss = compute_G_smith(self.tex_alpha_ss, theta_i, theta_o) 

        DG_s = rho_s * D_s * G_s 
        DG_ss = rho_ss * D_ss * G_ss 

        _, _, Ts_i, Tp_i = compute_Fresnel(ndotl, 1.0, self.tex_refrac_idx, False)
        _, _, Ts_o, Tp_o = compute_Fresnel(ndotv, self.tex_refrac_idx, 1.0, True)
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


    def forward(self, data:edict) -> edict:
        update_batch = edict()
        batch_size = data.mask_patch.shape[0]
        for k, v in self.named_parameters():
            if k == 'refrac_idx' or k == 'alpha_s' or k == 'alpha_ss':
                update_batch[k] = v.repeat_interleave(batch_size, axis=0)[data.mask_patch].reshape(batch_size,  self.patch_size, self.patch_size, -1)
            else:
                update_batch[k] = v[data.frame_idx][data.mask_patch].reshape(batch_size, self.patch_size, self.patch_size, -1)

        db_vertex = data.I_vertex
        db_normal = data.I_normal

        cam_y_dirs = (data.cam_to_world[:, 1, :3])[:, None, None, :].repeat(1, self.patch_size, self.patch_size, 1)

        ref_to_frame_mat = data.ref_to_frame
        cam_pos = data.cam_to_world[:, :3, 3] 

        I_alpha = data.I_alpha
        I_spec = data.I_specular
        I_diffuse = data.I_diffuse

        height_vertex = compute_height_vertex(db_vertex, db_normal, update_batch.height)
        height_normal = compute_batch_height_normal(db_vertex, db_normal, update_batch.height)

        one_tensor = torch.ones(height_vertex.shape[:-1],
                                dtype=height_vertex.dtype, device=height_vertex.device)[..., None]
        zero_tensor = torch.zeros(height_normal.shape[:-1],
                                  dtype=height_normal.dtype, device=height_normal.device)[..., None]

        vert4map = torch.cat([height_vertex, one_tensor], dim=-1)
        normal4map = torch.cat([height_normal, zero_tensor], dim=-1)

        n_t = (ref_to_frame_mat[:, None, None, :, :] *
               normal4map[:, :, :, None, :]).sum(axis=-1)[:, :, :, :3]
        v_t = (ref_to_frame_mat[:, None, None, :, :] *
               vert4map[:, :, :, None, :]).sum(axis=-1)[:, :, :, :3]

        n_t = F.normalize(n_t, dim=-1) 
        v_dirs_t = F.normalize(cam_pos[:, None, None, :] - v_t, dim=-1) 
        l_dirs_t = v_dirs_t.clone() 
        h_dirs_t = F.normalize(v_dirs_t + l_dirs_t, dim=-1)

        hdotl = torch.clamp((h_dirs_t * l_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0 - 1e-4)  
        ndotl = torch.clamp((n_t * l_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0 - 1e-4)  
        ndoth = torch.clamp((n_t * h_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0 - 1e-4)  
        ndotv = torch.clamp((n_t * v_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0 - 1e-4)  

        mask_image = data.I_mask.clone()

        mask_shade = ndotl > self.threshold.ndotl
        mask_shade = torch.cat([mask_shade, mask_image.unsqueeze(-1)], dim=-1)
        mask_shade = torch.all(mask_shade, dim=-1)

        mask_diffuse = torch.mean(I_diffuse, dim=-1, keepdim=True) > self.threshold.diffuse
        mask_all = torch.all(torch.cat([mask_diffuse, mask_shade.unsqueeze(-1)], dim=-1), dim=-1)
        mask_all[:, 0, :] = False
        mask_all[:, -1, :] = False
        mask_all[:, :, 0] = False
        mask_all[:, :, -1] = False

        mask_ndoth = ndoth > self.threshold.ndoth
        mask_specular = torch.cat([mask_all.unsqueeze(-1), mask_ndoth], dim=-1)
        mask_specular = torch.all(mask_specular, dim=-1)
        mask_single_scattering = torch.cat([mask_all.unsqueeze(-1), ~mask_ndoth], dim=-1)
        mask_single_scattering = torch.all(mask_single_scattering, dim=-1)

        distance_t = torch.linalg.norm(cam_pos[:, None, None, :] - v_t, dim=-1, keepdim=True) / 1000.0
        attenuation = torch.zeros_like(distance_t)
        attenuation[mask_all] = (1.0 / (distance_t * distance_t)[mask_all])


        theta_h = torch.zeros_like(ndoth)
        theta_i = torch.zeros_like(ndoth)
        theta_o = torch.zeros_like(ndoth)

        theta_h[mask_all] = torch.acos(ndoth[mask_all])  
        theta_i[mask_all] = torch.acos(ndotl[mask_all])  
        theta_o[mask_all] = torch.acos(ndotv[mask_all])  
        Rs_i, Rp_i, _, _ = compute_Fresnel(hdotl, 1.0, update_batch.refrac_idx, False)
        Rpos = (Rs_i + Rp_i) / 2.0

        D_s = compute_D_GGX(update_batch.alpha_s, theta_h)  
        D_ss = compute_D_GGX(update_batch.alpha_ss, theta_h)  

        G_s = compute_G_smith(update_batch.alpha_s, theta_i, theta_o)  
        G_ss = compute_G_smith(update_batch.alpha_ss, theta_i, theta_o)  

        DG_s = update_batch.rho_s * D_s * G_s  
        DG_ss = update_batch.rho_ss * D_ss * G_ss  

        _, _, Ts_i, Tp_i = compute_Fresnel(ndotl, 1.0, update_batch.refrac_idx, False)
        _, _, Ts_o, Tp_o = compute_Fresnel(ndotv, update_batch.refrac_idx, 1.0, True)
        Tpos_o = (Ts_o + Tp_o) / 2.0
        Tpos_i = (Ts_i + Tp_i) / 2.0
        TpoTpi = Tpos_o * Tpos_i

        rho_d = update_batch.rho_d

        rendered_d = torch.zeros([batch_size, self.patch_size, self.patch_size, self.c], dtype=rho_d.dtype, device=rho_d.device)
        rendered_s = torch.zeros([batch_size, self.patch_size, self.patch_size, 1], dtype=rho_d.dtype, device=rho_d.device)
        rendered_ss = torch.zeros([batch_size, self.patch_size, self.patch_size, self.c], dtype=rho_d.dtype, device=rho_d.device)

        rendered_d[mask_all] = \
            (rho_d * ndotl * attenuation)[mask_all] * TpoTpi[mask_all]        

        rendered_s[mask_specular] = (Rpos[mask_specular] * DG_s[mask_specular] * attenuation[mask_specular])
        rendered_s[mask_specular] = (rendered_s[mask_specular]) / ((4.0 * ndotv)[mask_specular]) # ndotl disappear in the rendering process
        rendered_s[mask_single_scattering] = ((Rpos * DG_s * attenuation)[mask_single_scattering]).data.clone().detach()
        rendered_s[mask_single_scattering] = ((rendered_s[mask_single_scattering]) / ((4.0 * ndotv)[mask_single_scattering])).data.clone().detach() # ndotl disappear in the rendering process
        
        rendered_ss[mask_single_scattering] = (Rpos[mask_single_scattering] * DG_ss[mask_single_scattering] * attenuation[mask_single_scattering])  # ndotl disappear in the rendering process
        rendered_ss[mask_single_scattering] = (rendered_ss[mask_single_scattering]) / ((4.0 * ndotv)[mask_single_scattering])
        rendered_ss[mask_specular] = ((Rpos * DG_ss * attenuation)[mask_specular]).data.clone().detach()
        rendered_ss[mask_specular] = ((rendered_s[mask_specular]) / ((4.0 * ndotv)[mask_specular])).data.clone().detach() # ndotl disappear in the rendering process
           
        rendered_d3, rendered_ss3 = grab_images_by_index(data.cam_idx, rendered_d, rendered_ss)

        I_beta = torch.zeros_like(rendered_ss3)
        I_beta[mask_all] = -(I_spec[mask_all] - rendered_s[mask_all] - rendered_ss3[mask_all]) 

        normalized_color = F.normalize(I_diffuse[mask_all], dim=-1)
        mean_color = torch.mean(normalized_color, dim=0, keepdim=True)
        normalized_mean_color = F.normalize(mean_color, dim=-1)
        weight_color = normalized_mean_color / torch.sum(normalized_mean_color)
        weight_color = weight_color[None, None, :, :]
        
        I_diffuse_mono = torch.sum(I_diffuse * weight_color, dim=-1, keepdim=True)
        beta_mono = torch.sum(I_beta * weight_color, dim=-1, keepdim=True)
        alpha_mono = torch.sum(I_alpha * weight_color, dim=-1, keepdim=True)

        I_diffuse_polarization_mono = torch.zeros_like(alpha_mono)
        I_diffuse_polarization_mono[mask_all] = torch.sqrt(alpha_mono * alpha_mono + beta_mono * beta_mono + 1e-8)[mask_all]

        I_DoP = torch.zeros_like(I_diffuse_polarization_mono)

        I_DoP[mask_all] = I_diffuse_polarization_mono[mask_all] / I_diffuse_mono[mask_all]

        I_DoP_mask = (I_DoP < 1.0)
        mask_all = torch.all(torch.cat([mask_all.unsqueeze(-1), I_DoP_mask], dim=-1), dim=-1)
        
        rendered_DoP = compute_DoP_mask(theta_o, update_batch.refrac_idx, mask_all)
        rendered_DoP[~mask_all] = 0

        alpha, beta = compute_angle(n_t, v_dirs_t, -cam_y_dirs, mask_all)

        rendered_alpha = -alpha * I_diffuse_polarization_mono
        rendered_beta = -beta * I_diffuse_polarization_mono

        I_alpha_mono = torch.sum(I_alpha * weight_color, dim=-1, keepdim=True)
        I_beta_mono = torch.sum(I_beta * weight_color, dim=-1, keepdim=True)

        predictions = edict()
        predictions.diffuse = rendered_d3
        predictions.specular = rendered_s
        predictions.single_scattering = rendered_ss3
        predictions.mask_all = mask_all
        predictions.I_alpha_mono = I_alpha_mono
        predictions.I_beta_mono = I_beta_mono
        predictions.alpha_mono = rendered_alpha
        predictions.beta_mono = rendered_beta
        predictions.I_DoP = I_DoP
        predictions.DoP = rendered_DoP

        predictions.height = update_batch.height
        predictions.rho_s = update_batch.rho_s
        reg_rho_d, reg_rho_ss = grab_images_by_index(data.cam_idx, update_batch.rho_d, update_batch.rho_ss)
        predictions.rho_ss = reg_rho_ss
        predictions.rho_d = reg_rho_d

        predictions.prev_rho_d = data.tex_rho_d
        predictions.prev_rho_s = data.tex_rho_s
        predictions.prev_rho_ss = data.tex_rho_ss
        predictions.rho_d_initial = data.tex_rho_d

        return predictions