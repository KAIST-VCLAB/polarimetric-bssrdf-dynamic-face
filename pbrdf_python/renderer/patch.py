from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn

from renderer.utils import *
from renderer.base import PbrdfRenderer


class PbrdfRendererPatch(PbrdfRenderer):
    def __init__(self, args:edict) -> None:
        super().__init__(args)
        self.refrac_idx = nn.Parameter(args.init.refrac_idx * torch.ones([args.data.h, args.data.w, 1]), requires_grad=args.opt.use_refrac)
        self.rho_s = nn.Parameter(args.init.rho_s * torch.ones([args.data.h, args.data.w, 1]))
        self.rho_ss = nn.Parameter(args.init.rho_ss * torch.ones([args.data.h, args.data.w, args.data.c]))
        self.alpha_s = nn.Parameter(args.init.alpha_s * torch.ones([args.data.h, args.data.w, 1]))
        self.alpha_ss = nn.Parameter(args.init.alpha_ss * torch.ones([args.data.h, args.data.w, 1]))
        self.height = nn.Parameter(torch.zeros([args.data.h, args.data.w, 1]))
        self.rho_d = nn.Parameter(args.init.rho_d * torch.ones([args.data.h, args.data.w, args.data.c])) #, requires_grad=False)
        

    def to_render(self):
        self.tex_refrac_idx = self.refrac_idx.detach().clone()
        self.tex_rho_s = self.rho_s.detach().clone()
        self.tex_rho_ss = self.rho_ss.detach().clone()
        self.tex_alpha_s = self.alpha_s.detach().clone()
        self.tex_alpha_ss = self.alpha_ss.detach().clone()
        self.tex_height = self.height.detach().clone()
        self.tex_rho_d = self.rho_d.detach().clone()


    def load_ckpt(self, args):
        if args.opt.restore_ckpt[0]:
            assert os.path.exists(args.opt.ckpt_path[0]),'No checkpoint exists'
            self._load_ckpt(args, args.opt.ckpt_path[0], 'stage 1 ckpt -> stage 1 renderer')
    
    
    @torch.no_grad()
    def update_rho_d(self, args, vertex_map, normal_map, I_diffuse, I_mask, data):
        h = self.h
        w = self.w
        c = I_diffuse.shape[-1]
        N = I_diffuse.shape[0]
        
        I_diffuse = I_diffuse.cuda()
        ref_to_frame_mat = data.ref_to_frame.cuda()
        cam_pos = data.cam_to_world[:, :3, 3].cuda()
        I_mask = I_mask.cuda()

        # [H, W, 4]
        vertmap = vertex_map.cuda()
        normalmap = normal_map.cuda()

        rho_d = torch.zeros([h, w, 6]).cuda()
        print(rho_d.shape)
        for y in range(0, self.h, self.patch_size):
            for x in range(0, self.w, self.patch_size):
                ey = y + self.patch_size
                ex = x + self.patch_size
                if ey > self.h: ey = self.h
                if ex > self.w: ex = self.w

                vert4map = vertmap[y:ey, x:ex, :]
                normal4map = normalmap[y:ey, x:ex, :]

                I_mask_patch = I_mask[:, y:ey, x:ex]
                if I_mask_patch.sum() == 0: 
                    continue
                I_diffuse_patch = I_diffuse[:, y:ey, x:ex, :]
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
                I_mask_patch[~(shade_mask.squeeze())] = False

                rho_d_weights = ndotl.clone()
                rho_d_weights[~I_mask_patch] = 0
                thres_mask = (rho_d_weights >= args.threshold.ndotl)
                rho_d_weights[~thres_mask] = 0

                bad_region = ~torch.any(thres_mask, dim=0, keepdim=True)
                bad_region = bad_region.repeat_interleave(N, dim=0)
                rho_d_weights[bad_region] = ndotl[bad_region]
                rho_d_weights[~I_mask_patch] = 0

                rho_d_weights = rho_d_weights * rho_d_weights

                rendered_shade = torch.ones([N, ey-y, ex-x, 1], dtype=self.tex_rho_d.dtype, device=self.tex_rho_d.device)
                rho_d_image = torch.zeros([N, ey-y, ex-x, c], dtype=self.tex_rho_d.dtype, device=self.tex_rho_d.device)

                rendered_shade[I_mask_patch] = (TpoTpi * ndotl * attenuation)[I_mask_patch]
                rendered_shade = rendered_shade.repeat_interleave(c, dim=-1)
                rho_d_image[I_mask_patch] = I_diffuse_patch[I_mask_patch] / rendered_shade[I_mask_patch]

                sum_weights = torch.sum(rho_d_weights[:self.num_frames], dim=0)
                rho_d_patch = \
                    torch.sum(rho_d_image[:self.num_frames] * rho_d_weights[:self.num_frames], dim=0) / sum_weights
                rho_d_patch[~(sum_weights.squeeze() > 0.025)] = 0
                rho_d[y:ey, x:ex, :3] = rho_d_patch

                sum_weights = torch.sum(rho_d_weights[self.num_frames:2 * self.num_frames], dim=0)
                rho_d_patch = \
                    torch.sum(rho_d_image[self.num_frames:2 * self.num_frames] *
                                rho_d_weights[self.num_frames:2 * self.num_frames], dim=0) / sum_weights
                rho_d_patch[~(sum_weights.squeeze() > 0.025)] = 0
                rho_d[y:ey, x:ex, 3:6] = rho_d_patch


        return rho_d
    

    @torch.no_grad()
    def update_rho_d_patch(self, I_diffuse, ndotl, mask_all, mask_patch, TpoTpi, attenuation):
        rho_d_weights = ndotl.clone()
        rho_d_weights[~mask_all] = 0
        thres_mask = (rho_d_weights >= self.threshold.ndotl)
        rho_d_weights[~thres_mask] = 0

        bad_region = ~torch.any(thres_mask, dim=0, keepdim=True)
        bad_region = bad_region.repeat_interleave(self.N, dim=0)
        rho_d_weights[bad_region] = ndotl[bad_region]
        rho_d_weights[~mask_all] = 0

        rendered_shade = torch.ones([self.N, self.patch_size, self.patch_size, 1], dtype=ndotl.dtype, device=ndotl.device)
        rho_d_image = torch.zeros([self.N, self.patch_size, self.patch_size, 3], dtype=I_diffuse.dtype, device=I_diffuse.device)

        rendered_shade[mask_all] = (TpoTpi * ndotl * attenuation)[mask_all]
        rendered_shade = rendered_shade.repeat_interleave(3, dim=-1)
        rho_d_image[mask_all] = I_diffuse[mask_all] / rendered_shade[mask_all]

        rho_d = \
            torch.sum(rho_d_image[:self.num_frames] * rho_d_weights[:self.num_frames], dim=0) / \
            torch.sum(rho_d_weights[:self.num_frames], dim=0)
        weight_mask = (torch.sum(rho_d_weights[:self.num_frames], dim=0) > 0.025).squeeze(-1)
        region_weight_mask = mask_patch.clone()
        temp_mask = region_weight_mask[mask_patch]
        temp_mask[~(weight_mask.ravel())] = False
        region_weight_mask[mask_patch] = temp_mask
        self.rho_d[:, :, :3][region_weight_mask] = rho_d[weight_mask]

        rho_d = \
            torch.sum(rho_d_image[self.num_frames:2 * self.num_frames] *
                        rho_d_weights[self.num_frames:2 * self.num_frames], dim=0) / \
            torch.sum(rho_d_weights[self.num_frames:2 * self.num_frames], dim=0)
        weight_mask = (torch.sum(rho_d_weights[self.num_frames:2 * self.num_frames], dim=0) > 0.025).squeeze(-1)
        region_weight_mask = mask_patch.clone()
        temp_mask = region_weight_mask[mask_patch]
        temp_mask[~(weight_mask.ravel())] = False
        region_weight_mask[mask_patch] = temp_mask
        self.rho_d[:, :, 3:6][region_weight_mask] = rho_d[weight_mask]


    def forward(self, data:edict) -> edict:
        for k, v in data.items():
            assert v.shape[0] == 1
            data[k] = v.squeeze(0)

        update_patch = edict()
        for k, v in self.named_parameters():
            update_patch[k] = v[data.mask_patch].reshape(1, self.patch_size, self.patch_size, -1)

        db_vertex = data.I_vertex
        db_normal = data.I_normal

        cam_y_dirs = (data.cam_to_world[:, 1, :3])[:, None, None, :].repeat(1, self.patch_size, self.patch_size, 1)

        ref_to_frame_mat = data.ref_to_frame
        cam_pos = data.cam_to_world[:, :3, 3]  

        I_alpha = data.I_alpha
        I_spec = data.I_specular
        I_diffuse = data.I_diffuse

        height_vertex = compute_height_vertex(db_vertex, db_normal, update_patch.height.squeeze(0))
        height_normal = compute_height_normal(db_vertex, db_normal, update_patch.height.squeeze(0))

        one_tensor = torch.ones(height_vertex.shape[:-1],
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
        v_dirs_t = F.normalize(cam_pos[:, None, None, :] - v_t, dim=-1)
        l_dirs_t = v_dirs_t.clone() 
        h_dirs_t = F.normalize(v_dirs_t + l_dirs_t, dim=-1)

        hdotl = torch.clamp((h_dirs_t * l_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0 - 1e-6) 
        ndotl = torch.clamp((n_t * l_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0 - 1e-6)  
        ndoth = torch.clamp((n_t * h_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0 - 1e-6) 
        ndotv = torch.clamp((n_t * v_dirs_t).sum(axis=-1, keepdim=True), 0.0, 1.0 - 1e-6) 
        
        # Mask
        mask_image = data.I_mask.clone()
        mask_patch_N = data.mask_patch.unsqueeze(0).repeat_interleave(self.N, dim=0)

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
        attenuation[mask_all] = 1.0 / (distance_t * distance_t)[mask_all]


        theta_h = torch.zeros_like(ndoth)
        theta_i = torch.zeros_like(ndoth)
        theta_o = torch.zeros_like(ndoth)

        theta_h[mask_all] = torch.acos(ndoth[mask_all])
        theta_i[mask_all] = torch.acos(ndotl[mask_all]) 
        theta_o[mask_all] = torch.acos(ndotv[mask_all]) 


        detached_refrac_idx = update_patch.refrac_idx 

        Rs_i, Rp_i, _, _ = compute_Fresnel(hdotl, 1.0, detached_refrac_idx, False)
        Rpos = (Rs_i + Rp_i) / 2.0

        D_s = compute_D_GGX(update_patch.alpha_s, theta_h)  
        D_ss = compute_D_GGX(update_patch.alpha_ss, theta_h)  

        G_s = compute_G_smith(update_patch.alpha_s, theta_i, theta_o)  
        G_ss = compute_G_smith(update_patch.alpha_ss, theta_i, theta_o)  

        DG_s = update_patch.rho_s * D_s * G_s  
        DG_ss = update_patch.rho_ss * D_ss * G_ss  

        _, _, Ts_i, Tp_i = compute_Fresnel(ndotl, 1.0, detached_refrac_idx, False)
        _, _, Ts_o, Tp_o = compute_Fresnel(ndotv, detached_refrac_idx, 1.0, True)
        Tpos_o = (Ts_o + Tp_o) / 2.0
        Tpos_i = (Ts_i + Tp_i) / 2.0
        TpoTpi = Tpos_o * Tpos_i

        self.update_rho_d_patch(I_diffuse, ndotl, mask_all, data.mask_patch, TpoTpi, attenuation)
        rho_d = self.rho_d[data.mask_patch].reshape(1, self.patch_size, self.patch_size, -1)

        rendered_d = torch.zeros([self.N, self.patch_size, self.patch_size, self.c], dtype=rho_d.dtype, device=rho_d.device)
        rendered_s = torch.zeros([self.N, self.patch_size, self.patch_size, 1], dtype=rho_d.dtype, device=rho_d.device)
        rendered_ss = torch.zeros([self.N, self.patch_size, self.patch_size, self.c], dtype=rho_d.dtype, device=rho_d.device)


        rendered_d[mask_all] = \
            (rho_d * ndotl * attenuation)[mask_all] * TpoTpi[mask_all]

        rendered_s[mask_specular] = (Rpos[mask_specular] * DG_s[mask_specular] * attenuation[mask_specular])
        rendered_s[mask_specular] = (rendered_s[mask_specular]) / ((4.0 * ndotv)[mask_specular]) 

        rendered_s[mask_single_scattering] = ((Rpos * DG_s * attenuation)[mask_single_scattering]).data.clone().detach()
        rendered_s[mask_single_scattering] = ((rendered_s[mask_single_scattering]) / ((4.0 * ndotv)[mask_single_scattering])).data.clone().detach() 
        
        rendered_ss[mask_single_scattering] = (Rpos[mask_single_scattering] * DG_ss[mask_single_scattering] * attenuation[mask_single_scattering]) 
        rendered_ss[mask_single_scattering] = (rendered_ss[mask_single_scattering]) / ((4.0 * ndotv)[mask_single_scattering])
        rendered_ss[mask_specular] = ((Rpos * DG_ss * attenuation)[mask_specular]).data.clone().detach()
        rendered_ss[mask_specular] = ((rendered_s[mask_specular]) / ((4.0 * ndotv)[mask_specular])).data.clone().detach() 
        
        rendered_d3, rendered_ss3 = grab_images(self.num_cams, self.num_frames, rendered_d, rendered_ss, False)

        I_beta = torch.zeros_like(rendered_ss3)
        I_beta[mask_all] = -(I_spec[mask_all] - rendered_s[mask_all] - rendered_ss3[mask_all]) 
        
        normalized_color = F.normalize(I_diffuse[mask_all], dim=-1)
        mean_color = torch.mean(normalized_color, dim=0, keepdim=True)
        normalized_mean_color = mean_color.clone() 
        weight_color = normalized_mean_color / torch.sum(normalized_mean_color)
        weight_color = weight_color[None, None, :, :]
        
        I_diffuse_mono = torch.sum(I_diffuse * weight_color, dim=-1, keepdim=True)
        I_spec_mono = torch.sum(I_spec * weight_color, dim=-1, keepdim=True)
        beta_mono = torch.sum(I_beta * weight_color, dim=-1, keepdim=True)
        alpha_mono = torch.sum(I_alpha * weight_color, dim=-1, keepdim=True)

        mask_dop = (I_diffuse_mono > I_spec_mono)
        mask_dop_full = torch.all(torch.cat([mask_all.unsqueeze(-1), mask_dop], dim=-1), dim=-1)

        I_diffuse_polarization_mono = torch.zeros_like(alpha_mono)
        I_diffuse_polarization_mono[mask_dop_full] = torch.sqrt(alpha_mono * alpha_mono + beta_mono * beta_mono + 1e-8)[mask_dop_full]

        I_DoP = torch.zeros_like(I_diffuse_polarization_mono)

        I_DoP[mask_dop_full] = I_diffuse_polarization_mono[mask_dop_full] / I_diffuse_mono[mask_dop_full]
        
        I_DoP_mask = (I_DoP < 1.0)
        mask_dop_full = torch.all(torch.cat([mask_dop_full.unsqueeze(-1), I_DoP_mask], dim=-1), dim=-1)

        rendered_DoP = compute_DoP_mask(theta_o, update_patch.refrac_idx, mask_dop_full)
        rendered_DoP[~mask_dop_full] = 0

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
        predictions.mask_dop_full = mask_dop_full
        predictions.I_alpha_mono = I_alpha_mono
        predictions.I_beta_mono = I_beta_mono
        predictions.alpha_mono = rendered_alpha
        predictions.beta_mono = rendered_beta
        predictions.I_DoP = I_DoP
        predictions.DoP = rendered_DoP

        
        predictions.height = update_patch.height.squeeze(0)
        predictions.refrac_idx = update_patch.refrac_idx.squeeze(0)
        predictions.alpha_s = update_patch.alpha_s.squeeze(0)
        predictions.alpha_ss = update_patch.alpha_ss.squeeze(0)
        predictions.rho_d = rho_d.squeeze(0)

        return predictions
    