import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.umask(0)

from easydict import EasyDict as edict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import scipy.io

from utils.io import load_images_parallel, load_images_glob_parallel, load_ref_images_parallel, get_camera_params, load_cam_images_parallel, load_images_sequence_parallel



class PbrdfDatasetBase(Dataset):
    def __init__(self, args):
        path = args.data.path
        h = args.data.h
        w = args.data.w
        num_cams = args.data.num_cams
        num_frames = args.data.num_frames

        texture_filename = args.data.texture_filename

        texture_path = os.path.join(path, texture_filename)

        I_0, I_0_visibility = None, None
        I_45, I_45_visibility = None, None
        I_90, I_90_visibility = None, None
        I_135, I_135_visibility = None, None
        I_diffuse = None
        I_specular = None
        I_alpha = None

        white_bal = torch.Tensor(np.loadtxt(os.path.join(path, 'white_balance.txt')).astype(np.float32))

        I_0_path = os.path.join(texture_path, '0/exr')
        I_0, I_0_visibility, _, _, s_i = load_images_parallel(I_0_path, h, w, white_bal, num_cams, num_frames)
        I_0_finite_mask = torch.all(torch.isfinite(I_0), dim=-1, keepdim=True)

        I_45_path = os.path.join(texture_path, '45/exr')
        I_45, I_45_visibility, _, _, _ = load_images_parallel(I_45_path, h, w, white_bal, num_cams, num_frames, s_i)
        I_45_finite_mask = torch.all(torch.isfinite(I_45), dim=-1, keepdim=True)

        I_90_path = os.path.join(texture_path, '90/exr')
        I_90, I_90_visibility, _, _, _ = load_images_parallel(I_90_path, h, w, white_bal, num_cams, num_frames, s_i)
        I_90_finite_mask = torch.all(torch.isfinite(I_90), dim=-1, keepdim=True)

        I_135_path = os.path.join(texture_path, '135/exr')
        I_135, I_135_visibility, _, _, _ = load_images_parallel(I_135_path, h, w, white_bal, num_cams, num_frames, s_i)
        I_135_finite_mask = torch.all(torch.isfinite(I_135), dim=-1, keepdim=True)

        I_mask = torch.cat([I_0_visibility, I_45_visibility, I_90_visibility, I_135_visibility], dim=-1)

        I_mask = (I_mask > 0.5)
        I_mask = torch.all(I_mask, dim=-1, keepdim=True)
        I_mask = torch.cat([I_mask, I_0_finite_mask, I_45_finite_mask, I_90_finite_mask, I_135_finite_mask],
                        dim=-1)
        I_mask = torch.all(I_mask, dim=-1)

        I_0[~I_mask] = 0
        I_45[~I_mask] = 0
        I_90[~I_mask] = 0
        I_135[~I_mask] = 0

        I_vertex_path = os.path.join(texture_path, 'vertex.exr')
        I_vertex = cv2.imread(os.path.join(path, I_vertex_path), -1)[:, :, ::-1].copy()
        I_vertex = torch.from_numpy(I_vertex)

        I_normal_path = os.path.join(texture_path, 'normal.exr')
        I_normal = cv2.imread(os.path.join(path, I_normal_path), -1)[:, :, ::-1].copy()
        I_normal = torch.from_numpy(I_normal)

        if args.data.num_cluster > 0:
            I_cluster_path = os.path.join(texture_path, 'cluster.png')
            I_cluster = cv2.imread(os.path.join(path, I_cluster_path), -1)[:, :, :3][:, :, ::-1].copy()
            I_cluster = torch.from_numpy(I_cluster).sum(dim=-1).type(torch.int32)
        else:
            I_cluster = None

        if args.general.test:
            d65_mat = np.loadtxt(f'{path}/d65_mat.txt')
            d65_mat_inv = np.linalg.inv(d65_mat.reshape(-1, 3, 3)[2])

            I_ref_path = os.path.join(texture_path, '90/exr')
            I_ref, I_ref_visibility, num_ref_frames, ref_si = load_ref_images_parallel(I_ref_path, h, w, white_bal, d65_mat_inv, num_cams, args.data.num_frames, s_i)
            I_ref_finite_mask = torch.all(torch.isfinite(I_ref), dim=-1, keepdim=True)
            
            I_diffuse_ref = 2.0 * I_ref
            I_mask_ref = (I_ref_visibility > 0.5)
            I_mask_ref = torch.cat([I_mask_ref, I_ref_finite_mask], dim=-1)
            I_mask_ref = torch.all(I_mask_ref, dim=-1)

            I_ref[~I_mask_ref] = 0

        I_diffuse = 2.0 * I_90
        I_specular = I_0 - I_90
        I_alpha = I_135 - I_45
        I_alpha_inv = I_45 - I_135

        frame_to_ref = torch.zeros([num_frames, 4, 4])
        ref_to_frame = torch.zeros([num_frames, 4, 4])
        icp_pose_path = os.path.join(path, 'icp-pose')
        icp_pose_txt_list = os.listdir(icp_pose_path)
        icp_pose_txt_list.sort()
        icp_pose_filenames = np.array(icp_pose_txt_list)
        for idx, pose_filename in enumerate(icp_pose_filenames[s_i]):
            icp_pose = np.loadtxt(os.path.join(icp_pose_path, pose_filename))
            frame_to_ref[idx, :, :] = torch.Tensor(icp_pose)
            ref_to_frame[idx, :, :] = torch.Tensor(np.linalg.inv(icp_pose))

        cam_intrinsic, world_to_cam, cam_to_world = get_camera_params(path, num_cams)

        frame_to_ref = frame_to_ref.repeat(num_cams, 1, 1)
        ref_to_frame = ref_to_frame.repeat(num_cams, 1, 1)

        cam_intrinsic = cam_intrinsic.repeat_interleave(num_frames, dim=0)
        world_to_cam = world_to_cam.repeat_interleave(num_frames, dim=0)
        cam_to_world = cam_to_world.repeat_interleave(num_frames, dim=0)

        
        VARIABLE_NAMES = ['I_0', 'I_45', 'I_90', 'I_135',
                          'I_0_visibility', 'I_45_visibility', 'I_90_visibility', 'I_135_visibility',
                          'I_vertex', 'I_normal',
                          'I_diffuse', 'I_specular', 'I_alpha', 'I_alpha_inv', 'I_mask', 'I_cluster',
                          'cam_intrinsic', 'world_to_cam', 'cam_to_world', 'ref_to_frame', 'frame_to_ref']
        
        self.sample_names = ['I_vertex', 'I_normal', 'I_mask', 'I_alpha', 'I_specular', 'I_diffuse']
        self.camera_param_names = ['cam_intrinsic', 'world_to_cam', 'cam_to_world', 'ref_to_frame', 'frame_to_ref']            
        
        self.h = h
        self.w = w
        self.dataset = edict()
        for name in VARIABLE_NAMES:
            self.dataset[name] = locals()[name]
        self.num_cams = num_cams
        self.num_frames = num_frames
        self.N = num_cams * num_frames

        self.I_normal = I_normal
        self.I_vertex = I_vertex

        self.select_index = s_i

        if args.general.test:
            cam_intrinsic_ref, world_to_cam_ref, cam_to_world_ref = get_camera_params(path, 3)
            self.I_diffuse_ref = I_diffuse_ref
            self.I_mask_ref = I_mask_ref
            self.dataset.cam_intrinsic_ref = cam_intrinsic_ref[2:, :, :].repeat_interleave(num_ref_frames, dim=0)
            self.dataset.world_to_cam_ref = world_to_cam_ref[2:, :, :].repeat_interleave(num_ref_frames, dim=0)
            self.dataset.cam_to_world_ref = cam_to_world_ref[2:, :, :].repeat_interleave(num_ref_frames, dim=0)
            self.dataset.num_ref_frame = num_ref_frames

            self.dataset.frame_to_ref_full = torch.zeros([num_ref_frames, 4, 4])
            self.dataset.ref_to_frame_full = torch.zeros([num_ref_frames, 4, 4])
            
            for idx, pose_filename in enumerate(icp_pose_filenames[ref_si]):
                icp_pose_full = np.loadtxt(os.path.join(icp_pose_path, pose_filename))
                self.dataset.frame_to_ref_full[idx, :, :] = torch.Tensor(icp_pose_full)
                self.dataset.ref_to_frame_full[idx, :, :] = torch.Tensor(np.linalg.inv(icp_pose_full))


    def __len__(self):
        return self.N
    

    def __getitem__(self, index):
        data = edict()

        for name in self.camera_param_names:
            data[name] = self.dataset[name][index]

        for name in self.sample_names:
            if (name == 'I_vertex') or (name == 'I_normal'):
                data[name] = self.dataset[name]
            else:
                data[name] = self.dataset[name][index]
        
        data.index = index
        data.cam_idx = index // self.num_frames

        return data
    


class PbrdfDatasetPatch(PbrdfDatasetBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.patch_size = args.data.patch_size

    
    def __len__(self):
        return int(self.h * self.w / self.patch_size / self.patch_size)
    
    
    def __getitem__(self, index):
        y0 = torch.randint(0, self.h - self.patch_size, (1,)).item()
        x0 = torch.randint(0, self.w - self.patch_size, (1,)).item()

        d_patch = edict()
        for name in self.sample_names:
            if (name == 'I_vertex') or (name == 'I_normal'):
                d_patch[name] = self.dataset[name][y0:y0+self.patch_size, x0:x0+self.patch_size, ...]
            else:
                d_patch[name] = self.dataset[name][:, y0:y0+self.patch_size, x0:x0+self.patch_size, ...]

        for name in self.camera_param_names:
            d_patch[name] = self.dataset[name]
        
        mask_patch = torch.zeros((self.h, self.w), dtype=torch.bool)
        mask_patch[y0:y0+self.patch_size, x0:x0+self.patch_size] = True
        d_patch['mask_patch'] = mask_patch

        return d_patch


class PbrdfDynamicDatasetBase(Dataset):
    def __init__(self, args):
        path = args.data.path
        h = args.data.h
        w = args.data.w
        num_cams = args.data.num_cams
        num_frames = args.data.num_frames
        texture_filename = args.data.texture_filename
        self.patch_size = args.data.patch_size

        texture_path = os.path.join(path, texture_filename)

        I_0, I_0_visibility = None, None
        I_45, I_45_visibility = None, None
        I_90, I_90_visibility = None, None
        I_135, I_135_visibility = None, None
        I_diffuse = None
        I_specular = None
        I_alpha = None

        white_bal = torch.Tensor(np.loadtxt(os.path.join(path, 'white_balance.txt')).astype(np.float32))

        I_0_path = os.path.join(texture_path, '0/exr')
        I_0, I_0_visibility, _, _, s_i = load_images_sequence_parallel(I_0_path, h, w, white_bal, args.data.start_frame, args.data.end_frame, num_cams)
        I_0_finite_mask = torch.all(torch.isfinite(I_0), dim=-1, keepdim=True)

        I_45_path = os.path.join(texture_path, '45/exr')
        I_45, I_45_visibility, _, _, _ = load_images_sequence_parallel(I_45_path, h, w, white_bal, args.data.start_frame, args.data.end_frame, num_cams, s_i)
        I_45_finite_mask = torch.all(torch.isfinite(I_45), dim=-1, keepdim=True)

        I_90_path = os.path.join(texture_path, '90/exr')
        I_90, I_90_visibility, _, _, _ = load_images_sequence_parallel(I_90_path, h, w, white_bal, args.data.start_frame, args.data.end_frame, num_cams, s_i)
        I_90_finite_mask = torch.all(torch.isfinite(I_90), dim=-1, keepdim=True)

        I_135_path = os.path.join(texture_path, '135/exr')
        I_135, I_135_visibility, _, _, _ = load_images_sequence_parallel(I_135_path, h, w, white_bal, args.data.start_frame, args.data.end_frame, num_cams, s_i)
        I_135_finite_mask = torch.all(torch.isfinite(I_135), dim=-1, keepdim=True)

        I_mask = torch.cat([I_0_visibility, I_45_visibility, I_90_visibility, I_135_visibility], dim=-1)

        I_mask = (I_mask > 0.5)
        I_mask = torch.all(I_mask, dim=-1, keepdim=True)
        I_mask = torch.cat([I_mask, I_0_finite_mask, I_45_finite_mask, I_90_finite_mask, I_135_finite_mask],
                        dim=-1)
        I_mask = torch.all(I_mask, dim=-1)

        I_0[~I_mask] = 0
        I_45[~I_mask] = 0
        I_90[~I_mask] = 0
        I_135[~I_mask] = 0

        I_vertex = load_images_glob_parallel(f'{texture_path}/tracked', 'vertex', h, w, num_frames, s_i)
        I_normal = load_images_glob_parallel(f'{texture_path}/tracked', 'normal', h, w, num_frames, s_i)

        I_vertex = I_vertex.repeat(num_cams, 1, 1, 1)
        I_normal = I_normal.repeat(num_cams, 1, 1, 1)

        I_diffuse = 2.0 * I_90
        I_specular = I_0 - I_90
        I_alpha = I_135 - I_45
        I_alpha_inv = I_45 - I_135

        frame_to_ref = torch.eye(4)
        ref_to_frame = torch.eye(4)

        frame_to_ref = frame_to_ref.repeat(num_cams * num_frames, 1, 1)
        ref_to_frame = ref_to_frame.repeat(num_cams * num_frames, 1, 1)

        cam_intrinsic, world_to_cam, cam_to_world = get_camera_params(path, num_cams)

        cam_intrinsic = cam_intrinsic.repeat_interleave(num_frames, dim=0)
        world_to_cam = world_to_cam.repeat_interleave(num_frames, dim=0)
        cam_to_world = cam_to_world.repeat_interleave(num_frames, dim=0)

        I_cluster = None

        VARIABLE_NAMES = ['I_0', 'I_45', 'I_90', 'I_135',
                          'I_0_visibility', 'I_45_visibility', 'I_90_visibility', 'I_135_visibility',
                          'I_vertex', 'I_normal',
                          'I_diffuse', 'I_specular', 'I_alpha', 'I_alpha_inv', 'I_mask', 'I_cluster',
                          'cam_intrinsic', 'world_to_cam', 'cam_to_world', 'ref_to_frame', 'frame_to_ref']
        
        self.sample_names = ['I_vertex', 'I_normal', 'I_mask', 'I_alpha', 'I_specular', 'I_diffuse']
        self.camera_param_names = ['cam_intrinsic', 'world_to_cam', 'cam_to_world', 'ref_to_frame', 'frame_to_ref']
        self.static_textures_name = ['tex_alpha_s', 'tex_alpha_ss', 'tex_rho_s', 'tex_rho_ss_0', 'tex_rho_ss_1',
                                     'tex_rho_d_0', 'tex_rho_d_1', 'tex_refrac_idx', 'tex_height']
        
        self.static_textures = edict()
        for name in self.static_textures_name:
            tex_img = cv2.imread(f'{texture_path}/{name}.exr', -1)
            if len(tex_img.shape) > 2:
                tex_img = tex_img[:, :, ::-1]
            else:
                tex_img = tex_img[:, :, None]
            self.static_textures[name] = torch.tensor(tex_img.copy())
        self.static_textures.tex_rho_d = torch.cat([self.static_textures.tex_rho_d_0, self.static_textures.tex_rho_d_1], dim=-1)
        self.static_textures.tex_rho_ss = torch.cat([self.static_textures.tex_rho_ss_0, self.static_textures.tex_rho_ss_1], dim=-1)
        self.static_textures.tex_height = torch.zeros_like(self.static_textures.tex_height)

        self.h = h
        self.w = w
        self.dataset = edict()
        for name in VARIABLE_NAMES:
            self.dataset[name] = locals()[name]
        for k, v in self.static_textures.items():
            self.dataset[k] = v
        self.num_cams = num_cams
        self.num_frames = num_frames
        self.N = num_cams * num_frames

        self.I_normal = I_normal
        self.I_vertex = I_vertex

    def __len__(self):
        return int(self.N * self.h * self.w // self.patch_size // self.patch_size)

    def __getitem__(self, index):
        num_patches = self.h * self.w // self.patch_size // self.patch_size
        total_idx = min(index // num_patches, self.num_cams * self.num_frames - 1)

        data = edict()

        data.index = index
        data.tota_idx = total_idx
        data.frame_idx = total_idx % self.num_frames
        data.cam_idx = total_idx // self.num_frames

        if self.h == self.patch_size: 
            y0 = 0
        else:
            y0 = torch.randint(0, self.h - self.patch_size, (1,)).item()
        if self.w == self.patch_size:
            x0 = 0
        else:
            x0 = torch.randint(400, self.w - self.patch_size - 400, (1,)).item()

        for name in self.camera_param_names:
            data[name] = self.dataset[name][total_idx]

        for name in self.sample_names:
            data[name] = self.dataset[name][total_idx, y0:y0+self.patch_size, x0:x0+self.patch_size, ...]
        
        for name in self.static_textures_name:
            data[name] = self.dataset[name][y0:y0+self.patch_size, x0:x0+self.patch_size, ...]

        data.tex_rho_ss = self.dataset.tex_rho_ss[y0:y0+self.patch_size, x0:x0+self.patch_size, 3*data.cam_idx:3*(data.cam_idx + 1)]
        data.tex_rho_d = self.dataset.tex_rho_d[y0:y0+self.patch_size, x0:x0+self.patch_size, 3*data.cam_idx:3*(data.cam_idx + 1)]

        mask_patch = torch.zeros((self.h, self.w), dtype=torch.bool)
        mask_patch[y0:y0+self.patch_size, x0:x0+self.patch_size] = True
        data['mask_patch'] = mask_patch

        return data
    
class PbrdfDynamicDatasetTest(PbrdfDynamicDatasetBase):
    def __init__(self, args):
        super().__init__(args)

    def __len__(self):
        return int(self.N)

    def __getitem__(self, index):
        data = edict()

        data.index = index
        data.tota_idx = index
        data.frame_idx = index % self.num_frames
        data.cam_idx = index // self.num_frames

        for name in self.camera_param_names:
            data[name] = self.dataset[name][index]

        for name in self.sample_names:
            data[name] = self.dataset[name][index]
        
        for name in self.static_textures_name:
            data[name] = self.dataset[name]

        return data
    
        

class PbrdfDatasetValidation(PbrdfDatasetBase):
    def __init__(self, args) -> None:
        path = args.data.path
        h = args.data.h
        w = args.data.w
        num_cams = args.data.num_cams
        num_frames = args.data.num_frames

        img_path = args.data.input_img_path
        white_bal = torch.Tensor(np.loadtxt(os.path.join(args.data.path, 'white_balance.txt')).astype(np.float32))
        d65_mat = torch.Tensor(np.loadtxt(os.path.join(args.data.path, 'd65_mat.txt')).astype(np.float32).reshape(-1, 3, 3))

        num_cams = args.data.num_cams
        num_frames = args.data.num_frames
        num_batch = args.data.batch_size
        start_idx = args.data.start_idx

        I_0, _, _, s_i = load_cam_images_parallel(img_path, 0, start_idx, white_bal, d65_mat, num_cams, num_batch)
        I_45, _, _, _ = load_cam_images_parallel(img_path, 45, start_idx, white_bal, d65_mat, num_cams, num_batch, s_i)
        I_90, _, _, _ = load_cam_images_parallel(img_path, 90, start_idx, white_bal, d65_mat, num_cams, num_batch, s_i)
        I_135, _, _, _ = load_cam_images_parallel(img_path, 135, start_idx, white_bal, d65_mat, num_cams, num_batch, s_i)

        self.dataset = edict()
        
        self.dataset.I_0_cam = I_0
        self.dataset.I_45_cam = I_45
        self.dataset.I_90_cam = I_90
        self.dataset.I_135_cam = I_135
        
        self.num_cams = num_cams
        self.num_frames = num_batch
        self.N = num_cams * num_batch

        self.select_index = s_i


    def __len__(self):
        return self.N
    

    def __getitem__(self, index):
        data = edict()
        
        data.index = index
        data.cam_idx = index // self.num_frames

        data['I_0_cam'] = self.dataset.I_0_cam[index]
        data['I_45_cam'] = self.dataset.I_45_cam[index]
        data['I_90_cam'] = self.dataset.I_90_cam[index]
        data['I_135_cam'] = self.dataset.I_135_cam[index]

        return data
    

if __name__=='__main__':
    path = '/root/local/data/polar-face/90'
    h = 1024
    w = 2048
    num_cams = 2
    num_frames = 2
    dataset = PbrdfDatasetBase(path, h, w, num_cams, num_frames)

