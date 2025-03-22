import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import cv2
import torch.nn.functional as F


def grab_images(num_cams, num_frames, d, ss, has_ir=False):
    _, h, w, _ = d.shape
    d3 = torch.zeros([num_cams * num_frames, h, w, 3], dtype=d.dtype, device=d.device)
    ss3 = torch.zeros([num_cams * num_frames, h, w, 3], dtype=ss.dtype, device=ss.device)

    d3[:num_frames, :, :, :] = d[:num_frames, :, :, :3]
    ss3[:num_frames, :, :, :] = ss[:num_frames, :, :, :3]

    if not has_ir:
        d3[num_frames:2 * num_frames, :, :, :] = d[num_frames:2 * num_frames, :, :, 3:6]
        ss3[num_frames:2 * num_frames, :, :, :] = ss[num_frames:2 * num_frames, :, :, 3:6]
    else:
        d3[num_frames:2 * num_frames, :, :, 6] = d[:num_frames, :, :, 6]
        ss3[num_frames:2 * num_frames, :, :, 6] = ss[:num_frames, :, :, 6]

        d3[2 * num_frames:3 * num_frames, :, :, :] = d[2 * num_frames:3 * num_frames, :, :, 3:6]
        ss3[2 * num_frames:3 * num_frames, :, :, :] = ss[2 * num_frames:3 * num_frames, :, :, 3:6]

    return d3, ss3


def grab_images_by_index(cam_idx, d, ss):
    n, h, w, _ = d.shape
    d3 = torch.zeros([n, h, w, 3], dtype=d.dtype, device=d.device)
    ss3 = torch.zeros([n, h, w, 3], dtype=ss.dtype, device=ss.device)

    is_cam1 = (cam_idx == 0)
    d3[is_cam1] = d[is_cam1][:, :, :, :3]
    d3[~is_cam1] = d[~is_cam1][:, :, :, 3:6]
    ss3[is_cam1] = ss[is_cam1][:, :, :, :3]
    ss3[~is_cam1] = ss[~is_cam1][:, :, :, 3:6]

    return d3, ss3


def compute_D_GGX(alpha, theta_h):
    cos_hn = torch.cos(theta_h)
    tan_hn = torch.tan(theta_h)
    alpha2 = alpha * alpha
    tan_hn2 = tan_hn * tan_hn

    return alpha2 / (torch.pi * (cos_hn ** 4) * (alpha2 + tan_hn2) * (alpha2 + tan_hn2))


def compute_D_Riviere(alpha, ndoth):
    return alpha * 14.0 * torch.pow(ndoth, 12.0) / (2.0 * torch.pi) + (1 - alpha) * 50.0 * torch.pow(ndoth, 48.0) / (2 * torch.pi)


def compute_G_smith(alpha, theta_i, theta_o):
    tan_nv = torch.tan(theta_o)
    tan_ni = torch.tan(theta_i)
    alpha2 = alpha * alpha

    Gi = 2.0 / (1.0 + torch.sqrt(1.0 + alpha2 * tan_ni * tan_ni))
    Go = 2.0 / (1.0 + torch.sqrt(1.0 + alpha2 * tan_nv * tan_nv))

    return Gi * Go


def compute_G_Riviere(ndoth, ndotv, ndotl, hdotv):
    G_ones = torch.ones_like(ndoth)

    G1 = 2.0 * ndoth * ndotv / hdotv
    G2 = 2.0 * ndoth * ndotl / hdotv

    G = torch.min(torch.min(G1, G2), G_ones)
    
    return G


def compute_Fresnel(cos_theta, n1, n2, is_exitant_angle=False):
    if not is_exitant_angle:
        cos_theta_i = cos_theta
        sin_theta_i = torch.sqrt(1.0 - cos_theta_i * cos_theta_i + 1e-16)
    else:
        cos_theta_e = cos_theta
        sin_theta_e = torch.sqrt(1.0 - cos_theta_e * cos_theta_e + 1e-16)
        sin_theta_i = (n2 / n1) * sin_theta_e
        cos_theta_i = torch.sqrt(1.0 - sin_theta_i * sin_theta_i + 1e-16)

    sq_one_m_refrac_sin_theta_i2 = torch.sqrt(1.0 - (n1 / n2) * sin_theta_i * (n1 / n2) * sin_theta_i + 1e-16)
    Rs_numer = n1 * cos_theta_i - n2 * sq_one_m_refrac_sin_theta_i2
    Rs_denom = n1 * cos_theta_i + n2 * sq_one_m_refrac_sin_theta_i2
    Rs = Rs_numer * Rs_numer / (Rs_denom * Rs_denom)

    Rp_numer = n1 * sq_one_m_refrac_sin_theta_i2 - n2 * cos_theta_i
    Rp_denom = n1 * sq_one_m_refrac_sin_theta_i2 + n2 * cos_theta_i
    Rp = Rp_numer * Rp_numer / (Rp_denom * Rp_denom)

    Ts = 1.0 - Rs
    Tp = 1.0 - Rp

    return Rs, Rp, Ts, Tp


def compute_DoP(zenith, eta):
    sin_zenith = torch.sin(zenith)
    cos_zenith = torch.cos(zenith)
    eta2 = eta * eta
    sin_zenith2 = sin_zenith * sin_zenith

    DoP_numen = (eta - 1.0 / eta) * (eta - 1.0 / eta) * sin_zenith2
    DoP_denom = 2.0 + 2.0 * eta2 - (eta + 1.0 / eta) * (eta + 1.0 / eta) * sin_zenith2 + 4.0 * cos_zenith * \
                torch.sqrt(eta2 - sin_zenith2 + 1e-16)

    return DoP_numen / DoP_denom


def compute_DoP_mask(zenith, eta, mask):
    sin_zenith = torch.sin(zenith)
    cos_zenith = torch.cos(zenith)
    eta2 = eta * eta
    sin_zenith2 = sin_zenith * sin_zenith

    DoP_numen = (eta - 1.0 / eta) * (eta - 1.0 / eta) * sin_zenith2
    DoP_denom = 2.0 + 2.0 * eta2 - (eta + 1.0 / eta) * (eta + 1.0 / eta) * sin_zenith2 + 4.0 * cos_zenith * \
                torch.sqrt(eta2 - sin_zenith2 + 1e-16)

    DoP = torch.zeros_like(DoP_numen)
    DoP[mask] = DoP_numen[mask] / DoP_denom[mask]

    return DoP


def compute_angle(target, v_dirs, y_dirs, mask):
    z_o = F.normalize(v_dirs, dim=-1)
    y_o = F.normalize(y_dirs - (y_dirs * z_o).sum(dim=-1, keepdim=True) * z_o, dim=-1)
    x_o = F.normalize(torch.cross(y_o, z_o, dim=-1), dim=-1)
    sin_azimuth_numer = (y_o * target).sum(dim=-1, keepdim=True) 
    cos_azimuth_numer = (x_o * target).sum(dim=-1, keepdim=True) 

    sin_azimuth = torch.zeros_like(sin_azimuth_numer)
    cos_azimuth = torch.zeros_like(cos_azimuth_numer)

    azimuth_denom = torch.sqrt(sin_azimuth_numer * sin_azimuth_numer + cos_azimuth_numer * cos_azimuth_numer + 1e-16)
    sin_azimuth[mask] = sin_azimuth_numer[mask] / azimuth_denom[mask]
    cos_azimuth[mask] = cos_azimuth_numer[mask] / azimuth_denom[mask]

    alpha_o = torch.zeros_like(sin_azimuth)
    beta_o = torch.zeros_like(cos_azimuth)
    alpha_o[mask] = (2.0 * sin_azimuth * cos_azimuth)[mask]
    beta_o[mask] = (2.0 * cos_azimuth * cos_azimuth - 1.0)[mask]

    return alpha_o, beta_o 




def compute_height_normal(vertmap, normalmap_mesh, heightmap, mask=None):
    h, w, c = vertmap.shape
    result_map = normalmap_mesh.clone()
    heightmap_normal = normalmap_mesh[1:-1, 1:-1, :].clone().reshape(-1, c)[..., None]

    tu = (vertmap[1:-1, 2:, :] - vertmap[1:-1, :-2, :]).reshape(-1, c)
    tv = (vertmap[2:, 1:-1, :] - vertmap[:-2, 1:-1, :]).reshape(-1, c)
    su = torch.norm(tu, dim=1, keepdim=True)
    sv = torch.norm(tv, dim=1, keepdim=True)

    world_n = normalmap_mesh[1:-1, 1:-1, :].reshape(-1, c) * su * sv

    du = (heightmap[1:-1, 2:] - heightmap[1:-1, :-2]).reshape(-1)
    dv = (heightmap[2:, 1:-1] - heightmap[:-2, 1:-1]).reshape(-1)

    TBN = torch.cat([tu, tv, world_n], dim=1).reshape(-1, c, c).permute(0, 2, 1) 
    one_tensor = torch.ones_like(du)
    duv = torch.stack([-du, -dv, one_tensor], dim=1)[..., None]  

    height_n = torch.bmm(TBN, duv) 

    norm_n = torch.norm(height_n, dim=1, keepdim=True).tile(1, c, 1)  
    mask_n = (norm_n != 0)
    heightmap_normal[mask_n] = height_n[mask_n] / norm_n[mask_n]

    result_map[1:h-1, 1:w-1, :] = heightmap_normal.reshape(h-2, w-2, c)

    return result_map


def compute_batch_height_normal(vertmap, normalmap_mesh, heightmap, mask=None):
    n, h, w, c = vertmap.shape
    result_map = normalmap_mesh.clone()
    heightmap_normal = normalmap_mesh[:, 1:-1, 1:-1, :].clone().reshape(-1, c)[..., None]

    tu = (vertmap[:, 1:-1, 2:, :] - vertmap[:, 1:-1, :-2, :]).reshape(-1, c)
    tv = (vertmap[:, 2:, 1:-1, :] - vertmap[:, :-2, 1:-1, :]).reshape(-1, c)
    su = torch.norm(tu, dim=1, keepdim=True)
    sv = torch.norm(tv, dim=1, keepdim=True)

    world_n = normalmap_mesh[:, 1:-1, 1:-1, :].reshape(-1, c) * su * sv

    du = (heightmap[:, 1:-1, 2:] - heightmap[:, 1:-1, :-2]).reshape(-1)
    dv = (heightmap[:, 2:, 1:-1] - heightmap[:, :-2, 1:-1]).reshape(-1)

    TBN = torch.cat([tu, tv, world_n], dim=1).reshape(-1, c, c).permute(0, 2, 1) 
    one_tensor = torch.ones_like(du)
    duv = torch.stack([-du, -dv, one_tensor], dim=1)[..., None] 

    height_n = torch.bmm(TBN, duv)

    norm_n = torch.norm(height_n, dim=1, keepdim=True).tile(1, c, 1) 
    mask_n = (norm_n != 0)
    heightmap_normal[mask_n] = height_n[mask_n] / norm_n[mask_n]

    result_map[:, 1:h-1, 1:w-1, :] = heightmap_normal.reshape(n, h-2, w-2, c)

    return result_map

def compute_height_vertex(vertmap, normalmap_mesh, heightmap, mask=None):
    return vertmap + normalmap_mesh * heightmap


def compute_vertex_normal(vertmap):
    h, w, c = vertmap.shape

    result_map = torch.zeros_like(vertmap)
    tu = (vertmap[1:-1, 2:, :] - vertmap[1:-1, :-2, :]).reshape(-1, c)
    tv = (vertmap[2:, 1:-1, :] - vertmap[:-2, 1:-1, :]).reshape(-1, c)

    n = torch.cross(tv, tu, dim=-1).reshape(h-2, w-2, c)
    n = F.normalize(n, dim=-1)
    n = torch.nan_to_num(n, nan=0, posinf=0, neginf=0)

    result_map[1:h-1, 1:w-1] = n

    return result_map


def export_texture(path, dataset, texture, idx):
    for name in texture.name:
        img = getattr(texture, name).cpu().detach().numpy()
        if img.shape[-1] > 1:
            img1 = img[:, :, 2::-1]
            img3 = img[:, :, 5:2:-1]
            cv2.imwrite(os.path.join(path, f'{name}1_{idx}.exr'), img1)
            cv2.imwrite(os.path.join(path, f'{name}3_{idx}.exr'), img3)
        else:
            cv2.imwrite(os.path.join(path, f'{name}_{idx}.exr'), img)

    height_normal_map = compute_height_normal(dataset['I_vertex'].cuda(), dataset['I_normal'].cuda(), texture.height)
    cv2.imwrite('./results/normal_{0}.exr'.format(idx), height_normal_map.cpu().detach().numpy()[:, :, ::-1])