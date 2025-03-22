from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_conv2d(x):
    x = torch.permute(x, (0, 3, 1, 2))
    n, c, h, w = x.shape
    laplacian = torch.Tensor([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
    laplacian = laplacian.type(x.type())

    if x.is_cuda:
        laplacian = laplacian.cuda()
    laplacian = laplacian[None, None, ...].repeat(c, 1, 1, 1)

    if len(x.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
    y = F.conv2d(x, weight=laplacian, stride=1, padding=1, groups=c)

    return y.permute(0, 2, 3, 1)


def mean_conv2d(x, ksize):
    x = x.unsqueeze(0)
    kernel_width = ksize * 2 + 1
    kernel_weight = kernel_width * kernel_width - 1
    x = x.permute(3, 0, 1, 2)
    x = F.pad(x, (ksize, ksize, ksize, ksize), 'reflect')
    mean_kernel = torch.ones(kernel_width, kernel_width)[None, None, :, :].float().to(x.device)
    mean_kernel = mean_kernel / kernel_weight
    mean_kernel[:, :, ksize, ksize] = 0
    mean_value = F.conv2d(x, mean_kernel)
    return mean_value.permute(1, 2, 3, 0).squeeze(0)


def convert_intensity(x):
    intensity = 0.2989 * x[:, :, 0] + 0.5870 * x[:, :, 1] +  0.1140 * x[:, :, 2]
    return intensity.unsqueeze(-1)


def convert_intensity1d(x):
    intensity = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] +  0.1140 * x[:, 2]
    return intensity.unsqueeze(-1)


def pbrdf_loss(predictions:edict, targets:edict, args:edict):
    loss_dict = edict()
    mse = nn.MSELoss(reduction='sum')
    mask = predictions.mask_all
    texture_mask = torch.any(mask, dim=0)

    if args.w_diffuse:
        loss_dict.diffuse = mse(predictions.diffuse[mask], targets.I_diffuse[mask])
    
    if args.w_specular:
        loss_dict.specular = mse((predictions.specular + predictions.single_scattering)[mask], targets.I_specular[mask])

    if args.w_temporal_height:
        initial_heights = torch.zeros_like(predictions.height)
        loss_dict.temporal_height = mse(predictions.height[texture_mask], initial_heights[texture_mask])

    if args.w_laplacian_height:
        laplacian_height = laplacian_conv2d(predictions.height.unsqueeze(0))
        laplacian_zeros = torch.zeros_like(laplacian_height)
        loss_dict.laplacian_height = mse(laplacian_height, laplacian_zeros)
    
    if args.w_refrac_idx:
        mean_refrac_idx = mean_conv2d(predictions.refrac_idx, 2)
        loss_dict.refrac_idx = mse(mean_refrac_idx[texture_mask], predictions.refrac_idx[texture_mask])

    if args.w_alpha_s:
        mean_alpha_s = mean_conv2d(predictions.alpha_s, 2)
        loss_dict.alpha_s = mse(mean_alpha_s[texture_mask], predictions.alpha_s[texture_mask])
    
    if args.w_alpha_ss:
        mean_alpha_ss = mean_conv2d(predictions.alpha_ss, 2)
        loss_dict.alpha_ss = mse(mean_alpha_ss[texture_mask], predictions.alpha_ss[texture_mask])
    
    if args.w_smooth_rho_d:
        mean_rho_d = mean_conv2d(predictions.rho_d, 1)
        loss_dict.smooth_rho_d = mse(mean_rho_d[texture_mask], predictions.rho_d[texture_mask])

    if args.w_azimuthal:
        loss_dict.azimuthal = mse(predictions.I_alpha_mono[mask], predictions.alpha_mono[mask]) \
                              + mse(predictions.I_beta_mono[mask], predictions.beta_mono[mask])
    
    if args.w_DoP:
        loss_dict.DoP = mse(predictions.DoP[predictions.mask_dop_full], predictions.I_DoP[predictions.mask_dop_full])



    loss = 0.0
    loss_items = edict()
    for k, v in loss_dict.items():
        loss += args[f'w_{k}'] * v
        loss_items[k] = v.detach().clone()
    loss_items['all'] = loss.detach().clone()

    return loss, loss_items

def pbrdf_dynamic_loss(predictions:edict, targets:edict, args:edict):
    loss_dict = edict()
    mse = nn.MSELoss(reduction='sum')
    mask = predictions.mask_all
    texture_mask = torch.any(mask, dim=0)

    if args.w_diffuse:
        loss_dict.diffuse = mse(predictions.diffuse[mask], targets.I_diffuse[mask])
    
    if args.w_specular:
        loss_dict.specular = mse((predictions.specular + predictions.single_scattering)[mask], targets.I_specular[mask])
   
    if args.w_azimuthal:
        loss_dict.azimuthal = mse(predictions.I_alpha_mono[mask], predictions.alpha_mono[mask]) \
                              + mse(predictions.I_beta_mono[mask], predictions.beta_mono[mask])
    
    if args.w_DoP:
        loss_dict.DoP = mse(predictions.DoP[mask], predictions.I_DoP[mask])

    if args.w_temporal_height:
        loss_dict.temporal_height = mse(predictions.height[mask], targets.tex_height[mask])

    if args.w_laplacian_height:
        laplacian_height = laplacian_conv2d(predictions.height)
        laplacian_zeros = torch.zeros_like(laplacian_height)
        loss_dict.laplacian_height = mse(laplacian_height, laplacian_zeros)

    if args.w_temporal_rho_s:
        loss_dict.temporal_rho_s = mse(predictions.rho_s[mask], predictions.prev_rho_s[mask])
    
    if args.w_temporal_rho_ss:
        loss_dict.temporal_rho_ss = mse(predictions.rho_ss[mask], predictions.prev_rho_ss[mask])
    
    if args.w_temporal_init_rho_d:
        with torch.no_grad():
            inten_pred_rho_d = convert_intensity1d(predictions.rho_d[mask].detach())
            inten_tar_rho_d = convert_intensity1d(predictions.rho_d_initial[mask].detach())
            weight_intensity = torch.sqrt(torch.abs(inten_pred_rho_d - inten_tar_rho_d))
        loss_dict.temporal_init_rho_d = mse(weight_intensity * predictions.rho_d[mask], weight_intensity * predictions.rho_d_initial[mask])

    loss = 0.0
    loss_items = edict()
    for k, v in loss_dict.items():
        loss += args[f'w_{k}'] * v
        loss_items[k] = v.detach().clone()
    loss_items['all'] = loss.detach().clone()

    return loss, loss_items
