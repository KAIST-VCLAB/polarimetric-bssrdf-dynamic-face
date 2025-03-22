import torch
import numpy as np
import cv2
from utils.plot import *
import matplotlib
from typing import Union, Optional


def compute_dop(dataset, rear_str=''):
    s1 = torch.mean(dataset[f'I_0{rear_str}'] - dataset[f'I_90{rear_str}'], dim=-1, keepdim=True)
    s2 = torch.mean(dataset[f'I_45{rear_str}'] - dataset[f'I_135{rear_str}'], dim=-1, keepdim=True)
    s0 = torch.mean((dataset[f'I_0{rear_str}'] + dataset[f'I_90{rear_str}'] + dataset[f'I_45{rear_str}'] + dataset[f'I_135{rear_str}']) / 2.0, dim=-1, keepdim=True)
    dop = torch.sqrt(s1 * s1 + s2 * s2) / s0
    if len(rear_str) == 0:
        dop[~dataset.I_mask] = 0
    dop[s0 <= 0] = 0

    return dop


def compute_aolp(dataset, rear_str=''):
    s1 = torch.mean(dataset[f'I_0{rear_str}'] - dataset[f'I_90{rear_str}'], dim=-1, keepdim=True)
    s2 = torch.mean(dataset[f'I_45{rear_str}'] - dataset[f'I_135{rear_str}'], dim=-1, keepdim=True)
    
    return torch.fmod(0.5 * torch.atan2(s2, s1), np.pi)


def applyColorMap(x: np.ndarray, colormap: Union[str, np.ndarray], vmin: float = 0.0, vmax: float = 255.0) -> np.ndarray:
    x_normalized = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0) 
    x_normalized_u8 = (255 * x_normalized).astype(np.uint8) 

    if isinstance(colormap, (str, matplotlib.colors.Colormap)):
        cmap = matplotlib.cm.get_cmap(colormap, 256)
        lut = cmap(range(256)) 
        lut = lut[:, :3]  
        lut = lut[:, ::-1]  
        lut_u8 = np.clip(255 * lut, 0, 255).astype(np.uint8)  
    elif isinstance(colormap, np.ndarray) and colormap.shape == (256, 3) and colormap.dtype == np.uint8:
        lut_u8 = colormap
    else:
        raise TypeError(f"'colormap' must be 'str' or 'np.ndarray ((256, 3), np.uint8)'.")

    x_colored = lut_u8[x_normalized_u8] 

    return x_colored


def applyColorToAoLP(aolp: np.ndarray, saturation: Union[float, np.ndarray] = 1.0, value: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    ones = np.ones_like(aolp)

    hue = (np.mod(aolp, np.pi) / np.pi * 179).astype(np.uint8) 
    saturation = np.clip(ones * saturation * 255, 0, 255).astype(np.uint8)
    value = np.clip(ones * value * 255, 0, 255).astype(np.uint8)

    hsv =  np.concatenate([hue, saturation, value], axis=-1)
    aolp_colored = np.zeros_like(hsv)
    for i in range(aolp_colored.shape[0]):
        aolp_colored[i] = cv2.cvtColor(hsv[i], cv2.COLOR_HSV2RGB)
    return aolp_colored.astype(np.float32) / 255.0


def main():
    pass


if __name__ == '__main__':
    main()

