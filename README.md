# Polarimetric BSSRDF Acquisition of Dynamic Faces

#### [Project Page](https://vclab.kaist.ac.kr/siggraphasia2024/index.html) | [Paper](https://vclab.kaist.ac.kr/siggraphasia2024/polar-face-main.pdf) | [Supplemental material](https://vclab.kaist.ac.kr/siggraphasia2024/polar-face-supple_v2.pdf) | [Dataset](https://drive.google.com/drive/folders/1wTLj6-Rx7SUT7Yqj7Z3gVo3xUlyLMc4w?usp=drive_link)

[Hyunho Ha](https://sites.google.com/view/hyunhoha), [Inseung Hwang](https://sites.google.com/view/inseunghwang/), [Nestor Monzon](https://nestor98.github.io/), [Jaemin Cho](https://vclab.kaist.ac.kr/jmcho/index.html), [Donggun Kim](https://sites.google.com/view/dgkim03), [Seung-Hwan Baek](https://sites.google.com/view/shbaek/), [Adolfo Muñoz](https://webdiis.unizar.es/~amunoz/es/), [Diego Gutierrez](http://giga.cps.unizar.es/~diegog/), and [Min H. Kim](https://vclab.kaist.ac.kr/minhkim/)	 	 	 
 	
Institute: [KAIST Visual Computing Laboratory](https://vclab.kaist.ac.kr/index.html), [POSTECH](https://sites.google.com/view/shbaek/), and [Universidad de Zaragoza Graphics and Imaging Lab](https://graphics.unizar.es/)

If you use our code or dataset for your academic work, please cite our paper:
```
@Article{polarface:SIGA:2024,
  author  = {Hyunho Ha and Inseung Hwang and Nestor Monzon and Jaemin Cho
             and Donggun Kim and Seung-Hwan Baek and Adolfo Muñoz and
             Diego Gutierrez and Min H. Kim},
  title   = {Polarimetric BSSRDF Acquisition of Dynamic Faces},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2024)},
  year    = {2024},
  volume  = {43},
  number  = {6},
}
```

## Minimum Requirements for Hardware

Our code needs to read and process a large number of textures:
- Two cameras
- 200 frames for static and approximately 100 frames for dynamic
- 4 polarization states

With the high resolution of texture data (2000 $\times$ 2000), processing all the data requires at least **256GB RAM**. We tested the code on a **single RTX 4090 24GB** (Windows, minimum) and a **single A100 80GB** (Ubuntu, preferred).

If you don't have enough RAM memory, you could edit the `num_frames` parameter in `config/*.json`. When the GPU memory is insufficient, for static optimization, you can edit the `patch_size` parameter in `config/pface.json`. 
For dynamic optimization, you can edit both the `patch_size` and `num_group_frame` parameters in `config/pface_dynamic.json`. 

## Installation

Our implementation has been tested on Windows 10 and Ubuntu with the following configurations:
- Python 3.7, Pytorch 1.13.1, CUDA 11.6 (`docker_env/`)
- Python 3.11, Pytorch 2.5.1. CUDA 12.4 (`environment.sh`)

A Docker environment is also provided in the `docker_env` folder.

### Conda install
```bash
conda env create --file environment.yaml
conda activate pbssrdf
```

### Docker install

Ensure that Docker with CUDA support is properly installed. To build the Docker image, run the following commands:
```
cd docker_env
docker build -t pface:latest .
```
Update the `HOST_CODE_DIR` and `HOST_DATA_DIR` variables in the `docker_env/run_server.sh` to point to the polar-face-code and polar-face-dataset directories, respectively. Then, execute the following script to start the Docker container:
```
bash run_server.sh
```

## Dataset

Due to the policy regarding face data, we provide the author's face capture data in [here](https://drive.google.com/drive/folders/1wTLj6-Rx7SUT7Yqj7Z3gVo3xUlyLMc4w?usp=drive_link). The structure of our data is as follows:
```
$(DATASET_DIR)
└── $(PARTICIPANT)
    ├── 5_Static 
    │   └── 90
    │       ├── texture
    │       ├── icp-pose
    │       ├── *.ply
    │       ├── cameras.txt
    │       ├── images.txt
    │       ├── points3D.txt (unused)
    │       ├── color_calib.txt (Intensity normalization (unused))
    │       ├── ColorCalib.txt (Color calibration 3x3 matrix)
    │       ├── d65_mat.txt (Same as ColorCalib.txt but with index)
    │       ├── d65_mat_cam2.txt (Color calibration 3x3 matrix for cam1 and cam2 (unused))
    │       ├── d65_mat_cam2_white_balanced.txt (d65_mat_cam2.txt with white_balance.txt (unused))
    │       ├── spectralon.txt (Intensity value of spectralon)
    │       └── white_balance.txt (Normalized white balance with intensity using spectralon.txt)
    └── 6_Dynamic_1 $(DYNAMIC)_$(DYNAMIC_NUM)
        └── 90
            ├── texture
            ├── icp-pose
            └── *.txt (Similar to 5_Static)
```
We provide our preprocessed data under the static sequence `5_Static` and the dynamic sequence `6_Dynamic_1`. `cam1` to `cam3` contain linearly polarized images. For details about the camera setup, please refer to the [Hardware](#hardware) section. Note that `cam3` is the reference polarization camera, which we do not use in our optimization.


## Hardware

![hardware](./fig/Hardware.png)

For more details about the specifications of our cameras, please refer to our main paper.

## Usage

We provide an example script in `run_bash.sh`, which you can run the bash script as follows:
```bash
# bash run_bash.sh $(DATASET_DIR) $(PARTICIPANT) $(STATIC_MODULE) $(DYNAMIC_MODULE) $(DYNAMIC_NUM)
bash run_bash.sh "D:/Data/pface" "FaceData" "5_Static" "6_Dynamic" "1"
# When the RAM is less than 256 GB (Setting for 128 GB)
bash run_bash_small.sh "D:/Data/pface" "FaceData" "5_Static" "6_Dynamic" "1"
```
Running the bash script will generate the static results in `$(DATASET_DIR)/$(PARTICIPANT)/output/$(STATIC_MODULE)/pbrdf` and the dynamic results in `$(DATASET_DIR)/$(PARTICIPANT)/output/$(DYNAMIC_MODULE)_$(DYNAMIC_NUM)/pbrdf`

After `optimize.py` is done, `test.py` generates the result textures for the static scene in `$(DATASET_DIR)/$(PARTICIPANT)/output/$(STATIC_MODULE)/pbrdf/results/s1`. Similarly, after `optimize_dynamic.py` is done, `test_dynamic.py` generates the result textures for the dynamic scene in `$(DATASET_DIR)/$(PARTICIPANT)/output/$(DYNAMIC_MODULE)_$(DYNAMIC_NUM)/pbrdf/results/s6`.

## License

Hyunho Ha, Inseung Hwang, Nestor Monzon, Jaemin Cho, Donggun Kim, Seung-Hwan Baek, Adolfo Muñoz, Diego Gutierrez, and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:

Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.

The use of the software is for Non-Commercial purposes only as this Agreement uses "Non-Commercial Purpose" for education or research in a non-commercial organization only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service that is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].

Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY THE LICENSEE AS A RESULT OF USING, MODIFYING, OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.

Please refer to license.txt for more details. 

## Contact

If you have any questions, please feel free to contact us.

Hyunho Ha (vchhha615@gmail.com)

Min H. Kim (minhkim@kaist.ac.kr)
