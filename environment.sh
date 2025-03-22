conda create -y --name pbssrdf python=3.11
conda activate pbssrdf
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -y imageio joblib matplotlib numpy scikit-image opt_einsum scipy tensorboard
pip install --no-input easydict tqdm opencv-python-headless opencv-contrib-python-headless chardet