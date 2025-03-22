source "/opt/conda/etc/profile.d/conda.sh"

# conda packages from default channel
conda install -y \
    imageio \
    joblib \
    matplotlib \
    numpy \
    scikit-image \
    opt_einsum \
    scipy \
    tensorboard

# pypi packages
pip install --no-input \
    easydict \
    tqdm \
    opencv-python-headless \
    opencv-contrib-python-headless \
    chardet