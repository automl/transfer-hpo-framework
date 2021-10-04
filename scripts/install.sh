conda create -n rgpe -y
source activate rgpe
which conda
conda env list
conda install python=3.7 numpy=1.18.1 scipy=1.4.1 scikit-learn=0.22.1 gxx_linux-64 gcc_linux-64 \
    swig cython=0.29.13 ipython jupyter matplotlib pandas=0.25 -y
pip install ConfigSpace==0.4.11 pyrfr==0.8.0
pip install git+https://github.com/automl/HPOlib1.5@0449121d31e0dcd4f63435ba5b27a0dee5bbd55f --no-deps
pip install smac[all]==0.12.3
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install botorch==0.2.5
pip install lockfile
