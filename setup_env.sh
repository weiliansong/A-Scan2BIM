# for automatic conda environment setup
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda remove -y -n bim --all
conda create -y -n bim python=3.8
conda activate bim

conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt

cd code/learn/models/ops
python setup.py build install

conda install -y tqdm shapely
pip install tensorboard rtree shapely pytorch-metric-learning laspy[lazrs] open3d typer[all]