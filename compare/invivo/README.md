


conda create -n invivo python=3.6
conda activate invivo
sh ./setup_env.sh

TreFiDe
git clone https://github.com/ikinsella/trefide.git
cp preprocess.py trefide/trefide/
cd trefide
make
pip install .


git clone https://github.com/adamcohenlab/invivo-imaging.git
if the env name is different from invivo, edit invivo-imaging/denoise/main.m

aslo modify Line 111 of denoise.py as follows:
disc_idx = np.array([], dtype=int)    

