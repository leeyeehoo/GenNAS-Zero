conda create -n gennaszero python=3.9
source activate gennaszero
# https://pytorch.org/get-started/previous-versions/ you might change the torch cuda version
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install pyyaml scipy matplotlib ptflops einops yacs simplejson colorlover plotly thop 
