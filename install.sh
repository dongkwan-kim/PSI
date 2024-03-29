CUDA=${1:-"cu102"}
TORCH=${2:-"1.6.0"}
echo "CUDA: ${CUDA} / TORCH: ${TORCH}"
pip3 install torch==${TORCH} -f https://download.pytorch.org/whl/${CUDA}/torch_stable.html
pip3 install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-sparse==0.6.7 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-cluster==1.5.7 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-geometric==1.6.1
pip3 install -r requirements.txt
echo "Install completed"
