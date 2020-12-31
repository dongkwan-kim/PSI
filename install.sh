CUDA=${1:-"cu102"}
TORCH=${2:-"1.6.0"}
echo "CUDA: ${CUDA} / TORCH: ${TORCH}"
pip3 install torch==${TORCH}
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-geometric
pip3 install -r requirements.txt
echo "Install completed"
