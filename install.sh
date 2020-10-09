CUDA=${1:-"cu102"}
TORCH=${2:-"1.6.0"}
echo "CUDA: ${CUDA} / TORCH: ${TORCH}"
pip3 install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip3 install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip3 install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip3 install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip3 install torch-geometric
pip3 install -r requirements.txt
echo "Install completed"
