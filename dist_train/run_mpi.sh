PYTHON=/usr/bin/python
LD_LIBRARY_PATH="/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

NNODES=${#ADDR_LIST[@]}
MASTER_ADDR=${ADDR_LIST[0]}

for a2a_ffn_overlap_degree in 2; do
        mpiexec -x PATH=$PATH -x CUDA_HOME=/usr/local/cuda/ -x NCCL_DEBUG=WARN -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x MASTER_ADDR=ethgpu9 -x LOCAL_SIZE=4 --prefix /usr/local/ --host ethgpu9,ethgpu10 -bind-to none $PYTHON launch.py pre_test.py --a2a_ffn_overlap_degree=$a2a_ffn_overlap_degree --log='test.log'
        sleep 5s
done
