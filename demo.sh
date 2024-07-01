# python inference_all.py 
# CUDA_VISIBLE_DEVICES=6 
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 29501 inference_all.py 
# demo-k8s.sh
# torchrun --nproc_per_node 1 --master_port 25649 inference_all.py 