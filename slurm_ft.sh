script_name="ft_stabai"
cat >$script_name <<EOD
#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2               # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=4     # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:4            # Request N GPUs per machine
#SBATCH --mem=0
#SBATCH --time=0-02:00:00
#SBATCH --output=/data/fsx/$script_name-%j.out # Set the output file name
#SBATCH --error=/data/fsx/$script_name-%j.err   # Set the error file ame
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# Force IPv4
export NCCL_SOCKET_FAMILY=AF_INET
PATH=/home/.openmpi/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LD_LIBRARY_PATH=/home/.openmpi/lib:/lib/x86_64-linux-gnu:/opt/conda/lib:/usr/local/lib:/usr/local/lib:
export MPLCONFIGDIR=/data/fsx/mpl/.config/
export TRANSFORMERS_CACHE=/data/fsx/transformers/.cache/
source /opt/conda/etc/profile.d/conda.sh
conda activate litgpt
# srun python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# srun litgpt download NousResearch/Llama-3.2-1B --checkpoint_dir /data/fsx/checkpoints --model_name Llama-3.2-1B
srun litgpt finetune_lora /data/fsx/checkpoints/NousResearch/Llama-3.2-1B  --data Alpaca --data.download_dir /data/fsx/ftdata  --data.val_split_fraction 0.1  --train.epochs 1  --out_dir /data/fsx/out/llama-3.2-finetuned  --precision bf16-true
EOD

chmod +x $script_name
sbatch $script_name