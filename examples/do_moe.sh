#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Runs the "175B" parameter model
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=localhost
export MASTER_PORT=7766
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/etc/nccl_debug_file

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


CHECKPOINT_PATH=/root/checkpoint
TENSORBOARD_LOGS_PATH=/root/tensorboard
VOCAB_FILE=/jizhicfs/pinxuezhao/small_dataset/gpt2-vocab.json
MERGE_FILE=/jizhicfs/pinxuezhao/small_dataset/gpt2-merges.txt
DATA_PATH=/jizhicfs/pinxuezhao/small_dataset/lambada_data_text_document

export TORCH_NCCL_AVOID_RECORD_STREAMS=1

#SEQLEN=212992
#SEQLEN=131072
#SEQLEN=65536
#SEQLEN=32768
#SEQLEN=16384
SEQLEN=$((2*1024))

#--moe-pad-expert-input-to-capacity
#    --moe-expert-capacity-factor 2
MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 8
    --moe-grouped-gemm
    --disable-bias-linear
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type lshalltoall
    --moe-pad-expert-input-to-capacity
    --moe-expert-capacity-factor 2
)

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32 
    --seq-length $SEQLEN
    --max-position-embeddings 32768
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 8
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.999 
    --init-method-std 0.002
    --clip-grad 1.0 
    --bf16
    --lr 0.00015
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-warmup-fraction .01 
    --lr-decay-iters 320000 
    --use-mcore-models
    --position-embedding-type rope
)

RECOMP_ARGS=(
    --recompute-granularity full        
    --recompute-method uniform
    --recompute-num-layers 1
)

TE_ARGS=(
    --transformer-impl transformer_engine        
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
    --context-parallel-size 1
	--pipeline-model-parallel-size 1 
    --sequence-parallel 
    --use-distributed-optimizer
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --no-create-attention-mask-in-dataloader
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 100000000 
    --eval-interval 100000000
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 100000
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

NP=16
HOSTFILE=/jizhicfs/pinxuezhao/hostfile_2node

mpirun --allow-run-as-root --np $NP --hostfile $HOSTFILE \
       --mca btl_tcp_if_include bond1 \
       --bind-to numa \
        --map-by ppr:8:node \
        -mca coll_hcoll_enable 0 \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_SOCKET_IFNAME=bond1 \
        -x GLOO_SOCKET_IFNAME=bond1 \
        -x NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 \
        -x NCCL_NET_GDR_LEVEL=2 \
    -x NCCL_NET_GDR_READ=1 \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_IB_TC=160 \
        -x NCCL_PXN_DISABLE=0 \
        -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_IB_TIMEOUT=22 \
        -x MASTER_ADDR=30.207.99.83 \
         -x MASTER_PORT=7788 \
       bash /jizhicfs/pinxuezhao/pytorch2.4_conda.sh \
       python3 ../pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TE_ARGS[@]} \
    ${RECOMP_ARGS[@]} \
    ${MOE_ARGS[@]} 

