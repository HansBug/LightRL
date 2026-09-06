# GLM-5.1 (744B-A40B) model args for the Megatron bridge path.
#
# The architecture mirrors the zai-org/GLM-5.1 HF config (3 dense + 75 MoE
# layers, MLA + DSA indexer attention, 256 routed experts).  With
# --megatron-to-hf-mode bridge the mbridge glm_moe_dsa plugin re-derives these
# from the HF config; keep both in sync when upgrading the checkpoint.
MODEL_ARGS=(
   --disable-bias-linear
   --qk-layernorm
   --multi-latent-attention
   --experimental-attention-variant dsa
   --num-attention-heads 64
   --num-layers 78
   --hidden-size 6144
   --ffn-hidden-size 12288
   --max-position-embeddings 202752
   --normalization RMSNorm
   --position-embedding-type rope
   --rotary-percent 1.0
   --rotary-base 1000000
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 154880
   --q-lora-rank 2048
   --kv-lora-rank 512
   --qk-head-dim 256
   --qk-pos-emb-head-dim 64
   --v-head-dim 256
   --dsa-indexer-n-heads 32
   --dsa-indexer-head-dim 128
   --dsa-indexer-topk 2048
   --moe-ffn-hidden-size 2048
   --moe-shared-expert-intermediate-size 2048
   --moe-router-pre-softmax
   --moe-router-score-function sigmoid
   --moe-router-enable-expert-bias
   --moe-router-bias-update-rate 0
   --moe-router-load-balancing-type none
   --moe-router-topk 8
   --moe-router-topk-scaling-factor 2.5
   --moe-layer-freq "[0]*3+[1]*75"
   --num-experts 256
   --moe-grouped-gemm
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0
)
