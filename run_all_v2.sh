set -euo pipefail

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  1. 路徑設定                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
DATA_ROOT="C:/Users/user/Desktop/ARCADE"
SAM3_CHECKPOINT="C:/Users/user/Desktop/SAM3gnn_net/sam3_weights/sam3.pt"
SAVE_DIR="./checkpoints_v2"
RESULT_DIR="./results_v2"
VESSELS="LAD,LCx,RCA"

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  2. 訓練模式                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
PER_VESSEL=false

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  3. Train/Val 分割                                                     │
# └─────────────────────────────────────────────────────────────────────────┘
MERGE_SPLIT=true
VAL_RATIO=0.1
SPLIT_SEED=67

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  4. 訓練超參                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
IMG_SIZE=512
EPOCHS=100
BATCH=6
ACCUM_STEPS=1
LR=1e-4
SCALE_LR=false
WORKERS=0
SAVE_EVERY=10
RESUME=""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  5. LR 排程                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
WARMUP_EPOCHS=5

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  6. Backbone                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
UNFREEZE=true
BACKBONE_LR_SCALE=0.01

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  7. 模組開關                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
USE_SEMANTIC_PROMPT=true
USE_SPARSE_GAT=true
USE_REID=true

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  8. SparseGAT 參數                                                     │
# └─────────────────────────────────────────────────────────────────────────┘
GAT_LAYERS=2
GAT_HEADS=4
K_NEIGHBORS=16
MAX_NODES=4096
NODE_THRESHOLD=0.3

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  9. 密集 GNN 後備                                                     │
# └─────────────────────────────────────────────────────────────────────────┘
GNN_ITERS=3

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  10. 語意提示 & ReID                                                   │
# └─────────────────────────────────────────────────────────────────────────┘
N_PROMPT_TOKENS=8
REID_EMBED_DIM=128
LAMBDA_REID=0.1

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  11. 偽影增強                                                          │
# └─────────────────────────────────────────────────────────────────────────┘
ARTIFACT_PROB=0           #訓練資料帶偽影（導管/導絲/縫線）

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  12. 評估設定                                                          │
# └─────────────────────────────────────────────────────────────────────────┘
CLDICE_ITER=10
PP_MIN_SIZE=50
N_VIS=20
TOP_K=4

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  13. 執行環境                                                          │
# └─────────────────────────────────────────────────────────────────────────┘
USE_AMP=true
USE_COMPILE=false
NO_CACHE=false


# =============================================================================
#  環境檢查
# =============================================================================
echo "================================================"
echo " SAM3 UNet V2"
echo " SemanticPrompt + SparseGAT + ReID"
echo " + Bezier Artifact Aug"
echo "================================================"
python -c "
import torch, sam3
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
try:
    import torch_geometric
    print(f'PyG     : {torch_geometric.__version__}')
except ImportError:
    print('PyG     : NOT INSTALLED (dense GNN fallback)')
" 2>/dev/null || { echo "[ERROR] PyTorch/SAM3 not found."; exit 1; }
python -c "import scipy"   2>/dev/null || { echo "[ERROR] SciPy not found."; exit 1; }
python -c "import cv2"     2>/dev/null || { echo "[ERROR] opencv-python not found."; exit 1; }
python -c "import skimage" 2>/dev/null || { echo "[ERROR] scikit-image not found."; exit 1; }

echo ""
echo "Settings:"
echo "  PER_VESSEL=${PER_VESSEL}  MERGE_SPLIT=${MERGE_SPLIT}"
echo "  EPOCHS=${EPOCHS}  BATCH=${BATCH}x${ACCUM_STEPS}  LR=${LR}"
echo "  SemanticPrompt=${USE_SEMANTIC_PROMPT}  ReID=${USE_REID}"
echo "  SparseGAT=${USE_SPARSE_GAT} (${GAT_LAYERS}L ${GAT_HEADS}H k=${K_NEIGHBORS})"
echo "  λ_reid=${LAMBDA_REID}  artifact=${ARTIFACT_PROB}"
echo ""


build_train_cmd() {
    local VESSEL_ARG="$1"
    local SAVE="$2"

    local CMD="python train_v2.py"
    CMD+=" --data ${DATA_ROOT}"
    CMD+=" --vessels ${VESSEL_ARG}"
    CMD+=" --img_size ${IMG_SIZE}"
    CMD+=" --epochs ${EPOCHS}"
    CMD+=" --batch ${BATCH}"
    CMD+=" --accum_steps ${ACCUM_STEPS}"
    CMD+=" --lr ${LR}"
    CMD+=" --workers ${WORKERS}"
    CMD+=" --save_dir ${SAVE}"
    CMD+=" --save_every ${SAVE_EVERY}"
    CMD+=" --warmup_epochs ${WARMUP_EPOCHS}"
    CMD+=" --backbone_lr_scale ${BACKBONE_LR_SCALE}"
    CMD+=" --val_ratio ${VAL_RATIO}"
    CMD+=" --split_seed ${SPLIT_SEED}"
    CMD+=" --n_prompt_tokens ${N_PROMPT_TOKENS}"
    CMD+=" --reid_embed_dim ${REID_EMBED_DIM}"
    CMD+=" --lambda_reid ${LAMBDA_REID}"
    CMD+=" --gat_layers ${GAT_LAYERS}"
    CMD+=" --gat_heads ${GAT_HEADS}"
    CMD+=" --k_neighbors ${K_NEIGHBORS}"
    CMD+=" --max_nodes ${MAX_NODES}"
    CMD+=" --node_threshold ${NODE_THRESHOLD}"
    CMD+=" --gnn_iters ${GNN_ITERS}"
    CMD+=" --artifact_prob ${ARTIFACT_PROB}"

    [[ "${USE_AMP}"             == true  ]] && CMD+=" --amp"
    [[ "${USE_COMPILE}"         == true  ]] && CMD+=" --compile"
    [[ "${SCALE_LR}"            == true  ]] && CMD+=" --scale_lr"
    [[ "${NO_CACHE}"            == true  ]] && CMD+=" --no_cache"
    [[ "${UNFREEZE}"            == true  ]] && CMD+=" --unfreeze"
    [[ "${MERGE_SPLIT}"         == true  ]] && CMD+=" --merge_split"
    [[ "${USE_SEMANTIC_PROMPT}" == false ]] && CMD+=" --no_semantic_prompt"
    [[ "${USE_SPARSE_GAT}"      == false ]] && CMD+=" --use_dense_gnn"
    [[ "${USE_REID}"            == false ]] && CMD+=" --no_reid"
    [[ -n "${SAM3_CHECKPOINT}"  ]]         && CMD+=" --checkpoint ${SAM3_CHECKPOINT}"
    [[ -n "${RESUME}"           ]]         && CMD+=" --resume ${RESUME}"

    echo "${CMD}"
}


build_eval_cmd() {
    local VESSEL_ARG="$1"
    local SAVE="$2"
    local RESULT="$3"

    local CMD="python evaluate_v2.py"
    CMD+=" --data ${DATA_ROOT}"
    CMD+=" --ckpt ${SAVE}/best_model.pth"
    CMD+=" --log_csv ${SAVE}/train_log.csv"
    CMD+=" --vessels ${VESSEL_ARG}"
    CMD+=" --img_size ${IMG_SIZE}"
    CMD+=" --batch ${BATCH}"
    CMD+=" --workers ${WORKERS}"
    CMD+=" --out_dir ${RESULT}"
    CMD+=" --n_vis ${N_VIS}"
    CMD+=" --top_k ${TOP_K}"
    CMD+=" --cldice_iter ${CLDICE_ITER}"
    CMD+=" --gnn_iters ${GNN_ITERS}"
    CMD+=" --pp_min_size ${PP_MIN_SIZE}"

    [[ "${NO_CACHE}" == true ]] && CMD+=" --no_cache"

    echo "${CMD}"
}


run_one() {
    local LABEL="$1"
    local VESSEL_ARG="$2"
    local SAVE="$3"
    local RESULT="$4"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "  Training : ${LABEL}  (vessels=${VESSEL_ARG})"
    echo "╚══════════════════════════════════════════════╝"

    local TCMD; TCMD=$(build_train_cmd "${VESSEL_ARG}" "${SAVE}")
    echo "Command: ${TCMD}"
    echo ""
    eval "${TCMD}"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "  Evaluating : ${LABEL}"
    echo "╚══════════════════════════════════════════════╝"

    local ECMD; ECMD=$(build_eval_cmd "${VESSEL_ARG}" "${SAVE}" "${RESULT}")
    echo "Command: ${ECMD}"
    echo ""
    eval "${ECMD}"

    echo "  ✓ ${LABEL} done.  ckpt → ${SAVE}  results → ${RESULT}"
}


# =============================================================================
#  主流程
# =============================================================================
if [[ "${PER_VESSEL}" == true ]]; then
    echo "Mode: PER-VESSEL (3 models)"
    for V in LAD LCx RCA; do
        run_one "${V}" "${V}" "${SAVE_DIR}/${V}" "${RESULT_DIR}/${V}"
    done

    echo ""
    echo "============================================"
    echo " All per-vessel runs complete"
    echo "============================================"
    for V in LAD LCx RCA; do
        CSV="${RESULT_DIR}/${V}/test_metrics.csv"
        PROF="${RESULT_DIR}/${V}/inference_profile.txt"
        if [[ -f "${CSV}" ]]; then
            python -c "
import csv, statistics
with open('${CSV}') as f:
    rows = list(csv.DictReader(f))

def col(name):
    return [float(r[name]) for r in rows if name in r and r[name] != '']

dices = col('dice')
clds  = col('cldice')
ious  = col('iou')   # 新欄位（沒有 → 空 list）
hd95s = col('hd95')  # 新欄位
n = len(dices)

def fmt(vals):
    if not vals: return 'n/a'
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f'{m:.4f}+/-{s:.4f}'

def fmt_hd(vals):
    if not vals: return 'n/a'
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f'{m:.2f}+/-{s:.2f}'

print(f'  ${V:>3}: n={n:3d}  '
      f'Dice={fmt(dices)}  '
      f'clDice={fmt(clds)}  '
      f'IoU={fmt(ious)}  '
      f'HD95={fmt_hd(hd95s)}')
" 2>/dev/null || echo "  ${V}: (no results)"
        else
            echo "  ${V}: (no CSV found)"
        fi
        # 若 inference_profile.txt 存在，撈關鍵兩行附上
        if [[ -f "${PROF}" ]]; then
            GFLOPS=$(grep -E '^GFLOPs' "${PROF}" | head -1 | awk -F':' '{print $2}' | tr -d ' ')
            LATMED=$(grep -E '^Latency median' "${PROF}" | head -1 | awk -F':' '{print $2}' | tr -d ' ')
            echo "       └─ FLOPs=${GFLOPS} GFLOPs   Latency(median)=${LATMED}"
        fi
    done
else
    echo "Mode: JOINT (1 model)"
    run_one "ALL" "${VESSELS}" "${SAVE_DIR}" "${RESULT_DIR}"
fi

echo ""
echo "============================================"
echo " Done."
echo "============================================"