set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SAM3 UNet V2 — 純推論 / 評估腳本（不訓練）                                  ║
# ║  載入訓練好的 checkpoint → 跑 evaluate_v2.py                                 ║
# ║                                                                            ║
# ║  Outputs: test_metrics.csv, inference_profile.txt, summary_grid.png,       ║
# ║           fixed_samples.png, comparisons/, reid_tsne.png (若 ReID 開啟)     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  1. 路徑設定                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
DATA_ROOT="C:/Users/user/Desktop/ARCADE"
CKPT_DIR="./checkpoints_v2"          # 訓練時的 save_dir（包含 best_model.pth + model_config.pth）
RESULT_DIR="./results_v2"            # 評估輸出目錄
VESSELS="LAD,LCx,RCA"

# Checkpoint / log 檔名
CKPT_NAME="best_model.pth"           # 預設用 best；可改 latest.pth / epoch_N.pth
LOG_CSV_NAME="train_log.csv"         # 用來畫 training curves；不存在自動跳過

# 直接指定 checkpoint 路徑 (留空 = 用 ${CKPT_DIR}/${CKPT_NAME})
# 例： CKPT_OVERRIDE="C:/path/to/some_other_ckpt.pth"
CKPT_OVERRIDE="D:/SAM3gnn_net/checkpoints_v2/best_model.pth"
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  2. 評估模式                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
# true  → 對 LAD / LCx / RCA 各自評估（吃 ${CKPT_DIR}/{V}/best_model.pth）
# false → 單一聯合模型，吃 ${CKPT_DIR}/best_model.pth
PER_VESSEL=false

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  3. 資料 / 推論基本參數                                                 │
# └─────────────────────────────────────────────────────────────────────────┘
IMG_SIZE=512
BATCH=10
WORKERS=0
NO_CACHE=false

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  4. 後處理鏈                                                           │
# └─────────────────────────────────────────────────────────────────────────┘
PP_MIN_SIZE=50                       # remove_small_objects 的最小像素數
# 最大連通去雜訊（LCC）：
#   0  = 關閉
#   1  = 只留最大連通元件
#   2+ = 保留前 N 大
PP_KEEP_TOP_K=0
PP_LCC_MIN_RATIO=0                 # 元件需 ≥ 最大 × 比例 才保留（0=不啟用）

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  5. clDice / 視覺化                                                    │
# └─────────────────────────────────────────────────────────────────────────┘
CLDICE_ITER=10                       # soft skeleton 迭代次數
N_VIS=20                             # 存幾張 comparison 圖
TOP_K=4                              # summary_grid 每組（worst/median/best）抓 N 張

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  7. Dense GNN 後備迭代數（model_config 未指定時生效）                    │
# └─────────────────────────────────────────────────────────────────────────┘
GNN_ITERS=3


# =============================================================================
#  環境檢查
# =============================================================================
echo "================================================"
echo " SAM3 UNet V2 — Inference / Evaluation Only"
echo "================================================"
python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
try:
    import torch_geometric
    print(f'PyG     : {torch_geometric.__version__}')
except ImportError:
    print('PyG     : NOT INSTALLED (dense GNN fallback)')
" 2>/dev/null || { echo "[ERROR] PyTorch not found."; exit 1; }
python -c "import scipy"   2>/dev/null || { echo "[ERROR] SciPy not found."; exit 1; }
python -c "import cv2"     2>/dev/null || { echo "[ERROR] opencv-python not found."; exit 1; }
python -c "import skimage" 2>/dev/null || { echo "[ERROR] scikit-image not found."; exit 1; }

echo ""
echo "Settings:"
echo "  Mode        = $([[ "${PER_VESSEL}" == true ]] && echo PER-VESSEL || echo JOINT)"
echo "  Data root   = ${DATA_ROOT}"
echo "  Ckpt dir    = ${CKPT_DIR}"
echo "  Result dir  = ${RESULT_DIR}"
echo "  Batch       = ${BATCH}    Img size = ${IMG_SIZE}"
echo "  PP_MIN_SIZE = ${PP_MIN_SIZE}"
if [[ ${PP_KEEP_TOP_K} -gt 0 ]]; then
    echo "  LCC         = ON   (keep_top_k=${PP_KEEP_TOP_K}, min_ratio=${PP_LCC_MIN_RATIO})"
else
    echo "  LCC         = OFF"
fi
echo ""


# =============================================================================
#  指令組合
# =============================================================================
build_eval_cmd() {
    local VESSEL_ARG="$1"
    local CKPT_PATH="$2"
    local LOG_CSV="$3"
    local RESULT="$4"

    local CMD="python evaluate_v2.py"
    CMD+=" --data ${DATA_ROOT}"
    CMD+=" --ckpt ${CKPT_PATH}"
    CMD+=" --log_csv ${LOG_CSV}"
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
    CMD+=" --pp_keep_top_k ${PP_KEEP_TOP_K}"
    CMD+=" --pp_lcc_min_ratio ${PP_LCC_MIN_RATIO}"

    [[ "${NO_CACHE}"       == true ]] && CMD+=" --no_cache"

    echo "${CMD}"
}


# Pre-flight check：ckpt 必須存在；model_config 缺失只警告
check_ckpt() {
    local CKPT_PATH="$1"
    local LABEL="$2"
    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[ERROR] ${LABEL}: checkpoint not found at"
        echo "        ${CKPT_PATH}"
        return 1
    fi
    local CFG_PATH
    CFG_PATH="$(dirname "${CKPT_PATH}")/model_config.pth"
    if [[ ! -f "${CFG_PATH}" ]]; then
        echo "[WARN]  ${LABEL}: model_config.pth not found next to checkpoint"
        echo "        → evaluate_v2.py 將用 hard-coded defaults，"
        echo "          model 結構可能與訓練時不一致 (load_state_dict 會 strict=False)。"
    fi
    return 0
}


run_one_eval() {
    local LABEL="$1"
    local VESSEL_ARG="$2"
    local CKPT_DIR_LOCAL="$3"
    local RESULT="$4"

    local CKPT_PATH
    if [[ -n "${CKPT_OVERRIDE}" ]]; then
        CKPT_PATH="${CKPT_OVERRIDE}"
    else
        CKPT_PATH="${CKPT_DIR_LOCAL}/${CKPT_NAME}"
    fi
    local LOG_CSV="${CKPT_DIR_LOCAL}/${LOG_CSV_NAME}"

    if ! check_ckpt "${CKPT_PATH}" "${LABEL}"; then
        echo "  ✗ ${LABEL} skipped."
        return 1
    fi

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "  Evaluating : ${LABEL}  (vessels=${VESSEL_ARG})"
    echo "╚══════════════════════════════════════════════╝"
    echo "  Checkpoint : ${CKPT_PATH}"
    echo "  Result dir : ${RESULT}"

    local ECMD; ECMD=$(build_eval_cmd "${VESSEL_ARG}" "${CKPT_PATH}" "${LOG_CSV}" "${RESULT}")
    echo "Command: ${ECMD}"
    echo ""
    eval "${ECMD}"

    echo ""
    echo "  ✓ ${LABEL} done.  → ${RESULT}"
}


# =============================================================================
#  主流程
# =============================================================================
if [[ "${PER_VESSEL}" == true ]]; then
    echo "Mode: PER-VESSEL (up to 3 evaluations)"
    FAILED=0
    for V in LAD LCx RCA; do
        run_one_eval "${V}" "${V}" "${CKPT_DIR}/${V}" "${RESULT_DIR}/${V}" || FAILED=$((FAILED+1))
    done

    echo ""
    echo "============================================"
    echo " Per-vessel evaluation summary"
    if [[ ${FAILED} -gt 0 ]]; then
        echo "  (${FAILED} of 3 failed — see logs above)"
    fi
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
ious  = col('iou')
hd95s = col('hd95')
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
" 2>/dev/null || echo "  ${V}: (parse error)"
        else
            echo "  ${V}: (no CSV — eval skipped or failed)"
        fi
        if [[ -f "${PROF}" ]]; then
            GFLOPS=$(grep -E '^GFLOPs' "${PROF}" | head -1 | awk -F':' '{print $2}' | tr -d ' ')
            LATMED=$(grep -E '^Latency median' "${PROF}" | head -1 | awk -F':' '{print $2}' | tr -d ' ')
            echo "       └─ FLOPs=${GFLOPS} GFLOPs   Latency(median)=${LATMED}"
        fi
    done
else
    echo "Mode: JOINT (1 evaluation)"
    run_one_eval "ALL" "${VESSELS}" "${CKPT_DIR}" "${RESULT_DIR}"
fi

echo ""
echo "============================================"
echo " Done."
echo "============================================"