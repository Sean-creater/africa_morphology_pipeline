#!/bin/bash
# Within-cluster image generation (2-step, Drive-mediated pipeline).
# Step 1: imagery/export_to_drive.py submits batched GEE export tasks (cloud-side).
# Step 2: imagery/download_from_drive.py downloads the results and renders them.

PYTHON=/home/sjs96_file_share/anaconda3/envs/Africa_env/bin/python3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

PAIRS_CSV=$ROOT/within_cluster/outputs/within_pairs_k02_k30.csv
OUT_IMAGES=$ROOT/within_cluster/outputs/images
TIFF_CACHE=$ROOT/within_cluster/outputs/tiff_cache
TASK_JSON=$ROOT/logs/within_export_tasks.json
DRIVE_FOLDER="GEE_Within_Cluster_Exports"

mkdir -p "$OUT_IMAGES" "$TIFF_CACHE"

cd $ROOT

echo "=== Step 1: submit GEE export tasks ==="
$PYTHON imagery/export_to_drive.py \
    --project YOUR_GEE_PROJECT_ID \
    --pairs-csv "$PAIRS_CSV" \
    --task-json "$TASK_JSON" \
    --out-images "$OUT_IMAGES" \
    --drive-folder "$DRIVE_FOLDER"

echo ""
echo "=== Tasks submitted; run Step 2 once they complete ==="
echo "Monitor task status:"
echo "  $PYTHON imagery/export_to_drive.py --project YOUR_GEE_PROJECT_ID --task-json $TASK_JSON --status"
echo ""
echo "Step 2 (run once tasks complete):"
echo "  $PYTHON imagery/download_from_drive.py \\"
echo "    --project YOUR_GEE_PROJECT_ID \\"
echo "    --task-json $TASK_JSON \\"
echo "    --out-images $OUT_IMAGES \\"
echo "    --tiff-cache $TIFF_CACHE \\"
echo "    --drive-folder $DRIVE_FOLDER"
