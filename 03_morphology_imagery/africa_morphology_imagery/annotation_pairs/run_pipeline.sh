#!/bin/bash
#
# Runs the full anchor-based comparison-pair generation pipeline in order.
#
# Usage: bash run_pipeline.sh
#
# Expects render.py's png_integrated/ output to already exist and
# ../city_list.txt (see data/reference/annotation_city_list.txt for the
# 40-city list used in this study) to list the cities to index.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "Urban Morphology Comparison Pairs Generation Pipeline"
echo "========================================================================"

mkdir -p ../outputs

echo -e "\nStep 1/6: Building city index..."
python3 01_build_city_index.py --png-dir "$1"

echo -e "\nStep 2/6: Selecting city representatives..."
python3 02_select_representatives.py

echo -e "\nStep 3/6: Selecting global anchors..."
python3 03_select_anchors.py

echo -e "\nStep 4/6: Generating comparison pairs..."
python3 04_generate_pairs.py

echo -e "\nStep 5/6: Validating pairs..."
python3 05_validate_pairs.py

echo -e "\nStep 6/6: Preparing data for labeling..."
python3 06_prepare_for_labeling.py

echo -e "\n========================================================================"
echo "Pipeline complete."
echo "========================================================================"
echo "Generated files:"
echo "  - ../outputs/city_index.json"
echo "  - ../outputs/city_representatives.json"
echo "  - ../outputs/global_anchors.json"
echo "  - ../outputs/comparison_pairs.csv"
echo "  - ../labeling_data/master_index.json"
echo "  - ../labeling_data/batches/*.json"
