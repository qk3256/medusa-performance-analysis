#!/bin/bash
set -e # 如果任何命令失败，则立即退出

echo "STEP 1: Preparing a fixed set of test prompts..."
python 1_prepare_data.py

echo "\nSTEP 2: Running all benchmarks on the same data..."
python 2_run_benchmark.py --mode baseline --prompt-file test_prompts.json --output-file results/results_baseline.json
python 2_run_benchmark.py --mode medusa_base --prompt-file test_prompts.json --output-file results/results_medusa_base.json
python 2_run_benchmark.py --mode medusa_full --prompt-file test_prompts.json --output-file results/results_medusa_full.json

echo "\nSTEP 3: Generating final unified report..."
python 3_generate_report.py

echo "\nDone!"
