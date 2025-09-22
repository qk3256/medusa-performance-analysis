#!/bin/bash
set -e # 如果任何命令失败，则立即退出

# 创建必要的目录
mkdir -p ../data
mkdir -p ../results

echo "STEP 1: Preparing a fixed set of test prompts..."
python prepare_data.py

echo "\nSTEP 2: Running all benchmarks on the same data..."
python run_benchmark.py --mode baseline --prompt-file ../data/test_prompts.json --output-file ../results/results_baseline.json
python run_benchmark.py --mode medusa_base --prompt-file ../data/test_prompts.json --output-file ../results/results_medusa_base.json
python run_benchmark.py --mode medusa_full --prompt-file ../data/test_prompts.json --output-file ../results/results_medusa_full.json

echo "\nSTEP 3: Generating final unified report..."
python generate_report.py

echo "\nDone!"
