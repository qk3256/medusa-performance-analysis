import json
import numpy as np

def main():
    # 加载数据并提取 detailed_results
    baseline_data = json.load(open("results/results_baseline.json"))['detailed_results']
    medusa_base_data = json.load(open("results/results_medusa_base.json"))['detailed_results']
    medusa_full_data = json.load(open("results/results_medusa_full.json"))['detailed_results']

    print("--- DETAILED RESULTS PER PROMPT ---")
    for i in range(len(baseline_data)):
        prompt = baseline_data[i]['prompt']

        baseline_tps = baseline_data[i]['tokens_per_sec']
        base_tps = medusa_base_data[i]['tokens_per_sec']
        full_tps = medusa_full_data[i]['tokens_per_sec']

        print(f"\nPrompt #{i+1}: {prompt[:80]}...")
        print("-" * 80)
        print(f"  Baseline   : {baseline_tps:.2f} t/s -- Output: {baseline_data[i]['generated_text'][:100]}...")
        print(f"  Medusa Base: {base_tps:.2f} t/s -- Output: {medusa_base_data[i]['generated_text'][:100]}...")
        print(f"  Medusa Full: {full_tps:.2f} t/s -- Output: {medusa_full_data[i]['generated_text'][:100]}...")
        print("-" * 80)

    # 打印汇总统计
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # 加载统计信息
    baseline_stats = json.load(open("results/results_baseline.json"))['statistics']
    medusa_base_stats = json.load(open("results/results_medusa_base.json"))['statistics']
    medusa_full_stats = json.load(open("results/results_medusa_full.json"))['statistics']
    
    print(f"\nAverage Tokens per Second:")
    print(f"  Baseline   : {baseline_stats['avg_speed']:.2f} ± {baseline_stats['std_speed']:.2f}")
    print(f"  Medusa Base: {medusa_base_stats['avg_speed']:.2f} ± {medusa_base_stats['std_speed']:.2f}")
    print(f"  Medusa Full: {medusa_full_stats['avg_speed']:.2f} ± {medusa_full_stats['std_speed']:.2f}")
    
    print(f"\nSpeedup vs Baseline:")
    print(f"  Medusa Base: {medusa_base_stats['avg_speed']/baseline_stats['avg_speed']:.2f}x")
    print(f"  Medusa Full: {medusa_full_stats['avg_speed']/baseline_stats['avg_speed']:.2f}x")

if __name__ == "__main__":
    main()
