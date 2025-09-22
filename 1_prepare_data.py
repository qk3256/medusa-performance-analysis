# 1_prepare_data.py
import json
import random
from datasets import load_dataset

def load_test_prompts(num_prompts=10, dataset_name="tatsu-lab/alpaca"):
    """
    Load test prompts from HuggingFace official datasets
    """
    try:
        print(f"Loading prompts from {dataset_name}...")
        dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)

        prompts = []
        # Extract prompts based on the dataset's structure
        if 'instruction' in dataset.column_names:
            # Alpaca format: 'instruction' field
            prompts = [item['instruction'] for item in dataset if item['instruction']]
        elif 'conversations' in dataset.column_names:
            # ShareGPT/Vicuna format
            for item in dataset:
                if item['conversations'] and len(item['conversations']) > 0:
                    first_turn = item['conversations'][0]
                    if 'value' in first_turn:
                        prompts.append(first_turn['value'])
        elif 'text' in dataset.column_names:
            # Generic text format
            prompts = [item['text'] for item in dataset if item['text'] and len(item['text']) > 20]
        elif 'prompt' in dataset.column_names:
            # Other possible dataset formats
            prompts = [item['prompt'] for item in dataset if item['prompt']]

        # Filter prompts that are too short or too long
        filtered_prompts = []
        for prompt in prompts:
            if prompt and 10 < len(prompt) < 500:
                filtered_prompts.append(prompt)

        # Randomly select the specified number of prompts
        selected_prompts = random.sample(filtered_prompts, min(num_prompts, len(filtered_prompts)))
        print(f"Successfully loaded {len(selected_prompts)} prompts from {dataset_name}")
        return selected_prompts

    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        # Fallback prompts
        backup_prompts = [
            "Explain the theory of relativity in simple terms.",
            "Write a story about artificial intelligence.",
            "Describe the process of photosynthesis.",
            "What are the benefits of renewable energy?",
            "Explain how blockchain technology works.",
            "Discuss the impact of climate change.",
            "Describe the water cycle.",
            "What is machine learning?",
            "Explain quantum computing basics.",
            "Discuss the future of space exploration."
        ]
        return random.sample(backup_prompts, min(num_prompts, len(backup_prompts)))

def main():
    # --- 配置 ---
    NUM_PROMPTS = 10
    DATASET_NAME = "tatsu-lab/alpaca"
    OUTPUT_FILE = "test_prompts.json"
    print("Preparing a fixed set of test prompts...")
    # 1. 调用一次加载和随机抽样函数
    prompts = load_test_prompts(num_prompts=NUM_PROMPTS, dataset_name=DATASET_NAME)
    
    # 2. 将这组固定的 prompts 保存到文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully saved {len(prompts)} prompts to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
