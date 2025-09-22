# 2_run_benchmark.py
import torch                                                                    
import time                                                                     
import argparse 
import json                                                                
import gc                                                                       
import datetime 
import os                                                           
import numpy as np                                                              
from datasets import load_dataset                                               
from fastchat.model import get_conversation_template              

# Medusa imports
from transformers import AutoTokenizer, AutoModelForCausalLM

from medusa.model.utils import *
from medusa.model.medusa_model import MedusaModel                               
from medusa.model.kv_cache import initialize_past_key_values
from medusa.model.medusa_choices import *

# Baseline model with original KV-Cache (standard autoregressive decoding)
def baseline_forward(input_ids, model, tokenizer, temperature, _threshold, posterior_alpha, max_steps = 512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values

    input_len = input_ids.shape[1]
    total_length = input_len + max_steps
    attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=input_ids.device)], dim=1)

    torch.cuda.synchronize()
    start_time = time.time()

    outputs = model.generate(     
        input_ids=input_ids,     
        attention_mask=attention_mask,
        max_new_tokens=max_steps,  
        use_cache=True, 
        past_key_values=past_key_values,
        do_sample=False                                                                                                                                       
    ) 

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    new_token = outputs.shape[1] - input_len

    return outputs, new_token, new_token, total_time

# Baseline / Medusa model with custom KV-Cache implementation (Medusa-specific, differs from HuggingFace's original)
def medusa_forward(input_ids, model, tokenizer, temperature, _threshold, posterior_alpha, max_steps = 512, mode = "medusa_full", medusa_choices = None):
    """
    Medusa model forward pass with two modes:
    - medusa_base: Uses Medusa architecture but standard autoregressive decoding
    - medusa_full: Full Medusa speculative decoding with tree attention
    """
    max_steps = min(max_steps, 1024)
    assert mode == "medusa_base" or mode == "medusa_full"

    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    if medusa_choices is None:
        medusa_choices = model.get_medusa_choice(model.base_model_name_or_path)

    if hasattr(model, "medusa_choices") and model.medusa_choices == medusa_choices:
        # Load the cached medusa buffer
        medusa_buffers = model.medusa_buffers
    else:
        # Initialize the medusa buffer
        medusa_buffers = generate_medusa_buffers(
            medusa_choices, device=model.base_model.device
        )
    model.medusa_buffers = medusa_buffers
    model.medusa_choices = medusa_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]

    reset_medusa_mode(model)
    if mode == "medusa_full":
        medusa_logits, logits = initialize_medusa(
                input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
            ) 
    else:
        outputs = model.base_model(input_ids, past_key_values = past_key_values, use_cache=True)
        logits = outputs.logits
        medusa_logits = None
    new_token = 0
    
    torch.cuda.synchronize()
    start_time = time.time()

    for idx in range(max_steps): 
        if mode == "medusa_full":
            # Step 1: Generate multiple candidate tokens using Medusa heads
            candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                )
            # Step 2: Perform tree-based parallel decoding of candidates
            medusa_logits, logits, outputs = tree_decoding(
                    model,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )
            # Step 3: Evaluate and select the best candidate sequence
            best_candidate, accept_length = evaluate_posterior(
                    logits, candidates, temperature, _threshold, posterior_alpha
                )
            # Step 4: Update inputs and states based on accepted tokens
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    medusa_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )
        elif mode == "medusa_base":
            # Standard autoregressive decoding with Medusa architecture
            new_token += 1
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = model.base_model(input_id, use_cache=True, past_key_values = past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

        # Early stopping conditions
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_steps:
            break
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    return input_ids, new_token, idx, total_time

def run_single_test(mode, model, tokenizer, prompt_text, max_new_tokens, temperature=0.0, posterior_threshold=0.09, posterior_alpha=0.3):
    """
    Runs a single benchmark test with proper conversation template and CUDA timing
    """
    # Prepare prompt using FastChat's conversation template
    model_path = model.config._name_or_path
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    final_prompt = conv.get_prompt()
    
    # Tokenize with the formatted prompt
    input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Run generation based on mode
    if mode == 'baseline':
        output_ids, new_tokens, steps, total_time = baseline_forward(
            input_ids, model, tokenizer, temperature,
            posterior_threshold, posterior_alpha, max_steps=max_new_tokens
        )
    elif mode == 'medusa_base':
        output_ids, new_tokens, steps, total_time = medusa_forward(
            input_ids, model, tokenizer, temperature,
            posterior_threshold, posterior_alpha, max_steps=max_new_tokens, mode = mode
        )
    elif mode == 'medusa_full':
        output_ids, new_tokens, steps, total_time = medusa_forward(
            input_ids, model, tokenizer, temperature,
            posterior_threshold, posterior_alpha, max_steps=max_new_tokens, mode = mode
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Calculate precise timing (in seconds)
    tokens_per_sec = new_tokens / total_time if total_time > 0 else 0

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text ({mode}):")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

    return {
        "prompt": prompt_text,
        "generated_text": generated_text,
        "mode": mode,
        "tokens_per_sec": float(tokens_per_sec),
        "new_tokens_generated": int(new_tokens),
        "total_time_seconds": float(total_time)
    }

def benchmark_model(model, tokenizer, prompts, max_new_tokens, temperature, posterior_threshold, posterior_alpha, warmup_runs=3, mode = "baseline"):
    """Run benchmark tests for a specific model mode"""
    results = []

    # Warmup
    print(f"Running warmup for {mode}...")
    for _ in range(warmup_runs):
        test_prompt = "Explain quantum computing in simple terms."
        input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.cuda()
        if 'medusa' in mode:
            forward_func = medusa_forward
            _ = forward_func(input_ids, model, tokenizer, temperature, posterior_threshold, posterior_alpha, max_steps=10, mode = mode)
        else:
            forward_func = baseline_forward
            _ = forward_func(input_ids, model, tokenizer, temperature, posterior_threshold, posterior_alpha, max_steps=10)

    # Benchmark
    print(f"\nBENCHMARKING {mode.upper()}")
    print("="*60)

    for i, prompt in enumerate(prompts):
        print(f"{mode.title()} [{i+1}/{len(prompts)}]: {prompt[:50]}...")
       
        single_run_result = run_single_test(
            mode=mode,
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha
        )
        
        results.append(single_run_result)
        
        tokens_per_sec = single_run_result['tokens_per_sec']
        new_tokens = single_run_result['new_tokens_generated']
        print(f"  {tokens_per_sec:.2f} tokens/sec, {new_tokens} tokens")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark Medusa vs Baseline with CUDA timing")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Dataset to load prompts from")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--posterior-threshold", type=float, default=0.09, help="Posterior threshold for Medusa")
    parser.add_argument("--posterior-alpha", type=float, default=0.3, help="Posterior alpha for Medusa")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs before benchmarking")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--mode", type=str, required=True, choices=['baseline', 'medusa_base', 'medusa_full'])
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the JSON file containing test prompts.")
    parser.add_argument("--output-file", type = str, required = True, help = "Path to the JSON file containing the result.")

    args = parser.parse_args()

    # Configuration                                                                                                                                                  
    BASELINE_MODEL_PATH = "lmsys/vicuna-7b-v1.3"                                                                                                                     
    MEDUSA_MODEL_PATH = "FasterDecoding/medusa-vicuna-7b-v1.3"                                                                                                      
    
    # Load prompts from official dataset
    print("Loading test prompts...")                                                                                                                       
    with open(args.prompt_file, 'r') as f:
        prompts = json.load(f)

    # Benchmark configurations
    benchmark_configs = {
        'baseline': {
            'model_path': BASELINE_MODEL_PATH,
            'model_class': AutoModelForCausalLM
        },
        'medusa_full': {
            'model_path': MEDUSA_MODEL_PATH,
            'model_class': MedusaModel
        },
        'medusa_base': {
            'model_path': MEDUSA_MODEL_PATH,
            'model_class': MedusaModel
        }
    }

    config = benchmark_configs[args.mode]
    
    print(f"\n{'='*60}")
    print(f"LOADING {args.mode.upper()} MODEL")
    print(f"{'='*60}")
        
    model = config['model_class'].from_pretrained(
            config['model_path'], torch_dtype=torch.float16, device_map="auto"
        )
    if 'medusa' in args.mode:
        tokenizer = model.get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model_path'])                                                                                                   

    # Run benchmark using the new run_single_test function
    results = benchmark_model(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        posterior_threshold=args.posterior_threshold,
        posterior_alpha=args.posterior_alpha,
        warmup_runs=args.warmup_runs
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate and display statistics
    def calculate_stats(results):
        speeds = [r['tokens_per_sec'] for r in results]
        tokens = [r['new_tokens_generated'] for r in results]
        return {
            'avg_speed': float(np.mean(speeds)),  
            'std_speed': float(np.std(speeds)),   
            'avg_tokens': float(np.mean(tokens)),  
            'min_speed': float(np.min(speeds)),  
            'max_speed': float(np.max(speeds))  
        }

    stats = calculate_stats(results)
    
    # Print results
    output_data = {
        'mode': args.mode,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'posterior_threshold': args.posterior_threshold if 'medusa' in args.mode else None,
            'posterior_alpha': args.posterior_alpha if 'medusa' in args.mode else None,
            'warmup_runs': args.warmup_runs,
            'dataset': args.dataset,
            'prompt_count': len(prompts)
        },
        'statistics': stats,
        'detailed_results': results
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS FOR {args.mode.upper()}")
    print(f"{'='*60}")
    print(f"Average speed: {stats['avg_speed']:.2f} ± {stats['std_speed']:.2f} tokens/sec")
    print(f"Speed range: {stats['min_speed']:.2f} - {stats['max_speed']:.2f} tokens/sec")
    print(f"Average tokens generated: {stats['avg_tokens']:.1f}")
    print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()
