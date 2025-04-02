#!/usr/bin/env python3

import array
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from vllm import EngineArgs, LLMEngine, SamplingParams
import mlperf_loadgen as lg 

from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer


input_parameters_file_path = sys.argv[1]

input_parameters = {}

with open(input_parameters_file_path) as f:
    input_parameters = json.load(f)

## Non-preprocessed, non-tokenized original Openorca dataset as one JSON file
#
dataset_path                = input_parameters["dataset_path"]

## language model:
#
mlperf_model_name           = input_parameters["mlperf_model_name"]
model_name                  = input_parameters["model_name"]
hf_model_name               = input_parameters["hf_model_name"]
model_path                  = input_parameters["model_path"]
download_dir                = input_parameters["download_dir"]
dtype                       = input_parameters["dtype"]
backend                     = input_parameters["backend"]
enforce_eager               = bool(input_parameters["enforce_eager"])
gpu_memory_utilization      = float(input_parameters["gpu_memory_utilization"])
device                      = input_parameters["device"]
max_model_len               = int(input_parameters["max_model_len"])
max_seq_len_to_capture      = int(input_parameters["max_seq_len_to_capture"])
disable_custom_all_reduce   = bool(input_parameters["disable_custom_all_reduce"])
max_num_batched_tokens      = int(input_parameters["max_num_batched_tokens"])
max_num_seqs                = int(input_parameters["max_num_seqs"])
swap_space                  = float(input_parameters["swap_space"])
enable_lora                 = bool(input_parameters["enable_lora"])
max_loras                   = int(input_parameters["max_loras"])
lora_paths                  = input_parameters["lora_paths"]
max_lora_rank               = int(input_parameters["max_lora_rank"])
tensor_parallel_size        = int(input_parameters["tensor_parallel_size"])

## Processing by batches:
#
batch_size                  = input_parameters["batch_size"]

## Loadgen params:
#
scenario_str                = input_parameters["loadgen_scenario"]
mode_str                    = input_parameters["loadgen_mode"]
dataset_size                = input_parameters["loadgen_dataset_size"]
buffer_size                 = input_parameters["loadgen_buffer_size"]
mlperf_conf_path            = input_parameters["loadgen_mlperf_conf_path"]
user_conf_path              = input_parameters["loadgen_user_conf_path"]
verbosity                   = input_parameters["verbosity"]

## Sampling params:
#
top_k                       = int(input_parameters["top_k"])
temperature                 = float(input_parameters["temperature"])
detokenize                  = bool(input_parameters["detokenize"])
max_tokens                  = int(input_parameters["max_tokens"])
min_tokens                  = int(input_parameters["min_tokens"])
repetition_penalty          = int(input_parameters["repetition_penalty"])
frequency_penalty           = int(input_parameters["frequency_penalty"])
ignore_eos                  = bool(input_parameters["ignore_eos"])
early_stopping              = bool(input_parameters["early_stopping"])
use_beam_search             = bool(input_parameters["use_beam_search"])
skip_special_tokens         = bool(input_parameters["skip_special_tokens"])

disable_log_stats           = True

print(f"Backend: {backend}\n")

# Initialize tokenizer
print(f"Loading tokenizer from {hf_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
print(" done.\n")

print(f"Loading model with weights from {model_path} ...")
engine_args_params = {"model": model_path, "served_model_name": hf_model_name, "dtype": dtype, "disable_log_stats": disable_log_stats, "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_memory_utilization, "device": device,
        "max_model_len": max_model_len, "max_seq_len_to_capture": max_seq_len_to_capture,
        "disable_custom_all_reduce": disable_custom_all_reduce, "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs, "swap_space": swap_space, "download_dir": download_dir, "enable_lora": enable_lora, "max_loras": max_loras, "max_lora_rank": max_lora_rank,
        "tensor_parallel_size": tensor_parallel_size}

if device == "tpu":
    os.environ["VLLM_XLA_CACHE_PATH"] = str(Path.home() / ".cache" / "vllm" / "xla_cache")

engine_args     = EngineArgs(**engine_args_params)
print(f"Creating LLM engine with parameters: {engine_args}")
llm_engine      = LLMEngine.from_engine_args(engine_args)
sampling_params = SamplingParams(top_k=top_k, temperature=temperature, detokenize=detokenize, max_tokens=max_tokens,
                                 min_tokens=min_tokens, repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty,
                                 ignore_eos=ignore_eos, skip_special_tokens=skip_special_tokens)
print(" done.\n")


def generate_synthetic_data(target_seq_length, num_samples=500):
    """Generate synthetic data with specified sequence length.
    Args:
        target_seq_length: Target sequence length in tokens
        num_samples: Number of synthetic samples to generate
    Returns:
        List of dictionaries containing synthetic prompts
    """
    global max_model_len
    
    # Ensure we don't exceed max_model_len
    actual_seq_length = min(target_seq_length, max_model_len)
    if actual_seq_length < target_seq_length:
        print(f"Warning: Requested sequence length {target_seq_length} exceeds max_model_len {max_model_len}. Using {actual_seq_length} instead.")
    
    # Generate synthetic text with varied content
    prompts = [
        "Analyze the following text and provide a detailed summary: ",
        "Consider the implications of the following statement: ",
        "Explain the significance of this concept: ",
        "Evaluate the following scenario: ",
        "Describe the key aspects of: "
    ]
    topics = [
        "artificial intelligence and its impact on society",
        "sustainable development and environmental conservation",
        "advances in quantum computing and their applications",
        "global economic trends and their effects",
        "breakthrough discoveries in medical research"
    ]
    
    all_prompts = []
    for i in range(num_samples):
        # Create varied synthetic text by combining different prompts and topics
        base_prompt = prompts[i % len(prompts)]
        base_topic = topics[i % len(topics)]
        synthetic_text = f"{base_prompt}{base_topic}. "
        
        # Add context to reach desired length
        context = ("This analysis should consider multiple perspectives including technological, "  
                  "social, economic, and ethical implications. Consider both short-term and long-term effects, "  
                  "potential benefits and challenges, and recommendations for future developments. ")
        
        synthetic_text = synthetic_text + context * (actual_seq_length // 100 + 1)
        
        # Properly tokenize the text
        tokenized = tokenizer.encode(synthetic_text, add_special_tokens=False)
        
        # Ensure exact sequence length
        if len(tokenized) > actual_seq_length:
            tokenized = tokenized[:actual_seq_length]
        elif len(tokenized) < actual_seq_length:
            # Pad with varied meaningful content
            pad_texts = [
                " This requires careful consideration of various factors.",
                " We must evaluate the long-term consequences.",
                " Further research in this area would be valuable.",
                " These insights could lead to significant advances.",
                " The implications of this are far-reaching."
            ]
            while len(tokenized) < actual_seq_length:
                pad_text = pad_texts[len(tokenized) % len(pad_texts)]
                pad_tokens = tokenizer.encode(pad_text, add_special_tokens=False)
                remaining = actual_seq_length - len(tokenized)
                tokenized.extend(pad_tokens[:remaining])
        
        # Get the actual text from tokens for consistency
        synthetic_text = tokenizer.decode(tokenized)
        
        all_prompts.append({
            "string_prompt": synthetic_text,
            "prompt_token_ids": tokenized
        })
    
    return all_prompts

def load_and_substitute_dataset(dataset_path):
    """Load dataset from file or generate synthetic data based on sequence length.
    For sequences > 2048 tokens, automatically generates synthetic data.
    """
    global max_seq_len_to_capture, dataset_size
    
    if max_seq_len_to_capture > 2048:
        print(f"Sequence length {max_seq_len_to_capture} > 2048, using synthetic data...")
        # For synthetic data, use a reasonable number of samples (1024)
        # This matches max_num_seqs and helps prevent memory issues
        num_samples = 1024
        print(f"Generating {num_samples} synthetic samples...")
        return generate_synthetic_data(max_seq_len_to_capture, num_samples=num_samples)
    else:
        # For sequences <= 2048, use OpenOrca dataset
        import pandas as pd
        processed_data = pd.read_pickle(dataset_path)

        inputs = processed_data['input']
        input_tokens = processed_data['tok_input']

        all_prompts = []
        for i in range(len(inputs)):
            all_prompts.append({
                "string_prompt": inputs[i],
                "prompt_token_ids": input_tokens[i]
            })

    return all_prompts


print(f"Loading dataset from {dataset_path} ...")
all_prompts = load_and_substitute_dataset(dataset_path)
print(" done.\n")


def tick(letter, quantity=1):
    if verbosity:
        print(letter + (str(quantity) if quantity>1 else ''), end='')

def load_query_samples(sample_indices):
    if verbosity > 1:
        print(f"load_query_samples({sample_indices})")

    len_sample_indices = len(sample_indices)
    tick('B', len_sample_indices)


def unload_query_samples(sample_indices):
    #print(f"unload_query_samples({sample_indices})")
    tick('U')

    if verbosity:
        print('')

def worker_code():
    print(f"[W] Started")
    iter = 0

    global worker_is_needed     # primitive variable poll signalling
    worker_is_needed = True
    finished = 0

    while worker_is_needed:
        pre_step = time.time()
        request_outputs = llm_engine.step()
        #print(f"[W] ENGINE_STEP: {iter}, time={(time.time()-pre_step)*1000}ms")

        for i, request_output in enumerate(request_outputs):
            request_id  = int(request_output.request_id)
            token_ids   = request_output.outputs[0].token_ids
            output_text = request_output.outputs[0].text or "None"
            if request_output.finished or len(token_ids)==1:
                request_id  = int(request_output.request_id)
                finished += request_output.finished
                if request_output.finished:
                    print(f"Finished {finished}")
                #print(request_output)
                #print("------------")

                response_array = array.array("B", np.array(token_ids, np.int32).tobytes())
                bi = response_array.buffer_info()
                if request_output.finished:
                    response = [ lg.QuerySampleResponse(request_id, bi[0], bi[1], len(token_ids)) ]
                    lg.QuerySamplesComplete(response)
                    if verbosity:
                        print(f"[W] RESPONSE: {request_id} -> {token_ids} -> {output_text}")
                else:
                    response = [ lg.QuerySampleResponse(request_id, bi[0], bi[1]) ]
                    lg.FirstTokenComplete(response)

        if not llm_engine.has_unfinished_requests():
            if verbosity:
                print('[W] bzzz...')
        while worker_is_needed and not llm_engine.has_unfinished_requests():
            pass

        iter += 1

def issue_queries(query_samples):
    if verbosity > 2:
        printable_query = [(qs.index, qs.id) for qs in query_samples]
        print(f"issue_queries( {printable_query} )")

    tick('Q', len(query_samples))

    for j in range(0, len(query_samples), batch_size):
        batch = query_samples[j:j+batch_size]
        for index_in_batch, qs in enumerate(batch):
            global_index = qs.index

            prompt      = all_prompts[global_index]
            request_id  = str(qs.id)
            lora_request = None
            if enable_lora:
                lora_request = LoRARequest(
                    f"Lora_{index_in_batch}", 
                    index_in_batch + 1,
                    lora_path=lora_paths[index_in_batch % len(lora_paths)]
                )
                
            llm_engine.add_request(request_id, prompt, sampling_params, lora_request=lora_request)
            if verbosity:
                print(f"REQUEST: {request_id} -> ", prompt["string_prompt"], "\n")

    sys.stdout.flush()

def flush_queries():
    pass

def benchmark_using_loadgen():
    "Perform the benchmark using python API for the LoadGen library"


    total_examples  = len(all_prompts)
    print("Total examples available: {}".format(total_examples))

    loadgen_dataset_size = dataset_size or total_examples
    print("Number of selected samples: {}".format(loadgen_dataset_size))

    scenario = {
        #'SingleStream':     lg.TestScenario.SingleStream,
        #'MultiStream':      lg.TestScenario.MultiStream,
        'Server':           lg.TestScenario.Server,
        'Offline':          lg.TestScenario.Offline,
    }[scenario_str]

    mode = {
        'AccuracyOnly':     lg.TestMode.AccuracyOnly,
        'PerformanceOnly':  lg.TestMode.PerformanceOnly,
        'SubmissionRun':    lg.TestMode.SubmissionRun,
    }[mode_str]

    ts = lg.TestSettings()

    if(user_conf_path):
        ts.FromConfig(user_conf_path, mlperf_model_name, scenario_str)

    ts.scenario = scenario
    ts.mode     = mode

    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(loadgen_dataset_size, buffer_size, load_query_samples, unload_query_samples)
    log_settings = lg.LogSettings()
    log_settings.enable_trace = False

    worker_thread = threading.Thread(target=worker_code, args=(), daemon=False)
    worker_thread.start()
    print("Worker thread started.")

    lg.StartTestWithLogSettings(sut, qsl, ts, log_settings)

    global worker_is_needed     # primitive variable poll signalling
    worker_is_needed = False
    worker_thread.join()
    print("Worker thread joined")

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

try:
    benchmark_using_loadgen()
except Exception as e:
    print('Error: {}'.format(e))
    raise e
