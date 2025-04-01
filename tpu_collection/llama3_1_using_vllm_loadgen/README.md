# Running Llama3.1 model on TPU

## Define a local workspace directory
```
export WORKSPACE_DIR=/workspace
mkdir -p ${WORKSPACE_DIR}
mkdir -p ${WORKSPACE_DIR}/${USER}
```

## Install KRAI [AXS](https://github.com/krai/axs)

### Clone

Clone the AXS repository under `${WORKSPACE_DIR}/$USER`:
```
git clone https://github.com/krai/axs ${WORKSPACE_DIR}/$USER/axs
```

### Init

Define environment variables in your `~/.bashrc`:
```
echo "

# AXS.
export WORKSPACE_DIR=/workspace
export PATH=${WORKSPACE_DIR}/${USER}/axs:${PATH}
export AXS_WORK_COLLECTION=${WORKSPACE_DIR}/${USER}/work_collection

" >> ~/.bashrc
```

### Install rclone
```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

### Configure rclone
```
rclone config create mlc-inference s3 provider=Cloudflare \
access_key_id=f65ba5eef400db161ea49967de89f47b \
secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b \
endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
```

### Test
```
source ~/.bashrc
axs version
```

## Install vLLM

```
cd ~
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-tpu.txt
VLLM_TARGET_DEVICE="tpu" python setup.py develop
```

## Enable conda for vLLM
```
conda create -n vllm python=3.10 -y
conda activate vllm
pip install mlcommons-loadgen
```
and update `work_collection/python3.10_tool/data_axs.json` to conda python path
>    "tool_path": "/mnt/ssd4/anaconda3/envs/vllm/bin/python",


## Import public AXS repositories

Import the required public repos into your work collection:

```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2gcp
```

## Download artifacts

Use a [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens) (`export HF_TOKEN=...`) to download the model and its tokenizer.

### Model
The benchmark supports the following models:
* Llama3.1-8b
* Llama3.1-70b
To download a desired model, set the parameter `MODEL_FAMILY` to `llama3_1` and run
```
export MODEL_FAMILY=<model_family>
export VARIANT=<model_variant>

axs byquery downloaded,hf_model,model_family=${MODEL_FAMILY},variant=${VARIANT},hf_token=${HF_TOKEN}
```
You can also provide your own model by adding `,model_path=...` to the commands below.

### Tokenizer
```
axs byquery downloaded,hf_tokeniser,model_family=${MODEL_FAMILY},variant=${VARIANT},hf_token=${HF_TOKEN}
```

### Dataset (OpenOrca)
```
axs byquery shell_tool,can_download_url
axs byquery shell_tool,can_extract_tar
axs byquery downloaded,dataset_name=openorca,model_family=${MODEL_FAMILY},variant=${VARIANT}
```

## Benchmark
```
export DATASET_PATH_PKL=<open-orca_pkl_file_path>
export MODEL_PATH=<path/to/your_local_model>
export MAX_LORAS=${MAX_LORAS}
export LORA_PATHS=<lora_path1,lora_path2,..>
export LORA_RANK=${LORA_RANK}
```


### Offline

#### Accuracy

```
axs byquery loadgen_output,framework=vllm,hostname=host_tpu,device=tpu,backend=tpu,tensor_parallel_size=1,task=llama3_1,\
model_name=llama3_1,model_variant=${MODEL_VARIANT},mlperf_model_name=llama2-70b,dataset_path=${DATASET_PATH_PKL},model_path=${MODEL_PATH},loadgen_scenario=Offline,\
loadgen_mode=AccuracyOnly,loadgen_target_qps=3.25,loadgen_dataset_size=1024,loadgen_buffer_size=24576,loadgen_sample_concatenate_permutation=0,\
loadgen_min_duration_s=600,loadgen_min_query_count=1000,max_num_seqs=128,batch_size=1,max_num_batched_tokens=1024
```

#### Performance
```
axs byquery loadgen_output,framework=vllm,hostname=host_tpu,device=tpu,backend=tpu,tensor_parallel_size=1,task=llama3_1,model_name=llama3_1,\
model_variant=${MODEL_VARIANT},mlperf_model_name=llama2-70b,dataset_path=${DATASET_PATH_PKL},model_path=${MODEL_PATH},loadgen_scenario=Offline,\
loadgen_mode=PerformanceOnly,loadgen_target_qps=3.25,loadgen_dataset_size=1024,loadgen_buffer_size=24576,loadgen_sample_concatenate_permutation=0,\
loadgen_min_duration_s=600,loadgen_min_query_count=1000,max_num_seqs=128,batch_size=1,max_num_batched_tokens=1024
```

##### Enable lora:

```
axs byquery loadgen_output,framework=vllm,hostname=host_tpu,device=tpu,backend=tpu,tensor_parallel_size=1,task=llama3_1,model_name=llama3_1,\
model_variant=${MODEL_VARIANT},mlperf_model_name=llama2-70b,dataset_path=${DATASET_PATH_PKL},model_path=${MODEL_PATH},loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,\
loadgen_target_qps=3.25,loadgen_dataset_size=1024,loadgen_buffer_size=24576,loadgen_sample_concatenate_permutation=0,loadgen_min_duration_s=600,\
loadgen_min_query_count=1000,max_num_seqs=128,batch_size=1,max_num_batched_tokens=1024,\
enable_lora+,max_loras=1,lora_paths:=Sayantan54321/llama3.1_8b_finetuned,dtype=bfloat16,max_lora_rank=16
```
