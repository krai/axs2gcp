# MLPerf Inference - GPT-J - Google Cloud TPUs

This implementation runs the CNN-DailyMail summarization task on [Google Cloud
TPUs](https://cloud.google.com/tpu) (Tensor Processing Units) instanes using
the [Saxml](https://github.com/google/saxml) framework.

Currently it supports the following workloads:
- gptj-99

# Detailed instructions

## Setup TPU VM instances

See [README.setup.md](README.setup.md).

## Setup axs
### Kernel
```
git clone --branch stable https://github.com/krai/axs $HOME/axs
echo "export PATH='$PATH:$HOME/axs'" >> $HOME/.bashrc
source $HOME/.bashrc
```

### [Optional] Refresh state
```
axs byname work_collection , remove
```

### Get required repositories and packages
```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2gcp
axs byquery git_repo,repo_name=saxml_git,checkout=mlperf4.0
axs byquery git_repo,repo_name=mlperf_inference_git
axs byquery python_package,package_name==mlperf_loadgen,desired_python_version===3.10
axs byquery downloaded,dataset_name=cnndm_v3_0_0
```

### Install Saxml dependencies
```
cd $(axs byquery git_repo,repo_name=saxml_git,checkout=mlperf4.0 , get_path) 
sudo saxml/tools/init_cloud_vm.sh
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

## Set up model checkpoint

**NB:** You can perform the following steps on another machine (not necessarily a TPU instance).
However, you would require to set up axs on that machine as well.
The last step (copying) should be changed accordingly.

### Convert fine-tuned PT FP32 checkpoint provided by MLCommons to a PAX checkpoint
```
axs byquery pax_model,task=gptj
```

### Quantize checkpoint
```
axs byquery quantized_pax_model,task=gptj
```

### Copy quantized checkpoint to storage bucket
```
gsutil -m cp -r \
$(axs byquery quantized_pax_model,task=gptj , get_path)/* \
gs://sax_model_server_storage_bucket/gptj_sax_int8_converted/
```

## Build and run admin server
### Define environment variables
```
export INSTANCE=""
export SAX_SRC_DIR=$(axs byquery git_repo,repo_name=saxml_git,checkout=mlperf4.0 , get_path)
export SAX_BLD_DIR=$HOME/work_collection/saxml_build
export SAX_CELL="/sax/test"
export SAX_ADMIN_SERVER_PORT=10000
export SAX_ADMIN_SERVER_STORAGE_BUCKET="sax_admin_server_storage_bucket${INSTANCE}"
export SAX_ROOT="gs://${SAX_ADMIN_SERVER_STORAGE_BUCKET}/sax-root"
export SAX_FS_ROOT="gs://${SAX_ADMIN_SERVER_STORAGE_BUCKET}/sax-fs-root"
```
### Configure admin server
```
cd ${SAX_SRC_DIR}
bazel --output_user_root=${SAX_BLD_DIR} \
  run saxml/bin:admin_config -- \
  --sax_cell=${SAX_CELL} --sax_root=${SAX_ROOT} --fs_root=${SAX_FS_ROOT} \
  --alsologtostderr
```
### Run admin server
```
cd ${SAX_SRC_DIR}
bazel --output_user_root=${SAX_BLD_DIR} \
  run saxml/bin:admin_server -- \
  --sax_cell=${SAX_CELL} --sax_root=${SAX_ROOT} --port=${SAX_ADMIN_SERVER_PORT} \
  --alsologtostderr
```

## Build and run model server
### Define environment variables
```
export INSTANCE=""
export SAX_SRC_DIR=$(axs byquery git_repo,repo_name=saxml_git,checkout=mlperf4.0 , get_path)
export SAX_BLD_DIR=$HOME/work_collection/saxml_build
export SAX_ADMIN_SERVER_STORAGE_BUCKET="sax_admin_server_storage_bucket${INSTANCE}"
export SAX_ROOT="gs://${SAX_ADMIN_SERVER_STORAGE_BUCKET}/sax-root"
export SAX_CELL="/sax/test"
export SAX_MODEL_SERVER_PORT=10001
export PLATFORM_CHIP="tpuv4"
export PLATFORM_TOPOLOGY="2x2"
```
### Run model server
```
cd ${SAX_SRC_DIR}
nohup bazel --output_user_root=${SAX_BLD_DIR} \
  run saxml/server:server -- \
  --sax_cell=${SAX_CELL} \
  --port=${SAX_MODEL_SERVER_PORT} \
  --platform_chip=${PLATFORM_CHIP} \
  --platform_topology=${PLATFORM_TOPOLOGY} \
  --alsologtostderr &
```

## Serve models

### Set up Docker
```
gcloud auth configure-docker us-docker.pkg.dev
sudo usermod -aG docker ${USER}
newgrp docker
```

### Install utility image
```
export SAX_VERSION="v1.1.0"
export SAX_UTIL_IMAGE_NAME="us-docker.pkg.dev/cloud-tpu-images/inference/sax-util"
export SAX_UTIL_IMAGE_URL="${SAX_UTIL_IMAGE_NAME}:${SAX_VERSION}"
docker pull ${SAX_UTIL_IMAGE_URL}
```

### Select model
#### Offline
```
export MODEL_NAME="gptj4tokenizedint8bs32xlawait40mb6offline"
export MODEL_CONFIG_PATH="saxml.server.pax.lm.params.gptj.GPTJ4TokenizedINT8BS32XLAWait40MB6Offline"
```
#### Server
```
export MODEL_NAME="gptj4tokenizedint8bs32xlawait40mb6server"
export MODEL_CONFIG_PATH="saxml.server.pax.lm.params.gptj.GPTJ4TokenizedINT8BS32XLAWait40MB6Server"
```

### Publish model
```
export INSTANCE=""
export SAX_VERSION="v1.1.0"
export SAX_UTIL_IMAGE_NAME="us-docker.pkg.dev/cloud-tpu-images/inference/sax-util"
export SAX_UTIL_IMAGE_URL="${SAX_UTIL_IMAGE_NAME}:${SAX_VERSION}"
export SAX_ADMIN_SERVER_STORAGE_BUCKET="sax_admin_server_storage_bucket${INSTANCE}"
export SAX_ROOT="gs://${SAX_ADMIN_SERVER_STORAGE_BUCKET}/sax-root"
export SAX_CELL="/sax/test"
export SAX_MODEL_SERVER_STORAGE_BUCKET="sax_model_server_storage_bucket"
export CHECKPOINT_PATH="gs://${SAX_MODEL_SERVER_STORAGE_BUCKET}/gptj_sax_int8_converted/checkpoint_00000000"
export REPLICA=1

docker run ${SAX_UTIL_IMAGE_URL} --sax_root=${SAX_ROOT} \
  publish ${SAX_CELL}/${MODEL_NAME} ${MODEL_CONFIG_PATH} ${CHECKPOINT_PATH} ${REPLICA}
```

### Unpublish model
```
export INSTANCE=""
export SAX_VERSION="v1.1.0"
export SAX_UTIL_IMAGE_NAME="us-docker.pkg.dev/cloud-tpu-images/inference/sax-util"
export SAX_UTIL_IMAGE_URL="${SAX_UTIL_IMAGE_NAME}:${SAX_VERSION}"
export SAX_ADMIN_SERVER_STORAGE_BUCKET="sax_admin_server_storage_bucket${INSTANCE}"
export SAX_ROOT="gs://${SAX_ADMIN_SERVER_STORAGE_BUCKET}/sax-root"
export SAX_CELL="/sax/test"
export SAX_MODEL_SERVER_STORAGE_BUCKET="sax_model_server_storage_bucket"
export CHECKPOINT_PATH="gs://${SAX_MODEL_SERVER_STORAGE_BUCKET}/gptj_sax_int8_converted/checkpoint_00000000"
export REPLICA=1

docker run ${SAX_UTIL_IMAGE_URL} --sax_root=${SAX_ROOT} \
  unpublish ${SAX_CELL}/${MODEL_NAME}
```

### List published models
```
export INSTANCE=""
export SAX_VERSION="v1.1.0"
export SAX_UTIL_IMAGE_NAME="us-docker.pkg.dev/cloud-tpu-images/inference/sax-util"
export SAX_UTIL_IMAGE_URL="${SAX_UTIL_IMAGE_NAME}:${SAX_VERSION}"
export SAX_ADMIN_SERVER_STORAGE_BUCKET="sax_admin_server_storage_bucket${INSTANCE}"
export SAX_ROOT="gs://${SAX_ADMIN_SERVER_STORAGE_BUCKET}/sax-root"
export SAX_CELL="/sax/test"

docker run ${SAX_UTIL_IMAGE_URL} --sax_root=${SAX_ROOT} \
  ls ${SAX_CELL}
```

## Run experiments

**NB:** Set the target QPS (queries per second) metric to the desired value.

### Offline

#### Accuracy
```
axs byquery loadgen_output,framework=saxml,task=gptj,mlperf_model_name=gptj-99,sut_name=tpu_v5e_x4,\
loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,\
loadgen_dataset_size=13368,loadgen_buffer_size=13368 \
, get accuracy_report
```

#### Performance
```
axs byquery loadgen_output,framework=saxml,task=gptj,mlperf_model_name=gptj-99,sut_name=tpu_v5e_x4,\
loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,\
loadgen_dataset_size=13368,loadgen_buffer_size=13368,loadgen_target_qps=<desired qps> \
, get performance
```

### Server

#### Accuracy
```
axs byquery loadgen_output,framework=saxml,task=gptj,mlperf_model_name=gptj-99,sut_name=tpu_v5e_x4,\
loadgen_scenario=Server,loadgen_mode=AccuracyOnly,\
loadgen_dataset_size=13368,loadgen_buffer_size=13368,loadgen_target_qps=<desired qps> \
, get accuracy_report
```

#### Performance
```
axs byquery loadgen_output,framework=saxml,task=gptj,mlperf_model_name=gptj-99,sut_name=tpu_v5e_x4,\
loadgen_scenario=Server,loadgen_mode=AccuracyOnly,\
loadgen_dataset_size=13368,loadgen_buffer_size=13368,loadgen_target_qps=<desired qps>,loadgen_target_latency=20000 \
, get performance
```
