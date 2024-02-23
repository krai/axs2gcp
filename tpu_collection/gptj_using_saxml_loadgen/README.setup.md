# Set up Google Cloud TPU instances for MLPerf Inference benchmarking

## Set up `gcloud`
```
sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
gcloud auth login --no-launch-browser
```
## Define environment variables
```
export PROJECT_ID=<your project id>
export ZONE=<your zone e.g. us-west4-a>
export SERVICE_ACCOUNT=<your service account>
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
export ACCELERATOR_TYPE=v5litepod-4
export QUOTA_TYPE=
export QUEUED_RESOURCE_ID=mlperf
export TPU_NAME=mlperf
```
## Create a VM
```
gcloud alpha compute tpus queued-resources create ${QUEUED_RESOURCE_ID} \
--node-id ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT_ID} \
--accelerator-type ${ACCELERATOR_TYPE} \
--runtime-version ${RUNTIME_VERSION} \
--service-account ${SERVICE_ACCOUNT}
```
## Create a storage bucket for the admin server
```
export PROJECT_ID=tpu-evaluation
export SAX_ADMIN_SERVER_STORAGE_BUCKET="sax_admin_server_storage_bucket"
gcloud storage buckets create gs://${SAX_ADMIN_SERVER_STORAGE_BUCKET} --project=${PROJECT_ID}
```
## Create a storage bucket for the model server
```
export PROJECT_ID=tpu-evaluation
export SAX_MODEL_SERVER_STORAGE_BUCKET="sax_model_server_storage_bucket"
gcloud storage buckets create gs://${SAX_MODEL_SERVER_STORAGE_BUCKET} --project=${PROJECT_ID}
```
## Connect to the VM
```
export TPU_NAME=mlperf
export ZONE=us-west4-a
export PROJECT_ID=tpu-evaluation
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```
## Update
```
sudo apt update -y
sudo apt upgrade -y
sudo apt install \
  git vim htop tree net-tools \
  libssl-dev openssl libffi-dev libbz2-dev \
  zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev
```
