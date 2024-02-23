# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""mlperf inference GPT-J benchmark."""

import gc
import logging
import os

import pathlib
import shutil
import time
from absl import app
from absl import flags

import backend
import mlperf_loadgen as lg

_DATASET_PATH = flags.DEFINE_string(
    "dataset_path",
    default="cnn_eval.json",
    help="path to the dataset")
_SCENARIO = flags.DEFINE_enum(
    "scenario", default="Offline",
    enum_values=["Server", "Offline"],
    help="benchmark scenario.")
_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    default="/sax/test",
    help="path to the sax admin server model.")
_NUM_CLIENT_THREADS = flags.DEFINE_integer(
    "num_client_threads", default=200, help="Number of client threads to use."
)
_ACCURACY = flags.DEFINE_bool(
    "accuracy", default=False, help="enable accuracy pass.")
_LOG_PATH = flags.DEFINE_string(
    "log_path", default="/tmp/data", help="path to the dataset.")
_LOG_INTERVAL = flags.DEFINE_integer(
    "log_interval", default=10, help="interval for logging.")
_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", default=13368,
    help="Max examples to run. For a full run this needs to be 24K in offline")
_PERF_EXAMPLES = flags.DEFINE_integer(
    "perf_examples", default=13368, help="target qps estimate")
_MLPERF_CONF_PATH = flags.DEFINE_string(
    "mlperf_conf_path",
    default = os.path.dirname( os.path.realpath(__file__) ) + '/configs/mlperf.conf',
    help="When given overrides the default mlperf.conf path.",
)
_USER_CONF_PATH = flags.DEFINE_string(
    "user_conf_override_path",
    default = os.path.dirname( os.path.realpath(__file__) ) + '/configs/user.conf',
    help="When given overrides the default user.conf path.",
)
_SHORT_NAME = flags.DEFINE_string(
    "short_name", default="test", help="Experiment identifier.")
_SAVE_RESULT = flags.DEFINE_bool(
    "save_result", default=False, help="Preserve logs.")
_STORE_SAX_ACCURACY = flags.DEFINE_bool(
    "store_sax_accuracy", default=True, help="An unpacked accuracy log, alternative to mlperf_accuracy_log.json")
_EXTRA_LOG_DIR = flags.DEFINE_bool(
    "extra_log_dir", default=True, help="Create an intermediate log directory in the given one")
_TOKENIZER_PATH = flags.DEFINE_string(
    "tokenizer_path",
    default = os.path.dirname( os.path.realpath(__file__) ) + '/tpu/tokenizer',
    help="Specify the tokenizer to be used",
)

scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main(argv):
  del argv
  settings = lg.TestSettings()
  settings.scenario = scenario_map[_SCENARIO.value]

  mlperf_conf = os.path.abspath(_MLPERF_CONF_PATH.value)
  user_conf = os.path.abspath(_USER_CONF_PATH.value)

  logging.info("Mlperf config: %s", mlperf_conf)
  logging.info("User config: %s", user_conf)

  settings.FromConfig(mlperf_conf, "gptj", _SCENARIO.value)
  settings.FromConfig(user_conf, "gptj", _SCENARIO.value)

  cfg = _SCENARIO.value
  if _ACCURACY.value:
    cfg = cfg + "_accuracy"
    settings.mode = lg.TestMode.AccuracyOnly
  else:
    cfg = cfg + "_performance"
    settings.mode = lg.TestMode.PerformanceOnly
    settings.print_timestamps = True

  if _SCENARIO.value == "Server":
    logging.info("Server mode run. See issue threads > 1")

  log_path = os.path.join(_LOG_PATH.value, cfg) if _EXTRA_LOG_DIR.value else _LOG_PATH.value
  if not pathlib.Path(log_path).is_dir():
    pathlib.Path(log_path).mkdir(parents=True)

  log_output_settings = lg.LogOutputSettings()
  log_output_settings.outdir = log_path
  log_output_settings.copy_summary_to_stdout = True
  log_settings = lg.LogSettings()
  log_settings.log_output = log_output_settings
  # Some thread is not terminating even after all the outputs are written
  log_settings.enable_trace = True

  saxml_output_log = os.path.join(_LOG_PATH.value, "sax_accuracy.json") if _STORE_SAX_ACCURACY.value else None
  sut = backend.get_sut(
      scenario=_SCENARIO.value,
      model_path=_MODEL_PATH.value,
      dataset_path=_DATASET_PATH.value,
      num_client_threads=_NUM_CLIENT_THREADS.value,
      tokenizer_path=_TOKENIZER_PATH.value,
      max_examples=_MAX_EXAMPLES.value,
      perf_examples=_PERF_EXAMPLES.value,
      log_interval=_LOG_INTERVAL.value,
      log_path=saxml_output_log,
  )

  logging.info("Start Testing!")
  lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings)
  logging.info("Test Done!")

  logging.info("Destroying SUT...")
  lg.DestroySUT(sut.sut)

  logging.info("Destroying QSL...")
  lg.DestroyQSL(sut.qsl)

  if _SAVE_RESULT.value:
    opath = os.path.join("/logs/",
                         _SHORT_NAME.value + "_" +
                         time.strftime("%m%d_%H%M%S", time.gmtime()))
    shutil.copytree(_LOG_PATH.value, opath)
    logging.info("Output at %s", opath)


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)
