#!/bin/bash
# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# usage: scripts/ci/stage-test-data.sh [data dir] [parts...]
# Stages the data and PyTorch references of the accelerator data gates on a machine, once: what the
# PR workflow (.github/workflows/pr-tests.yml) passes as -Djitnn.mnist, -Djitnn.cifar10, -Djitnn.text,
# -Djitnn.hf and -Djitnn.llm.model. Run it from the jitNN checkout on each self-hosted runner; rerun it after the
# reference scripts change. The data dir defaults to ~/jitnn-data (the workflow's JITNN_DATA);
# parts are any of mnist cifar10 text hf llm (default: all).
# Environment:
#   PYTHON              a Python 3.10+ to make the venv with (default python3)
#   JITNN_TORCH_DEVICE  cuda or mps to make the references on a GPU (the scripts default to CPU)
#   TORNADOVM_DOCS      a TornadoVM checkout whose docs/source/*.rst form the text corpus
#                       (default: a shallow clone of github.com/beehive-lab/TornadoVM)
#   JITNN_LLM_MODEL_URL the GGUF model for LlmAccelTest, saved as llm/model.gguf
#                       (default: Llama 3.2 1B Instruct, Q8_0, 1.3 GB)
set -euo pipefail
JITNN=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${1:-$HOME/jitnn-data}
shift || true
PARTS=${*:-mnist cifar10 text hf llm}
mkdir -p "$DATA"
DATA=$(cd "$DATA" && pwd)

VENV=$DATA/torch-venv
if [ ! -x "$VENV/bin/python" ]; then
    "${PYTHON:-python3}" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q torch numpy transformers tokenizers huggingface_hub safetensors
fi
PY=$VENV/bin/python

has() { [[ " $PARTS " == *" $1 "* ]]; }
# downloads to <file>.part first, so an interrupted download is not taken for a finished one
fetch() { [ -f "$2" ] || { curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }

if has mnist; then
    d=$DATA/mnist; mkdir -p "$d"
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
        fetch "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz" "$d/$f.gz"
    done
    "$PY" "$JITNN/scripts/pytorch/mnist_mlp.py" "$d" "$d/mnist_mlp_pytorch.f32"
    "$PY" "$JITNN/scripts/pytorch/mnist_train.py" lenet "$d" "$d/lenet_pytorch.f32"
    for m in resnet18 mobilenetv2; do
        "$PY" "$JITNN/scripts/pytorch/cnn_train.py" "$m" "$d" "$d/${m}_pytorch.f32" 200 mnist
        "$PY" "$JITNN/scripts/pytorch/cnn_grad64.py" "$m" "$d" "$d/${m}_pytorch.f32" mnist
    done
    for m in vae dcgan unet; do
        "$PY" "$JITNN/scripts/pytorch/generative_recurrent.py" train "$m" "$d" "$d/${m}_pytorch.f32"
        "$PY" "$JITNN/scripts/pytorch/generative_recurrent.py" grad64 "$m" "$d" "$d/${m}_pytorch.f32"
    done
fi

if has cifar10; then
    d=$DATA/cifar10; mkdir -p "$d"
    if [ ! -f "$d/cifar-10-batches-bin/test_batch.bin" ]; then
        fetch https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz "$d/cifar-10-binary.tar.gz"
        tar xzf "$d/cifar-10-binary.tar.gz" -C "$d"
    fi
    for m in resnet18 mobilenetv2; do
        "$PY" "$JITNN/scripts/pytorch/cnn_train.py" "$m" "$d" "$d/${m}_cifar10_pytorch.f32" 0 cifar10
        "$PY" "$JITNN/scripts/pytorch/cnn_grad64.py" "$m" "$d" "$d/${m}_cifar10_pytorch.f32" cifar10
    done
fi

if has text; then
    d=$DATA/text; mkdir -p "$d"
    if [ ! -f "$d/corpus.txt" ]; then
        docs=${TORNADOVM_DOCS:-}
        if [ -z "$docs" ]; then
            docs=$DATA/tornadovm-src
            [ -d "$docs" ] || git clone -q --depth 1 https://github.com/beehive-lab/TornadoVM "$docs"
        fi
        cat "$docs"/docs/source/*.rst > "$d/corpus.txt"
    fi
    for m in gpt2 llama bert; do
        "$PY" "$JITNN/scripts/pytorch/transformers.py" train  "$m" "$d/corpus.txt" "$d/${m}_pytorch.f32" 300
        "$PY" "$JITNN/scripts/pytorch/transformers.py" grad64 "$m" "$d/corpus.txt" "$d/${m}_pytorch.f32"
    done
    for m in lstm gru; do
        "$PY" "$JITNN/scripts/pytorch/generative_recurrent.py" train "$m" "$d" "$d/${m}_pytorch.f32"
        "$PY" "$JITNN/scripts/pytorch/generative_recurrent.py" grad64 "$m" "$d" "$d/${m}_pytorch.f32"
    done
fi

if has hf; then
    d=$DATA/hf; mkdir -p "$d"
    "$PY" "$JITNN/scripts/hf/hf_reference.py" "$d"
    "$PY" "$JITNN/scripts/hf/hf_llama_reference.py" "$d"
fi

if has llm; then
    d=$DATA/llm; mkdir -p "$d"
    fetch "${JITNN_LLM_MODEL_URL:-https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf}" "$d/model.gguf"
fi

echo "staged in $DATA:"
ls "$DATA"/*/ 2>/dev/null | head -50
