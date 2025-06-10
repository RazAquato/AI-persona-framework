#!/usr/bin/env python3
# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse, subprocess, yaml, os
# Load environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_PATH = os.path.join(BASE_DIR, "..", "llama.cpp")

with open(os.path.join(BASE_DIR, "config", "model_configs.yaml"), "r") as f:
    configs = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

conf = configs.get(args.model)
if not conf:
    print(f"Model '{args.model}' not found, using default 'tinyllama'")
    args.model = "tinyllama"
    conf = configs["tinyllama"]

cmd = [
    os.path.join(LLAMA_PATH, "build", "bin", "llama-server"),
    "-m", conf["path"],
    "--port", str(conf["port"]),
    "--ctx-size", str(conf["ctx_size"]),
    "--n-gpu-layers", str(conf["n_gpu_layers"]),
    "--main-gpu", str(conf["main_gpu"]),
    "--host", "0.0.0.0",
    "--no-warmup"
]

print(f"Launching model: {args.model}")
subprocess.run(cmd)

