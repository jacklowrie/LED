#!/usr/bin/env bash
set -e

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found. Try installing via: apt-get install -y nvidia-nsight-systems"
  exit 1
fi

echo "Running Nsight Systems. Output: nsys_reproduce.qdrep"
nsys profile -o nsys_reproduce --trace=cuda,os python main_led_nba.py --cfg led_augment --gpu 0 --workers 2 --train 0 --info reproduce

echo "nsys output: nsys_reproduce.qdrep"
