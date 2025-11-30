#!/usr/bin/env bash
set -e

echo "Installing py-spy (if missing)..."
uv add py-spy

echo "Running py-spy for main_led_nba.py (this will run the reproduce command)..."
uv run py-spy record -o pyspy_reproduce.svg -- python main_led_nba.py --cfg led_augment --gpu 0 --workers 2 --train 0 --info reproduce

echo "py-spy output: pyspy_reproduce.svg"
