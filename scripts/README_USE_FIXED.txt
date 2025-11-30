Use this fixed profiler script if you encountered an AttributeError about 'learning_rate'.

Run in Colab from repo root:

python scripts/torch_profile_reproduce_fixed.py --cfg led_augment --gpu 0 --train 0 --info reproduce

This script sets a default --learning_rate (0.002) so Trainer can be instantiated. The original script is `scripts/torch_profile_reproduce.py` but if that raises an error, use the fixed script above.
