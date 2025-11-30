Profiling `reproduce` run (Colab-friendly)

Overview

- Place the `scripts/` folder into your Colab workspace (e.g., copy into Drive and `!cp` into the VM or `git clone` the repo).
- The scripts help capture three traces/artifacts:
  - `trace_reproduce.json` — Chrome trace from `torch.autograd.profiler`
  - `pyspy_reproduce.svg` — CPU sampling flamegraph from `py-spy`
  - `nsys_reproduce.qdrep` — Nsight Systems report (if `nsys` installed)

1. Quick steps to run in Colab

Copy or mount your repo into `/content/LED` and `cd` there, then run:

```bash
# (optional) install dependencies for py-spy
python -m pip install --quiet py-spy

# 1) Torch profiler (produces trace_reproduce.json)
python scripts/torch_profile_reproduce.py

# 2) py-spy CPU sampler (produces pyspy_reproduce.svg)
bash scripts/pyspy_run.sh

# 3) Nsight Systems (requires nsys available)
bash scripts/nsys_run.sh
```

2. Inspecting outputs

- Open `trace_reproduce.json` in Chrome: `chrome://tracing` or in TensorBoard (trace handler).
- Display `pyspy_reproduce.svg` in notebook or download and open in browser.
- For Nsight `qdrep`, open in Nsight Systems GUI or run `nsys stats nsys_reproduce.qdrep`.

3. Tips

- If full `reproduce` is long, modify `scripts/torch_profile_reproduce.py` to call a shorter test (e.g., add an argument to run 1 batch). This script is intentionally minimal to avoid changing repository code paths.
- Collect `nvidia-smi` output before/after runs to record peak GPU memory.

4. Next steps after profiling

- Share the three artifacts (trace JSON, py-spy SVG, nsys qdrep) and I will analyze and suggest targeted code edits.
