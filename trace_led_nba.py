import argparse
from trainer import train_led_trajectory_augment_input as led
import torch
import os


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument(
        "--use_amp", action="store_true", help="Enable AMP mixed precision"
    )
    parser.add_argument("--max_epochs", type=int, default=128)

    parser.add_argument("--cfg", default="led_augment")
    parser.add_argument("--gpu", type=int, default=0, help="Specify which GPU to use.")
    parser.add_argument(
        "--workers", type=int, default=-1, help="Number of data loading workers."
    )
    parser.add_argument(
        "--train", type=int, default=1, help="Whether train or evaluate."
    )

    parser.add_argument(
        "--info",
        type=str,
        default="",
        help="Name of the experiment. It will be used in file creation.",
    )
    return parser.parse_args()


def main(config):
    t = led.Trainer(config)
    if config.train == 1:
        # Profile the training run and export a chrome trace
        # Clear caches and reset peak stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print("Starting profiler and running fit() ...")
        with torch.autograd.profiler.profile(
            use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            try:
                t.fit()
                print("Training finished successfully.")
            except Exception as e:
                print("Exception during fit():", e)
                raise
            print("Training fit() completed, exiting profiler.")

        if torch.cuda.is_available():
            print("Synchronizing CUDA ...")
            torch.cuda.synchronize()
            print(
                "Max GPU memory allocated (bytes):", torch.cuda.max_memory_allocated()
            )

        # Print a short summary and export trace
        try:
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
            )
        except Exception:
            print("no summary table.")

        out_path = os.path.join(os.getcwd(), "trace_reproduce_train.json")
        try:
            print("exporting chrome trace ...")
            prof.export_chrome_trace(out_path)
            print("Exported chrome trace to:", out_path)
        except Exception as e:
            print("Could not export chrome trace:", e)
    else:
        # t.save_data()
        # t.test_single_model()
        # Clear caches and reset peak stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print("Starting profiler and running test_single_model() ...")
        with torch.autograd.profiler.profile(
            use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            try:
                t.test_single_model()
            except Exception as e:
                print("Exception during test_single_model():", e)
                raise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(
                "Max GPU memory allocated (bytes):", torch.cuda.max_memory_allocated()
            )

        # Print a short summary and export trace
        try:
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
            )
        except Exception:
            pass

        out_path = os.path.join(os.getcwd(), "trace_reproduce.json")
        try:
            prof.export_chrome_trace(out_path)
            print("Exported chrome trace to:", out_path)
        except Exception as e:
            print("Could not export chrome trace:", e)


if __name__ == "__main__":
    config = parse_config()
    main(config)
