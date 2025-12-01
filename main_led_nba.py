import argparse
from trainer import train_led_trajectory_augment_input as led


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--max_epochs", type=int, default=128)
    parser.add_argument(
        "--use_amp", action="store_true", help="Enable AMP mixed precision"
    )

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
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results/checkpoints/led_new.p",
        help="path of the model to test (only with --train 0).",
    )
    return parser.parse_args()


def main(config):
    t = led.Trainer(config)
    if config.train == 1:
        t.fit()
    else:
        # t.save_data()
        t.test_single_model(config.model_path)


if __name__ == "__main__":
    config = parse_config()
    main(config)
