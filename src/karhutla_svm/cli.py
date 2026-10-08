import argparse
import json
import sys

from .model import predict_text, train_model

DEFAULT_DATASET = "data/dataset_karhutla_normalisasi_final.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Klasifikasi sentimen Karhutla dengan TF-IDF dan SVM")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--dataset", default=DEFAULT_DATASET)
    train.add_argument("--output", default="artifacts")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict")
    predict.add_argument("--text", required=True)
    predict.add_argument("--model", default="artifacts/model.joblib")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            print(json.dumps(train_model(args.dataset, args.output, args.test_size, args.seed), indent=2))
        else:
            print(predict_text(args.text, args.model).capitalize())
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
