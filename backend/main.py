"""
main.py
-------
  Step 1 — pre-compute BERT embeddings (once per dataset)
      python bert_encoder.py --dataset plantvillage
      python bert_encoder.py --dataset plantdoc
      python bert_encoder.py --dataset plantwild

  Step 2 — train
      python main.py --dataset plantvillage --train
      python main.py --dataset plantdoc --train
      python main.py --dataset plantwild --train

  Step 3 — evaluate best checkpoint
      python main.py --dataset plantvillage --eval_only
      python main.py --dataset plantdoc --eval_only
      python main.py --dataset plantwild --eval_only
"""

import argparse

from bert_encoder import CACHE_DIR
from train import run_training, FROZEN_LAYERS
from evaluate import run_evaluation, run_predict


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Crop Disease Classifier")

    parser.add_argument(
        "--dataset",
        default="plantvillage",
        choices=["plantvillage", "plantdoc", "plantwild"],
        help="Which dataset to train / evaluate on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--frozen_layers",
        type=int,
        default=FROZEN_LAYERS,
        help="Number of ViT blocks to freeze from the bottom (0-12)",
    )
    parser.add_argument(
        "--cache_dir",
        default=str(CACHE_DIR),
        help="Directory containing pre-computed BERT embedding caches",
    )
    parser.add_argument(
        "--vit_checkpoint",
        type=str,
        default=None,
        help="Optional path to a pre-trained ViT checkpoint (.pt)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model on the specified dataset",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoints/<dataset>_latest.pt",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluate the best saved checkpoint",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        metavar="IMAGE_PATH",
        help="Path to image for classification (use with --text)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        metavar="SYMPTOM_DESCRIPTION",
        help="Symptom description to use alongside the image (use with --predict)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.predict:
        run_predict(args)
    elif args.eval_only:
        run_evaluation(args)
    elif args.train:
        run_training(args)
    else:
        print("No action specified. Use --train, --eval_only, or --predict.")
        print("Example: python main.py --dataset plantvillage --train")


if __name__ == "__main__":
    main()