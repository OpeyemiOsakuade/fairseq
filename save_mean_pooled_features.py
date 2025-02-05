import argparse
import logging
import os

import numpy as np

from examples.textless_nlp.gslm.speech2unit.clustering.utils import (
    get_audio_files,
)


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Save mean-pooled features to an output file."
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the file containing mean-pooled features (features.npy).",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_quantized_file_path",
        required=True,
        type=str,
        help="File path of quantized output.",
    )
    parser.add_argument(
        "--hide-fname", action='store_true',
        help="Hide file names in the output file."
    )
    return parser


def main(args, logger):
    # Load mean-pooled features
    logger.info(f"Loading mean-pooled features from {args.features_path}...")
    features_batch = np.load(args.features_path)
    logger.info(f"Loaded {len(features_batch)} mean-pooled feature vectors.")

    # Load filenames from manifest file
    logger.info(f"Loading filenames from {args.manifest_path}...")
    with open(args.manifest_path, "r") as f:
        fnames = [line.strip().split('\t')[0] for line in f]  # Adjust according to your manifest format
    logger.info(f"Loaded {len(fnames)} filenames.")

    # Check if the number of filenames matches the number of feature vectors
    if len(fnames) != len(features_batch):
        logger.error("The number of filenames in the manifest file does not match the number of feature vectors.")
        logger.error(f"Number of filenames: {len(fnames)}, Number of feature vectors: {len(features_batch)}")
        return

    os.makedirs(os.path.dirname(args.out_quantized_file_path), exist_ok=True)
    logger.info(f"Writing features to {args.out_quantized_file_path}")
    with open(args.out_quantized_file_path, "w") as fout:
        for i, feats in enumerate(features_batch):
            feats_str = " ".join(str(f) for f in feats)
            base_fname = os.path.basename(fnames[i])
            if not args.hide_fname:
                fout.write(f"{base_fname}|{feats_str}\n")
            else:
                fout.write(f"{feats_str}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
