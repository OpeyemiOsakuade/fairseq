# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import tqdm
from npy_append_array import NpyAppendArray


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("feature_utils")


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)
    return iterate, len(lines)


def dump_feature(reader, generator, num, split, nshard, rank, feat_dir):
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    skipped_files = []  # Keep track of skipped files
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            try:
                feat = reader.get_feats(path, nsample)
                if feat is None:
                    logger.warning(f"Skipping file {path} due to invalid features.")
                    skipped_files.append(path)
                    continue
                feat_f.append(feat.cpu().numpy())
                leng_f.write(f"{len(feat)}\n")
            except Exception as e:
                logger.error(f"Error processing file {path}: {e}")
                skipped_files.append(path)
                continue
    logger.info(f"Finished processing. Skipped {len(skipped_files)} files.")
    if skipped_files:
        skipped_log_path = f"{feat_dir}/{split}_{rank}_{nshard}_skipped.log"
        with open(skipped_log_path, "w") as skipped_log:
            skipped_log.write("\n".join(skipped_files))
        logger.info(f"Skipped file list saved to {skipped_log_path}")
    logger.info("finished successfully")


