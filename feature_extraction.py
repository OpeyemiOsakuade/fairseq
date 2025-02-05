# import argparse
# import logging
# import os
# import gc
# import random
# import shutil
# import numpy as np
# import tqdm
# import torch
# import soundfile as sf
# from praatio import tgio

# from examples.textless_nlp.gslm.speech2unit.pretrained.cpc_feature_reader import (
#     CpcFeatureReader,
# )
# from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
#     HubertFeatureReader,
# )
# from examples.textless_nlp.gslm.speech2unit.pretrained.logmel_feature_reader import (
#     LogMelFeatureReader,
# )
# from examples.textless_nlp.gslm.speech2unit.pretrained.w2v2_feature_reader import (
#     Wav2VecFeatureReader,
# )


# def get_parser():
#     parser = argparse.ArgumentParser(
#         description="Compute and dump acoustic features for vowel frames."
#     )
#     parser.add_argument(
#         "--feature_type",
#         type=str,
#         choices=["logmel", "hubert", "w2v2", "cpc"],
#         default=None,
#         help="Acoustic feature type",
#     )
#     parser.add_argument(
#         "--manifest_path",
#         type=str,
#         default=None,
#         help="Manifest file containing the root dir and file names",
#     )
#     parser.add_argument(
#         "--out_features_path",
#         type=str,
#         default=None,
#         help="Features file path to write to",
#     )
#     parser.add_argument(
#         "--out_labels_path",
#         type=str,
#         default=None,
#         help="Labels file path to write to",
#     )
#     parser.add_argument(
#         "--checkpoint_path",
#         type=str,
#         help="Pretrained acoustic model checkpoint",
#     )
#     parser.add_argument(
#         "--layer",
#         type=int,
#         help="The layer of the pretrained model to extract features from",
#         default=-1,
#     )
#     parser.add_argument(
#         "--sample_pct",
#         type=float,
#         help="Percent data to use for feature extraction",
#         default=0.1,
#     )
#     parser.add_argument(
#         "--textgrid_dir",
#         type=str,
#         help="Directory containing the TextGrid files for alignments",
#     )
#     parser.add_argument(
#         "--vowel_set",
#         type=str,
#         nargs='+',
#         default=['a', 'e', 'i', 'o', 'u'],
#         help="Set of vowels to extract",
#     )
#     return parser


# def get_logger():
#     log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
#     logging.basicConfig(format=log_format, level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     return logger


# def get_feature_reader(feature_type):
#     if feature_type == "logmel":
#         return LogMelFeatureReader
#     elif feature_type == "hubert":
#         return HubertFeatureReader
#     elif feature_type == "w2v2":
#         return Wav2VecFeatureReader
#     elif feature_type == "cpc":
#         return CpcFeatureReader
#     else:
#         raise NotImplementedError(f"{feature_type} is not supported.")


# def load_mfa_alignments(textgrid_dir, vowel_set):
#     alignments = {}
#     for filename in os.listdir(textgrid_dir):
#         if filename.endswith(".TextGrid"):
#             filepath = os.path.join(textgrid_dir, filename)
#             tg = tgio.openTextgrid(filepath)
#             phones_tier = tg.tierDict['phones']
#             phoneme_times = [(interval.label, interval.start, interval.end) for interval in phones_tier.entryList if interval.label in vowel_set]
#             alignments[filename.split('.')[0]] = phoneme_times
#     return alignments


# def get_feature_iterator(
#     feature_type, checkpoint_path, layer, manifest_path, sample_pct, alignments
# ):
#     feature_reader_cls = get_feature_reader(feature_type)
#     with open(manifest_path, "r") as fp:
#         lines = fp.read().split("\n")
#         root = lines.pop(0).strip()
#         file_path_list = [
#             os.path.join(root, line.split("\t")[0])
#             for line in lines
#             if len(line) > 0
#         ]
#         if sample_pct < 1.0:
#             file_path_list = random.sample(
#                 file_path_list, int(sample_pct * len(file_path_list))
#             )
#         num_files = len(file_path_list)
#         reader = feature_reader_cls(
#             checkpoint_path=checkpoint_path, layer=layer
#         )

#         def iterate():
#             for file_path in file_path_list:
#                 file_id = os.path.splitext(os.path.basename(file_path))[0]
#                 segments = alignments.get(file_id, [])
#                 if segments:
#                     waveform, sample_rate = sf.read(file_path)
#                     for label, start, end in segments:
#                         start_frame = int(start * sample_rate)
#                         end_frame = int(end * sample_rate)
#                         segment_waveform = waveform[start_frame:end_frame]
#                         if segment_waveform.size > 0:
#                             feats = reader.get_feats(segment_waveform)
#                             yield feats.cpu().numpy(), label

#     return iterate, num_files


# def get_features_and_labels(
#     feature_type, checkpoint_path, layer, manifest_path, sample_pct, alignments
# ):
#     generator, num_files = get_feature_iterator(
#         feature_type=feature_type,
#         checkpoint_path=checkpoint_path,
#         layer=layer,
#         manifest_path=manifest_path,
#         sample_pct=sample_pct,
#         alignments=alignments,
#     )
#     iterator = generator()

#     features_list = []
#     labels_list = []
#     for features, label in tqdm.tqdm(iterator, total=num_files):
#         if features.size > 0:
#             features_list.append(features)
#             labels_list.append(label)
#         else:
#             print("No features extracted for this segment.")

#     # Explicit clean up
#     del iterator
#     del generator
#     gc.collect()
#     torch.cuda.empty_cache()

#     return np.array(features_list), np.array(labels_list)


# def get_and_dump_features_and_labels(
#     feature_type,
#     checkpoint_path,
#     layer,
#     manifest_path,
#     sample_pct,
#     out_features_path,
#     out_labels_path,
#     alignments,
# ):
#     # Feature and label extraction
#     features, labels = get_features_and_labels(
#         feature_type=feature_type,
#         checkpoint_path=checkpoint_path,
#         layer=layer,
#         manifest_path=manifest_path,
#         sample_pct=sample_pct,
#         alignments=alignments,
#     )

#     # Save features and labels
#     out_dir_path = os.path.dirname(out_features_path)
#     os.makedirs(out_dir_path, exist_ok=True)
#     shutil.copyfile(
#         manifest_path,
#         os.path.join(out_dir_path, os.path.basename(manifest_path)),
#     )
#     np.save(out_features_path, features)
#     np.save(out_labels_path, labels)

#     return features, labels


# if __name__ == "__main__":
#     """
#     Example command:
#     python script.py \
#         --feature_type hubert \
#         --manifest_path /path/to/manifest.tsv \
#         --out_features_path /path/to/output_features.npy \
#         --out_labels_path /path/to/output_labels.npy \
#         --checkpoint_path /path/to/checkpoint \
#         --layer -1 \
#         --sample_pct 0.1 \
#         --textgrid_dir /path/to/textgrids \
#         --vowel_set a e i o u
#     """
#     parser = get_parser()
#     args = parser.parse_args()
#     logger = get_logger()
#     logger.info(args)

#     logger.info(f"Extracting {args.feature_type} acoustic features...")

#     # Load MFA alignments
#     alignments = load_mfa_alignments(args.textgrid_dir, set(args.vowel_set))
#     print("Alignments:", alignments)  # Debugging statement

#     features, labels = get_and_dump_features_and_labels(
#         feature_type=args.feature_type,
#         checkpoint_path=args.checkpoint_path,
#         layer=args.layer,
#         manifest_path=args.manifest_path,
#         sample_pct=args.sample_pct,
#         out_features_path=args.out_features_path,
#         out_labels_path=args.out_labels_path,
#         alignments=alignments,
#     )
#     logger.info(f"Saved extracted features at {args.out_features_path}")
#     logger.info(f"Saved extracted labels at {args.out_labels_path}")
#     logger.info(f"Features shape = {features.shape}")
#     logger.info(f"Labels shape = {labels.shape}")

import argparse
import logging
import os
import gc
import random
import shutil
import numpy as np
import tqdm
import torch
import soundfile as sf
from praatio import tgio

from examples.textless_nlp.gslm.speech2unit.pretrained.cpc_feature_reader import (
    CpcFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.logmel_feature_reader import (
    LogMelFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.w2v2_feature_reader import (
    Wav2VecFeatureReader,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute and dump acoustic features for vowel frames."
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_features_path",
        type=str,
        default=None,
        help="Features file path to write to",
    )
    parser.add_argument(
        "--out_labels_path",
        type=str,
        default=None,
        help="Labels file path to write to",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Pretrained acoustic model checkpoint",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--sample_pct",
        type=float,
        help="Percent data to use for feature extraction",
        default=0.1,
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        help="Directory containing the TextGrid files for alignments",
    )
    parser.add_argument(
        "--vowel_set",
        type=str,
        nargs='+',
        default=['a', 'e', 'i', 'o', 'u'],
        help="Set of vowels to extract",
    )
    return parser


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_feature_reader(feature_type):
    if feature_type == "logmel":
        return LogMelFeatureReader
    elif feature_type == "hubert":
        return HubertFeatureReader
    elif feature_type == "w2v2":
        return Wav2VecFeatureReader
    elif feature_type == "cpc":
        return CpcFeatureReader
    else:
        raise NotImplementedError(f"{feature_type} is not supported.")


def load_mfa_alignments(textgrid_dir, vowel_set):
    alignments = {}
    for filename in os.listdir(textgrid_dir):
        if filename.endswith(".TextGrid"):
            filepath = os.path.join(textgrid_dir, filename)
            tg = tgio.openTextgrid(filepath)
            phones_tier = tg.tierDict['phones']
            phoneme_times = [(interval.label, interval.start, interval.end) for interval in phones_tier.entryList if interval.label in vowel_set]
            alignments[filename.split('.')[0]] = phoneme_times
    return alignments

# def categorize_vowel(vowel):
#     """Categorize vowel into low, high, and mid-tone."""
#     if vowel.endswith('L'):
#         return 'low'
#     elif vowel.endswith('H'):
#         return 'high'
#     else:
#         return 'mid'

def categorize_vowel(vowel):
    """Collapse vowel into the mid tones: a, e, eR, i, o, oR, u."""
    vowel_map = {
        'a': 'a', 'aH': 'a', 'aL': 'a',
        'e': 'e', 'eH': 'e', 'eL': 'e',
        'eR': 'eR', 'eRH': 'eR', 'eRL': 'eR',
        'i': 'i', 'iH': 'i', 'iL': 'i',
        'o': 'o', 'oH': 'o', 'oL': 'o',
        'oR': 'oR', 'oRH': 'oR', 'oRL': 'oR',
        'u': 'u', 'uH': 'u', 'uL': 'u'
    }
    
    if vowel in vowel_map:
        return vowel_map[vowel]
    else:
        raise ValueError(f"Unexpected vowel: {vowel}")



def get_feature_iterator(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, alignments, min_length=4000
):
    feature_reader_cls = get_feature_reader(feature_type)
    with open(manifest_path, "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        file_path_list = [
            os.path.join(root, line.split("\t")[0])
            for line in lines
            if len(line) > 0
        ]
        if sample_pct < 1.0:
            file_path_list = random.sample(
                file_path_list, int(sample_pct * len(file_path_list))
            )
        num_files = len(file_path_list)
        reader = feature_reader_cls(
            checkpoint_path=checkpoint_path, layer=layer
        )

        def iterate():
            for file_path in file_path_list:
                file_id = os.path.splitext(os.path.basename(file_path))[0]
                segments = alignments.get(file_id, [])
                if segments:
                    waveform, sample_rate = sf.read(file_path)
                    for label, start, end in segments:
                        start_frame = int(start * sample_rate)
                        end_frame = int(end * sample_rate)
                        segment_waveform = waveform[start_frame:end_frame]
                        if segment_waveform.size >= min_length:
                            feats = reader.get_feats(segment_waveform)
                            yield feats.cpu().numpy(), categorize_vowel(label)
    return iterate, num_files


def get_features_and_labels(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, alignments, min_length=4000
):
    generator, num_files = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        alignments=alignments,
        min_length=min_length
    )
    iterator = generator()

    features_list = []
    labels_list = []
    for features, label in tqdm.tqdm(iterator, total=num_files):
        if features.size > 0:
            features_list.append(features)
            labels_list.append(label)
        else:
            print("No features extracted for this segment.")

    # Explicit clean up
    del iterator
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    return np.array(features_list), np.array(labels_list)


def get_and_dump_features_and_labels(
    feature_type,
    checkpoint_path,
    layer,
    manifest_path,
    sample_pct,
    out_features_path,
    out_labels_path,
    alignments,
    min_length=4000
):
    # Feature and label extraction
    features, labels = get_features_and_labels(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        alignments=alignments,
        min_length=min_length
    )

    # Save features and labels
    out_dir_path = os.path.dirname(out_features_path)
    os.makedirs(out_dir_path, exist_ok=True)
    shutil.copyfile(
        manifest_path,
        os.path.join(out_dir_path, os.path.basename(manifest_path)),
    )
    np.save(out_features_path, features)
    np.save(out_labels_path, labels)

    return features, labels


if __name__ == "__main__":
    """
    Example command:
    python script.py \
        --feature_type hubert \
        --manifest_path /path/to/manifest.tsv \
        --out_features_path /path/to/output_features.npy \
        --out_labels_path /path/to/output_labels.npy \
        --checkpoint_path /path/to/checkpoint \
        --layer -1 \
        --sample_pct 0.1 \
        --textgrid_dir /path/to/textgrids \
        --vowel_set a e i o u
    """
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)

    logger.info(f"Extracting {args.feature_type} acoustic features...")

    # Load MFA alignments
    alignments = load_mfa_alignments(args.textgrid_dir, set(args.vowel_set))
    print("Alignments:", alignments)  # Debugging statement

    features, labels = get_and_dump_features_and_labels(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint_path,
        layer=args.layer,
        manifest_path=args.manifest_path,
        sample_pct=args.sample_pct,
        out_features_path=args.out_features_path,
        out_labels_path=args.out_labels_path,
        alignments=alignments,
    )
    logger.info(f"Saved extracted features at {args.out_features_path}")
    logger.info(f"Saved extracted labels at {args.out_labels_path}")
    logger.info(f"Features shape = {features.shape}")
    logger.info(f"Labels shape = {labels.shape}")
