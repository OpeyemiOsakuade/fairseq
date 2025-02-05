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
import csv

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
        description="Compute and dump acoustic features for syllable frames."
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
        "--out_syllables_path",
        type=str,
        default=None,
        help="Syllables file path to write to",
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
        default=[
            'a', 'e', 'i', 'o', 'u',  # Simple vowels
            'an', 'anH', 'anL', 'en', 'enH', 'enL', 'eRn', 'eRnH', 'eRnL', 
            'in', 'inH', 'inL', 'on', 'onH', 'onL', 'oRn', 'oRnH', 'oRnL',
            'un', 'unH', 'unL'
        ],
        help="Set of vowels and nasal vowels to extract",
    )
    parser.add_argument(
        "--nasal_set",
        type=str,
        nargs='+',
        default=['n'],
        help="Set of nasals to extract",
    )
    parser.add_argument(
        "--pooling_window",
        type=int,
        default=10,
        help="Window size for mean pooling of features",
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
            syllable_times = extract_syllables_with_tones(phones_tier, vowel_set)
            alignments[filename.split('.')[0]] = syllable_times
    return alignments


def extract_syllables_with_tones(phones_tier, vowel_set):
    syllables = []
    current_syllable = []
    current_vowels = []  # New list to store vowels
    start_time = None
    tones = []

    def get_tone(vowel):
        if vowel.endswith('H'):
            return 'H'
        elif vowel.endswith('L'):
            return 'L'
        else:
            return 'M'
        
    for interval in phones_tier.entryList:
        phone = interval.label
        # print("phone:", phone)

        if phone == "":
            continue  # Skip empty labels

        if start_time is None:
            start_time = interval.start  # Mark the start time of the syllable

        if phone in vowel_set:  # If the phone is a vowel, end of syllable
            current_syllable.append(phone)
            current_vowels.append(phone)  # Add vowel to current vowels list
            tones.append(get_tone(phone))
            # Combine the phones into a single string to represent the syllable
            combined_syllable = ''.join(current_syllable)
            syllables.append({
                'phones': [combined_syllable],  # Store as a list with one element
                'tone': tones[-1],
                'start_time': start_time,
                'end_time': interval.end,
                'vowels': current_vowels  # Include the vowels in the syllable
            })
            current_syllable = []
            current_vowels = []  # Reset vowels for the next syllable
            start_time = None  # Reset start time for the next syllable
        else:
            current_syllable.append(phone)  # Add consonant to the current syllable

    return syllables

    # for interval in phones_tier.entryList:
    #     phone = interval.label
    #     # print("phone:", phone)

    #     if phone == "":
    #         continue  # Skip empty labels

    #     if start_time is None:
    #         start_time = interval.start  # Mark the start time of the syllable

    #     if phone in vowel_set:  # If the phone is a vowel, end of syllable
    #         current_syllable.append(phone)
    #         tones.append(get_tone(phone))
    #         # Combine the phones into a single string to represent the syllable
    #         combined_syllable = ''.join(current_syllable)
    #         # syllables.append({
    #         #     'phones': current_syllable,
    #         #     'tone': tones[-1],
    #         #     'start_time': start_time,
    #         #     'end_time': interval.end
    #         # })
    #         syllables.append({
    #             'phones': [combined_syllable],  # Store as a list with one element
    #             'tone': tones[-1],
    #             'start_time': start_time,
    #             'end_time': interval.end
    #         })
    #         current_syllable = []
    #         start_time = None  # Reset start time for the next syllable
    #     else:
    #         current_syllable.append(phone)  # Add consonant to the current syllable

    # return syllables


def mean_pooling(features, pooling_window):
    """
    Applies mean pooling over the time dimension of the features.

    Args:
        features (numpy.ndarray): A 2D array where the first dimension corresponds to time and 
                                  the second dimension corresponds to the feature dimensions.
        pooling_window (int): The size of the window over which to apply mean pooling.

    Returns:
        numpy.ndarray: A 2D array where the time dimension is reduced by mean pooling.
    """
    # Ensure the input features are at least 2D
    if len(features.shape) < 2:
        raise ValueError("Features should be a 2D array with time as the first dimension.")

    # Perform mean pooling over the time dimension
    pooled_features = []
    for i in range(0, features.shape[0], pooling_window):
        window = features[i:i + pooling_window]
        pooled_features.append(np.mean(window, axis=0))
    
    return np.array(pooled_features)


# def get_feature_iterator(
#     feature_type, checkpoint_path, layer, manifest_path, sample_pct, alignments, pooling_window #, min_length=4000
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
#                     feats = reader.get_feats(waveform).cpu().numpy()
#                     for segment in segments:
#                         start_frame = int(segment['start_time'] * sample_rate / 1000)  # Convert to milliseconds
#                         end_frame = int(segment['end_time'] * sample_rate / 1000)
#                         segment_feats = feats[start_frame:end_frame]
#                         # if segment_feats.size >= min_length:
#                         pooled_feats = mean_pooling(segment_feats, pooling_window)
#                         yield pooled_feats, segment['phones'], segment['tone']

#     return iterate, num_files

def get_feature_iterator(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, alignments, pooling_window
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
                results = []
                if segments:
                    waveform, sample_rate = sf.read(file_path)
                    feats = reader.get_feats(waveform).cpu().numpy()
                    for segment in segments:
                        start_frame = int(segment['start_time'] * sample_rate / 1000)  # Convert to milliseconds
                        end_frame = int(segment['end_time'] * sample_rate / 1000)
                        segment_feats = feats[start_frame:end_frame]
                        if segment_feats.size > 0:
                            pooled_feats = mean_pooling(segment_feats, pooling_window)
                            results.append((pooled_feats, segment['phones'], segment['tone'], segment))
                        else:
                            results.append(([], segment['phones'], segment['tone'], segment))
                yield file_path, results

    return iterate, num_files


# def get_features_and_labels(
#     feature_type, checkpoint_path, layer, manifest_path, sample_pct, alignments, pooling_window #, min_length=4000
# ):
#     generator, num_files = get_feature_iterator(
#         feature_type=feature_type,
#         checkpoint_path=checkpoint_path,
#         layer=layer,
#         manifest_path=manifest_path,
#         sample_pct=sample_pct,
#         alignments=alignments,
#         pooling_window=pooling_window,
#         # min_length=min_length
#     )
#     iterator = generator()

#     features_list = []
#     labels_list = []
#     tones_list = []
#     syllables_list = []
#     # for features, syllable, tone in tqdm.tqdm(iterator, total=num_files):
#     #     if features.size > 0:
#     #         features_list.append(features)
#     #         labels_list.append(syllable)
#     #         tones_list.append(tone)
#     #         syllables_list.append((syllable, tone))
#     #     else:
#     #         print("No features extracted for this segment.")
#     for file_path, segments in tqdm.tqdm(iterator, total=num_files):
#         for features, syllable, tone, segment in segments:
#             if features.size > 0:
#                 features_list.append(features)
#                 labels_list.append(syllable)
#                 tones_list.append(tone)
#                 syllables_list.append((syllable, tone))
#             else:
#                 # Determine why no features were extracted
#                 if features.size == 0:
#                     reason = "Feature array is empty"
#                 else:
#                     reason = "Unknown reason"
                
#                 # Log the file ID, segment details, and reason for failure
#                 print(f"No features extracted for this segment in file {file_path}: {segment} - Reason: {reason}")

#     # Explicit clean up
#     del iterator
#     del generator
#     gc.collect()
#     torch.cuda.empty_cache()

#     return features_list, labels_list, tones_list, syllables_list  # Return as lists

def get_features_and_labels(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, alignments, pooling_window
):
    generator, num_files = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        alignments=alignments,
        pooling_window=pooling_window,
    )
    iterator = generator()

    features_list = []
    labels_list = []
    tones_list = []
    syllables_list = []
    vowels_list = []  # New list to store vowels

    total_segments = 0
    extracted_segments = 0

    for file_path, segments in tqdm.tqdm(iterator, total=num_files):
        for features, syllable, tone, segment in segments:
            total_segments += 1  # Increment total segments counter

            if len(features) > 0:  # Check if features were extracted
                extracted_segments += 1  # Increment extracted segments counter
                features_list.append(features)
                labels_list.append(syllable)
                tones_list.append(tone)
                syllables_list.append((syllable, tone))
                vowels_list.append(segment['vowels'])  # Append vowels to vowels list
            else:
                reason = "Feature array is empty"
                print(f"No features extracted for this segment in file {file_path}: {segment} - Reason: {reason}")

    # Explicit clean up
    del iterator
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    # Print summary of segment extraction
    print(f"Total segments processed: {total_segments}")
    print(f"Segments with features extracted: {extracted_segments}")
    print(f"Segments without features extracted: {total_segments - extracted_segments}")

    return features_list, labels_list, tones_list, syllables_list, vowels_list  # Return vowels_list





def save_to_csv(data, filename, header=None):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(header)
        writer.writerows(data)

# def get_and_dump_features_and_labels(
#     feature_type,
#     checkpoint_path,
#     layer,
#     manifest_path,
#     sample_pct,
#     out_features_path,
#     out_labels_path,
#     out_syllables_path,
#     alignments,
#     pooling_window,
#     # min_length=4000
# ):
#     # Feature, label, and syllable extraction
#     features, labels, tones, syllables = get_features_and_labels(
#         feature_type=feature_type,
#         checkpoint_path=checkpoint_path,
#         layer=layer,
#         manifest_path=manifest_path,
#         sample_pct=sample_pct,
#         alignments=alignments,
#         pooling_window=pooling_window,
#         # min_length=min_length
#     )

#     # Save features, labels, tones, and syllables
#     out_dir_path = os.path.dirname(out_features_path)
#     os.makedirs(out_dir_path, exist_ok=True)
#     shutil.copyfile(
#         manifest_path,
#         os.path.join(out_dir_path, os.path.basename(manifest_path)),
#     )
#     np.save(out_features_path, np.array(features, dtype=object))  # Save features
#     np.save(out_labels_path, np.array(labels, dtype=object))  # Save labels
#     np.save(out_syllables_path, np.array(syllables, dtype=object))  # Save syllables
#     np.save(out_labels_path.replace("labels", "tones"), np.array(tones, dtype=object))  # Save tones

#     return features, labels, syllables, tones

def get_and_dump_features_and_labels(
    feature_type,
    checkpoint_path,
    layer,
    manifest_path,
    sample_pct,
    out_features_path,
    out_labels_path,
    out_syllables_path,
    alignments,
    pooling_window,
    # min_length=4000
):
    # Feature, label, and syllable extraction
    features, labels, tones, syllables, vowels = get_features_and_labels(  # Updated function call
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        alignments=alignments,
        pooling_window=pooling_window,
        # min_length=min_length
    )

    # Save features, labels, tones, syllables, and vowels as NumPy arrays
    out_dir_path = os.path.dirname(out_features_path)
    os.makedirs(out_dir_path, exist_ok=True)
    shutil.copyfile(
        manifest_path,
        os.path.join(out_dir_path, os.path.basename(manifest_path)),
    )
    np.save(out_features_path, np.array(features, dtype=object))  # Save features
    np.save(out_labels_path, np.array(labels, dtype=object))  # Save labels
    np.save(out_syllables_path, np.array(syllables, dtype=object))  # Save syllables
    np.save(out_labels_path.replace("labels", "tones"), np.array(tones, dtype=object))  # Save tones
    np.save(out_labels_path.replace("labels", "vowels"), np.array(vowels, dtype=object))  # Save vowels

    # Save features, labels, tones, syllables, and vowels as CSV files
    features_csv_path = out_features_path.replace(".npy", ".csv")
    labels_csv_path = out_labels_path.replace(".npy", ".csv")
    tones_csv_path = out_labels_path.replace("labels.npy", "tones.csv")
    syllables_csv_path = out_syllables_path.replace(".npy", ".csv")
    vowels_csv_path = out_labels_path.replace("labels.npy", "vowels.csv")  # New CSV file for vowels

    save_to_csv(features, features_csv_path, header=["features"])
    save_to_csv(labels, labels_csv_path, header=["Syllable"])
    save_to_csv(tones, tones_csv_path, header=["Tone"])
    save_to_csv(syllables, syllables_csv_path, header=["Syllable", "Tone"])
    save_to_csv(vowels, vowels_csv_path, header=["Vowels"])  # Save vowels to CSV

    return features, labels, syllables, tones, vowels  # Return vowels

if __name__ == "__main__":
    """
    Example command:
    python script.py \
        --feature_type hubert \
        --manifest_path /path/to/manifest.tsv \
        --out_features_path /path/to/output_features.npy \
        --out_labels_path /path/to/output_labels.npy \
        --out_syllables_path /path/to/output_syllables.npy \
        --checkpoint_path /path/to/checkpoint \
        --layer -1 \
        --sample_pct 0.1 \
        --textgrid_dir /path/to/textgrids \
        --vowel_set a e i o u \
        --nasal_set n \
        --pooling_window 10
    """
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)

    logger.info(f"Extracting {args.feature_type} acoustic features...")

    # Load MFA alignments
    alignments = load_mfa_alignments(args.textgrid_dir, set(args.vowel_set))
    print("Alignments:", alignments)  # Debugging statement

    features, labels, syllables, tones, vowels = get_and_dump_features_and_labels(  # Updated function call
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint_path,
        layer=args.layer,
        manifest_path=args.manifest_path,
        sample_pct=args.sample_pct,
        out_features_path=args.out_features_path,
        out_labels_path=args.out_labels_path,
        out_syllables_path=args.out_syllables_path,
        alignments=alignments,
        pooling_window=args.pooling_window,
    )
    logger.info(f"Saved extracted features at {args.out_features_path}")
    logger.info(f"Saved extracted labels at {args.out_labels_path}")
    logger.info(f"Saved extracted syllables at {args.out_syllables_path}")
    logger.info(f"Saved extracted tones at {args.out_labels_path.replace('labels', 'tones')}")
    logger.info(f"Saved extracted vowels at {args.out_labels_path.replace('labels', 'vowels')}")  # New log for vowels
    logger.info(f"Features saved: {len(features)} items")
    logger.info(f"Labels saved: {len(labels)} items")
    logger.info(f"Syllables saved: {len(syllables)} items")
    logger.info(f"Tones saved: {len(tones)} items")
    logger.info(f"Vowels saved: {len(vowels)} items")  # New log for vowels