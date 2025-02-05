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
import torchaudio
from praatio import tgio
import csv
from transformers import Wav2Vec2FeatureExtractor, HubertModel, Wav2Vec2Model

def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute and dump acoustic features for syllable frames."
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["hubert", "w2v2"],
        default=None,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Hugging Face model name or path to the model checkpoint.",
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
        "--pooling_type",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Type of pooling to apply: 'mean' or 'max'.",
    )
    return parser


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


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
    current_vowels = []
    current_plain_vowels = []
    start_time = None
    tones = []

    def get_tone(vowel):
        if vowel.endswith('H'):
            return 'H'
        elif vowel.endswith('L'):
            return 'L'
        else:
            return 'M'
        
    def strip_tone(vowel):
        if vowel.endswith('H') or vowel.endswith('L'):
            return vowel[:-1]
        return vowel
        
    for interval in phones_tier.entryList:
        phone = interval.label

        if phone == "":
            continue

        if start_time is None:
            start_time = interval.start

        if phone in vowel_set:
            current_syllable.append(phone)
            current_vowels.append(phone)
            current_plain_vowels.append(strip_tone(phone))
            tones.append(get_tone(phone))
            combined_syllable = ''.join(current_syllable)
            syllables.append({
                'phones': [combined_syllable],
                'tone': tones[-1],
                'start_time': start_time,
                'end_time': interval.end,
                'vowels': current_vowels,
                'plain_vowels': current_plain_vowels
            })
            current_syllable = []
            current_vowels = []
            current_plain_vowels = []
            start_time = None
        else:
            current_syllable.append(phone)

    return syllables


def apply_pooling(features, pooling_type):
    """
    Applies the specified pooling type over the time dimension of the features.

    Args:
        features (torch.Tensor): A 2D tensor where the first dimension corresponds to time and 
                                  the second dimension corresponds to the feature dimensions.
        pooling_type (str): Type of pooling to apply: 'mean' or 'max'.

    Returns:
        torch.Tensor: A 1D tensor where the time dimension is pooled according to the specified pooling type.
    """
    if pooling_type == "mean":
        pooled = torch.mean(features, dim=0)  # Mean pooling over time dimension
    elif pooling_type == "max":
        pooled, _ = torch.max(features, dim=0)  # Max pooling over time dimension
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
    
    return pooled

def get_feature_iterator(
    feature_type, model_name, layer, manifest_path, sample_pct, alignments, pooling_type
):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    if feature_type == "hubert":
        model = HubertModel.from_pretrained(model_name)
    elif feature_type == "w2v2":
        model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    # Define the minimum input length based on the model's first convolutional layer
    # min_input_size = model.config.conv_kernel[0]
    # print(f"The min input size: {model.config.conv_kernel[0]} ")
    min_input_size = 4000

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

        def iterate():
            for file_path in file_path_list:
                file_id = os.path.splitext(os.path.basename(file_path))[0]
                segments = alignments.get(file_id, [])
                results = []
                if segments:
                    waveform, sample_rate = sf.read(file_path)
                    for segment in segments:
                        start_sample = int(segment['start_time'] * sample_rate)
                        end_sample = int(segment['end_time'] * sample_rate)
                        waveform_segment = waveform[start_sample:end_sample]

                        # Ensure the waveform segment meets the minimum input size
                        if sample_rate != 16000:
                            waveform_segment = torchaudio.transforms.Resample(sample_rate, 16000)(waveform_segment)

                        # Log the segment information
                        print(f"Segment start: {segment['start_time']}, end: {segment['end_time']}, length: {waveform_segment.shape[0]} samples")
                        
                        # Check if the final input length is enough
                        if waveform_segment.shape[0] < min_input_size:
                            print(f"Segment start: {segment['start_time']}, end: {segment['end_time']}, length: {waveform_segment.shape[0]} samples")
                            print(f"Skipping segment in file {file_path} due to insufficient length after resampling: {waveform_segment.shape[0]} samples")
                            continue  # Skip this segment if it's too short

                        # Feature extraction
                        inputs = feature_extractor(waveform_segment, sampling_rate=16000, return_tensors="pt", padding=True)
                        input_values = inputs.input_values

                        # # Ensure the padded input meets the convolutional layer requirements
                        # if input_values.shape[-1] < min_input_size:
                        #     print(f"Skipping segment after padding in file {file_path} due to insufficient length: {input_values.shape[-1]} samples")
                        #     continue

                        # Ensure the padded input meets the convolutional layer requirements
                        if input_values.shape[-1] < min_input_size:
                            padding_size = min_input_size - input_values.shape[-1]
                            input_values = torch.nn.functional.pad(input_values, (0, padding_size))
                            print(f"Padded segment in file {file_path} to meet the minimum input size requirement.")

                        with torch.no_grad():
                            outputs = model(input_values, output_hidden_states=True)

                        # Extract hidden states for each layer
                        layer_features = outputs.hidden_states[layer].squeeze(0)  # Remove batch dimension if necessary

                        # Apply the selected pooling type over the time dimension
                        pooled_feats = apply_pooling(layer_features, pooling_type)

                        results.append((pooled_feats.cpu().numpy(), segment['phones'], segment['tone'], segment))
                yield file_path, results

    return iterate, num_files



def get_features_and_labels(
    feature_type, model_name, layer, manifest_path, sample_pct, alignments, pooling_type
):
    generator, num_files = get_feature_iterator(
        feature_type=feature_type,
        model_name=model_name,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        alignments=alignments,
        pooling_type=pooling_type,
    )
    iterator = generator()

    features_list = []
    labels_list = []
    tones_list = []
    syllables_list = []
    vowels_list = []
    plain_vowels_list = []

    total_segments = 0
    extracted_segments = 0

    for file_path, segments in tqdm.tqdm(iterator, total=num_files):
        for features, syllable, tone, segment in segments:
            total_segments += 1

            if features.size > 0:
                extracted_segments += 1
                features_list.append(features)
                labels_list.append(syllable)
                tones_list.append(tone)
                syllables_list.append((syllable, tone))
                vowels_list.append(segment['vowels'])
                plain_vowels_list.append(segment['plain_vowels'])
            else:
                reason = "Feature array is empty"
                print(f"No features extracted for this segment in file {file_path}: {segment} - Reason: {reason}")

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Total segments processed: {total_segments}")
    print(f"Segments with features extracted: {extracted_segments}")
    print(f"Segments without features extracted: {total_segments - extracted_segments}")

    return features_list, labels_list, tones_list, syllables_list, vowels_list, plain_vowels_list


def save_to_csv(data, filename, header=None):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(header)
        writer.writerows(data)


def get_and_dump_features_and_labels(
    feature_type,
    model_name,
    layer,
    manifest_path,
    sample_pct,
    out_features_path,
    out_labels_path,
    out_syllables_path,
    alignments,
    pooling_type,
):
    features, labels, tones, syllables, vowels, plain_vowels = get_features_and_labels(
        feature_type=feature_type,
        model_name=model_name,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        alignments=alignments,
        pooling_type=pooling_type,
    )

    out_dir_path = os.path.dirname(out_features_path)
    os.makedirs(out_dir_path, exist_ok=True)
    shutil.copyfile(
        manifest_path,
        os.path.join(out_dir_path, os.path.basename(manifest_path)),
    )
    np.save(out_features_path, np.array(features, dtype=object))
    np.save(out_labels_path, np.array(labels, dtype=object))
    np.save(out_syllables_path, np.array(syllables, dtype=object))
    np.save(out_labels_path.replace("labels", "tones"), np.array(tones, dtype=object))
    np.save(out_labels_path.replace("labels", "vowels"), np.array(vowels, dtype=object))
    np.save(out_labels_path.replace("labels", "plain_vowels"), np.array(plain_vowels, dtype=object))

    features_csv_path = out_features_path.replace(".npy", ".csv")
    labels_csv_path = out_labels_path.replace(".npy", ".csv")
    tones_csv_path = out_labels_path.replace("labels.npy", "tones.csv")
    syllables_csv_path = out_syllables_path.replace(".npy", ".csv")
    vowels_csv_path = out_labels_path.replace("labels.npy", "vowels.csv")
    plain_vowels_csv_path = out_labels_path.replace("labels.npy", "plain_vowels.csv")

    save_to_csv(features, features_csv_path, header=["features"])
    save_to_csv(labels, labels_csv_path, header=["Syllable"])
    save_to_csv(tones, tones_csv_path, header=["Tone"])
    save_to_csv(syllables, syllables_csv_path, header=["Syllable", "Tone"])
    save_to_csv(vowels, vowels_csv_path, header=["Vowels"])
    save_to_csv(plain_vowels, plain_vowels_csv_path, header=["Plain Vowels"])

    return features, labels, syllables, tones, vowels, plain_vowels


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)

    logger.info(f"Extracting {args.feature_type} acoustic features using {args.pooling_type} pooling...")

    alignments = load_mfa_alignments(args.textgrid_dir, set(args.vowel_set))
    # print("Alignments:", alignments)

    features, labels, syllables, tones, vowels, plain_vowels = get_and_dump_features_and_labels(
        feature_type=args.feature_type,
        model_name=args.model_name,
        layer=args.layer,
        manifest_path=args.manifest_path,
        sample_pct=args.sample_pct,
        out_features_path=args.out_features_path,
        out_labels_path=args.out_labels_path,
        out_syllables_path=args.out_syllables_path,
        alignments=alignments,
        pooling_type=args.pooling_type,
    )

    logger.info(f"Saved extracted features at {args.out_features_path}")
    logger.info(f"Saved extracted labels at {args.out_labels_path}")
    logger.info(f"Saved extracted syllables at {args.out_syllables_path}")
    logger.info(f"Saved extracted tones at {args.out_labels_path.replace('labels', 'tones')}")
    logger.info(f"Saved extracted vowels at {args.out_labels_path.replace('labels', 'vowels')}")
    logger.info(f"Saved extracted plain vowels at {args.out_labels_path.replace('labels', 'plain_vowels')}")
    logger.info(f"Features saved: {len(features)} items")
    logger.info(f"Labels saved: {len(labels)} items")
    logger.info(f"Syllables saved: {len(syllables)} items")
    logger.info(f"Tones saved: {len(tones)} items")
    logger.info(f"Vowels saved: {len(vowels)} items")
    logger.info(f"Plain Vowels saved: {len(plain_vowels)} items")
