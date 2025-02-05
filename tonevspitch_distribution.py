import argparse
import os
import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textgrid


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract pitch values from vowels in TextGrid files."
    )
    parser.add_argument(
        "--vowels", 
        type=str, 
        nargs='+',
        default=['a', 'e', 'i', 'o', 'u'], 
        help="Vowels to check for in the intervals",
    )
    parser.add_argument(
        "--textgrid_folder", 
        type=str, 
        required=True, 
        help="Folder containing the TextGrid files.",
    )
    parser.add_argument(
        "--audio_folder", 
        type=str, 
        required=True, 
        help="Folder containing the corresponding audio files.",
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        required=True, 
        help="Folder to save the output plots.",
    )
    return parser

def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def extract_pitch(y, sr, start_time, end_time):
    logger.info(f"Extracting pitch from {start_time:.2f} to {end_time:.2f} seconds.")
    y_segment = y[int(start_time * sr):int(end_time * sr)]
    pitches, voiced_flags, _ = librosa.pyin(y_segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_values = pitches[~np.isnan(pitches)]
    logger.info(f"Extracted {len(pitch_values)} pitch values.")
    return pitch_values

def process_files(textgrid_folder, audio_folder, vowels):
    logger.info("Starting to process files...")
    vowel_pitches = {vowel: {'L': [], 'M': [], 'H': []} for vowel in vowels}

    for file_name in os.listdir(textgrid_folder):
        if file_name.endswith('.TextGrid'):
            logger.info(f"Processing file: {file_name}")
            textgrid_path = os.path.join(textgrid_folder, file_name)
            audio_path = os.path.join(audio_folder, file_name.replace('.TextGrid', '.wav'))

            if os.path.exists(audio_path):
                tg = textgrid.TextGrid.fromFile(textgrid_path)
                y, sr = librosa.load(audio_path, sr=None)
                logger.info(f"Loaded audio file: {audio_path} with sample rate: {sr}")

                for interval in tg.getFirst("phones").intervals:
                    text = interval.mark
                    start_time = interval.minTime
                    end_time = interval.maxTime

                    # Match the vowel and assign to the correct tone
                    for vowel in vowels:
                        if vowel in text:
                            logger.info(f"Found vowel '{vowel}' in interval '{text}' from {start_time:.2f} to {end_time:.2f} seconds.")
                            pitch_values = extract_pitch(y, sr, start_time, end_time)

                            if text.endswith('L'):
                                vowel_pitches[vowel]['L'].extend(pitch_values)
                                logger.info(f"Assigned {len(pitch_values)} pitch values to '{vowel}' with 'L' tone.")
                            elif text.endswith('M'):
                                vowel_pitches[vowel]['M'].extend(pitch_values)
                                logger.info(f"Assigned {len(pitch_values)} pitch values to '{vowel}' with 'M' tone.")
                            elif text.endswith('H'):
                                vowel_pitches[vowel]['H'].extend(pitch_values)
                                logger.info(f"Assigned {len(pitch_values)} pitch values to '{vowel}' with 'H' tone.")
            else:
                logger.warning(f"Audio file not found for TextGrid: {file_name}")
    
    logger.info("Finished processing files.")
    return vowel_pitches

def plot_and_save_pitch_distribution(vowel_pitches, output_folder):
    logger.info("Starting to plot pitch distributions...")
    for vowel, pitches_dict in vowel_pitches.items():
        plt.figure(figsize=(10, 6))
        for tone, pitches in pitches_dict.items():
            if pitches:
                sns.histplot(pitches, kde=False, label=tone, bins=50)
                logger.info(f"Plotting pitch distribution for vowel '{vowel}' with tone '{tone}'.")
        plt.title(f'Pitch Distribution for Vowel "{vowel}" in Mandarin Speech')
        plt.xlabel('Fundamental Frequency (Hz)')
        plt.ylabel('Number of Occurrences')
        plt.legend(title='Tone')
        output_path = os.path.join(output_folder, f'pitch_distribution_{vowel}.png')
        plt.savefig(output_path)  # Save the plot as a file
        plt.close()  # Close the plot to free memory
        logger.info(f"Plot for vowel '{vowel}' saved to {output_path}")

def main():
    # Parse command line arguments
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Extract and process vowel pitches
    vowels = list(args.vowels)
    vowel_pitches = process_files(args.textgrid_folder, args.audio_folder, vowels)
    
    # Plot and save pitch distributions
    plot_and_save_pitch_distribution(vowel_pitches, args.output_folder)
    logger.info("All plots have been saved.")

if __name__ == "__main__":
    parser = get_parser()
    logger = get_logger()
    
    main()
