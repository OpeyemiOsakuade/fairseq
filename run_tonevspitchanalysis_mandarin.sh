#!/bin/bash
#SBATCH -o /home/%u/slogs/tonevspitch_yor_%A.out
#SBATCH -e /home/%u/slogs/tonevspitch_yor_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:2 # use 1 GPU
#SBATCH --partition=ILCC-Standard
#SBATCH -t 64:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=96000

# python feature_syllable_ssl_extraction_new.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/yoruba/manifest/manifest/train_dev.tsv \
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF/mean_syllables_plain_vowels/syllables_train_dev_yor_frames.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF/mean_syllables_plain_vowels/syllables_train_dev_yor_labels.npy \
#   --out_syllables_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF/mean_syllables_plain_vowels/syllables_train_dev_yor_syllables.npy \
#   --model_name facebook/hubert-base-ls960 \
#   --pooling_type mean \
#   --layer 9 \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/aligned \
#   --vowel_set a aH aL e eH eL eR eRH eRL i iH iL o oH oL oR oRH oRL u uH uL an anH anL en enH enL eRn eRnH eRnL in inH inL on onH onL oRn oRnH oRnL un unH unL \





# Adjust the vowels, TextGrid folder, audio folder, and output folder as needed
# VOWELS a e eR i o oR u
# TEXTGRID_FOLDER="/"
# AUDIO_FOLDER="/disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/audio_files"
# OUTPUT_FOLDER="/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_toneVSpitch_distribution"

python tonevspitch_distribution.py \
  --vowels a e i o u \
  --textgrid_folder /disk/ostrom/homedirs/s2324992/data_aishell/aligned \
  --audio_folder /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/all_audios \
  --output_folder /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_toneVSpitch_distribution











