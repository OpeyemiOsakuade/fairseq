#!/bin/bash
#SBATCH -o /home/%u/slogs/_hu_manvowels1000_l9_yor_%A.out
#SBATCH -e /home/%u/slogs/_hu_manvowels1000_yor_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:2 # use 1 GPU
#SBATCH --partition=ILCC-Standard
#SBATCH -t 24:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64000


# python feature_extraction.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/yoruba/manifest/manifest/test.tsv \
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_vectors/test_yor_vectors.npy \
#   --checkpoint_path /disk/ostrom/homedirs/s2324992/data/pretrained_models/hubert_base_ls960.pt \
#   --layer 9 \
#   --sample_pct 0.1 
#   vowel_set = {'a','aH','aL' 'e','eH','eL','eR','eRH','eRL','i','iH','iL','o','oH','oL','oR','oRH','oRL','u','uH','uL'}
#   # --flatten

# python feature_extraction.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/yoruba/manifest/manifest/train.tsv \
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_vectors/no_tones_alltrain_yor_vectors.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_vectors/no_tones_alltrain_yor_labels.npy \
#   --checkpoint_path /disk/ostrom/homedirs/s2324992/data/pretrained_models/hubert_base_ls960.pt \
#   --layer 9 \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/aligned \
#   --vowel_set a aH aL e eH eL eR eRH eRL i iH iL o oH oL oR oRH oRL u uH uL 


# python feature_vowel_classifier.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_vectors/no_tones_alltrain_yor_vectors.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_vectors/no_tones_alltrain_yor_labels.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/model_vectors_it.pkl

# nvidia-smi
# sacct -j 1856674 --format=JobID,JobName,MaxRSS,Elapsed,State,ExitCode

# python feature_extraction_tones.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/yoruba/manifest/manifest/train.tsv \
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/tones_alltrain_yor_vectors_160.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/tones_alltrain_yor_labels_160.npy \
#   --checkpoint_path /disk/ostrom/homedirs/s2324992/data/pretrained_models/hubert_base_ls960.pt \
#   --layer 9 \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/aligned \
#   --vowel_set a aH aL e eH eL eR eRH eRL i iH iL o oH oL oR oRH oRL u uH uL \
#   --downsample_factor 160

# python feature_vowel_classifier.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/tones_alltrain_yor_vectors_160.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/tones_alltrain_yor_labels_160.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/tone_model_frames_1000it_160.pkl

# python save_mean_pooled_features.py \
#   --features_path /home/s2324992/tonal_data/tone_quantization/yoruba/mean_pooled_features.npy \
#   --manifest_path /home/s2324992/tonal_data/tone_quantization/yoruba/all.tsv \
#   --out_quantized_file_path /home/s2324992/tonal_data/tone_quantization/yoruba/quantized.txt


# for file in  /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/yoruba/dev/*.wav; do
#     ffmpeg -i "$file" -ar 16000 "temp.wav"
#     mv "temp.wav" "$file"
# done


python feature_vowel_classifier_reps_lstm.py \
  --data_path /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/models/mandarinHuBERT_9/for_vowels_1000/updated_0.1_manhu_manl9c1000_vowels_reps.json \
  --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/discrete_features/models/ManHuno_tone_1000_lstm_model.pth