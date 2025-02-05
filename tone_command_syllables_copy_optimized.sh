#!/bin/bash
#SBATCH -o /home/%u/slogs/w2v2_mandarin_%A.out
#SBATCH -e /home/%u/slogs/w2v2_mandarin_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1 # use 1 GPU
#SBATCH --partition=ILCC-Standard
#SBATCH -t 64:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=24000

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

# python feature_vowel_classifier.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF/mean_syllables_plain_vowels/syllables_train_dev_yor_frames.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF/mean_syllables_plain_vowels/syllables_train_dev_yor_tones.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF/mean_syllables_plain_vowels/models/syllables_model_frames_tonesit_100.pkl

# python feature_vowel_classifier_reps.py \
#   --data_path /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/models/hubert_9/for_vowels/0.1_hu_manl9c200_phoneme_reps.json \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF/reps/models/model_reps_vowels_tonesit_100.pkl


# python feature_ssl_extraction_mandarin_optimized.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/manifest/train.tsv\
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vowels/train_man_frames.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vowels/train_man_labels.npy \
#   --model_name facebook/hubert-base-ls960 \
#   --pooling_type mean \
#   --layer 9 \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/data_aishell/aligned \
#   --vowel_set a1 a2 a3 a4 a5 e1 e2 e3 e4 e5 i1 i2 i3 i4 i5 o1 o2 o3 o4 o5 u1 u2 u3 u4 u5 



# python feature_vowel_classifier.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vowels/train_man_frames.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vowels/train_man_tones.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vowels/tone_model_vectors_it.pkl



# python feature_vowel_classifier_reps_lstm.py \
#   --data_path /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/models/hubert_9/for_vowels/0.1_yorHUl9c200_vowels_reps.json \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/vowel_with_tone_lstm_model.pth

# python feature_vowel_classifier_reps.py \
#   --data_path /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/models/hubert_9/for_vowels/0.1_yorHUl9c200_vowels_reps.json \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/vowel_with_tone_geo_mean_model.pkl


# python feature_vowel_classifier_vectors_MLP.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vectors/train_man_frames.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vectors/train_man_vowels_fromcsv.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vectors/MLP_vowels_tone_model.pkl


# python feature_ssl_extraction_mandarin_optimized.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/manifest/train.tsv\
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors_1000/train_man_features.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors_1000/train_man_labels.npy \
#   --model_name facebook/hubert-base-ls960 \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/data_aishell/aligned \
#   --vowel_set a1 a2 a3 a4 a5 e1 e2 e3 e4 e5 i1 i2 i3 i4 i5 o1 o2 o3 o4 o5 u1 u2 u3 u4 u5 

# python feature_vowel_classifier_vectors_lstm.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/train_man_features_layer_9.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/train_man_tones.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/models/lstm_tone_model_l9.pkl


# python feature_ssl_extraction_mandarinhubertfromscratch.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/manifest/train.tsv \
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/mandarinhu_train_man_features.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/mandarinhu_train_man_labels.npy \
#   --model_name TencentGameMate/chinese-hubert-base \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/data_aishell/aligned \
#   --vowel_set a1 a2 a3 a4 a5 e1 e2 e3 e4 e5 i1 i2 i3 i4 i5 o1 o2 o3 o4 o5 u1 u2 u3 u4 u5 

# python feature_vowel_classifier_vectors_lstm.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/all_vectors/xlsr_syllables_train_dev_yor_features_layer_9.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/all_vectors/syllables_train_dev_yor_vowels_with_tones_correct_1.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/all_vectors/models/xlsr_yor_lstm_vowels_with_tones_model_l9.pkl

# python feature_vowel_classifier_reps_lstm.py \
#   --data_path /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/models/xlsr_9/for_phones/correct_0.1_yorxlsrl9c200_vowels_reps.json \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/discretized_sequence/models/vowels_no_tone_lstm_model.pth

# python feature_vowel_classifier_reps_lstm.py \
#   --data_path /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/models/mandarinHuBERT_9/for_phones_1000/updated0.1_manhu_manl9c1000_phone_tones_reps.json \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/discrete_features/models/vowel_with_tone_1000_lstm_model.pth

# python feature_vowel_classifier.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vowels/train_man_frames.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vowels/train_man_tones.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vowels/tone_model_vectors_it.pkl


# python feature_ssl_extraction_mandarinhubertfromscratch_pooling.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/manifest/train.tsv \
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vectors/mandarinhu_train_man_features.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/mean_vectors/mandarinhu_train_man_labels.npy \
#   --model_name TencentGameMate/chinese-hubert-base \
#   --pooling_type mean \
#   --layer 9 \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/data_aishell/aligned \
#   --vowel_set a1 a2 a3 a4 a5 e1 e2 e3 e4 e5 i1 i2 i3 i4 i5 o1 o2 o3 o4 o5 u1 u2 u3 u4 u5 

# python feature_vowel_classifier_LR.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/mean_vectors/syllables_train_dev_yor_frames.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/mean_vectors/syllables_train_dev_yor_plain_vowels.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/mean_vectors/models/yor_xlsr_model_vowels_no_tone.pkl





# python feature_ssl_extraction_mandarin_optimized.py \
#   --feature_type w2v2 \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/manifest/train.tsv\
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/w2v2_all_vectors/train_man_features.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/w2v2_all_vectors/train_man_labels.npy \
#   --model_name facebook/wav2vec2-base-960h \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/data_aishell/aligned \
#   --vowel_set a1 a2 a3 a4 a5 e1 e2 e3 e4 e5 i1 i2 i3 i4 i5 o1 o2 o3 o4 o5 u1 u2 u3 u4 u5 


python feature_vowel_classifier_vectors_lstm_improved.py \
  --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/mandarinhu_train_man_features_layer_9.npy \
  --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/mandarinhu_train_man_vowels.npy \
  --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/models/mandarinhu_lstm_vowel_tones_model_l9_improved.pkl





















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
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/no_tone_model_vectors_it.pkl

# nvidia-smi
# sacct -j 1856674 --format=JobID,JobName,MaxRSS,Elapsed,State,ExitCode

# python feature_syllable_ssl_extraction_copy.py \
#   --feature_type hubert \
#   --manifest_path /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/yoruba/manifest/manifest/train_dev.tsv \
#   --out_features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/syllables_plain_vowels/syllables_train_dev_yor_frames.npy \
#   --out_labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/syllables_plain_vowels/syllables_train_dev_yor_labels.npy \
#   --out_syllables_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/syllables_plain_vowels/syllables_train_dev_yor_syllables.npy \
#   --checkpoint_path /disk/ostrom/homedirs/s2324992/data/pretrained_models/hubert_base_ls960.pt \
#   --layer 9 \
#   --sample_pct 1 \
#   --textgrid_dir /disk/ostrom/homedirs/s2324992/tonal_data/data/yoruba_wav/aligned \
#   --vowel_set a aH aL e eH eL eR eRH eRL i iH iL o oH oL oR oRH oRL u uH uL an anH anL en enH enL eRn eRnH eRnL in inH inL on onH onL oRn oRnH oRnL un unH unL \
#   --downsample_factor 320

# python feature_vowel_classifier.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/syllables_plain_vowels/syllables_train_dev_yor_frames.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/syllables_plain_vowels/syllables_train_dev_yor_plain_vowels.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/syllables_plain_vowels/syllables_model_frames_plain_vowelsit_1000.pkl
  #--labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/syllables/syllables_train_dev_yor_tones.npy \


# python feature_vowel_classifier.py \
#   --features_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/syllables_alltrain_yor_vectors_320.npy \
#   --labels_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/tones_alltrain_yor_labels_320.npy \
#   --model_output_path /disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/syllables_model_frames_1000it_320.pkl

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
