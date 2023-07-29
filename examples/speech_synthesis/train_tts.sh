#!/bin/bash
#SBATCH -N 1	  # nodes requested
#SBATCH -n 2	  # tasks requested
#SBATCH --gres=gpu:4  # use 1 GPU
#SBATCH -t 7-00:00:00  # time requested in hour:minute:seconds



# conda init bash
# conda activate tts

FEATURE_MANIFEST_ROOT=/home/s2324992/data/LJSpeech-1.1/feature_manifest
MODEL_NAME=train_tts_4
SAVE_DIR=/home/s2324992/facebook_dep/fairseq/examples/speech_synthesis/checkpoints/$MODEL_NAME
#USR_DIR=/home/s2324992/facebook/fairseq/examples/speech_synthesis
#USR_DIR=/home/s2324992/new_clone/fairseq/examples/speech_synthesis

fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-tokens 30000 --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss


# fairseq-train ${FEATURE_MANIFEST_ROOT} \
# --user-dir $USR_DIR --save-dir ${SAVE_DIR}   --config-yaml config.yaml \
# --train-subset train --valid-subset dev   --num-workers 2 --max-tokens 30000 \
# --max-update 200000   --task text_to_speech --criterion tacotron2 --arch tts_transformer \
# --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0   --dropout 0.1 --attention-dropout 0.1 \
# --activation-dropout 0.1   --encoder-normalize-before --decoder-normalize-before \
# --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
# --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss

# NUM_GPUS=2
# CPUS_PER_TASK=2
# MEM=32000
# EXCLUDE=arnold
# #EXCLUDE=duflo,arnold
# # srun --part=ILCC_GPU,CDT_GPU --gres=gpu:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash
# #srun --part=ILCC_GPU,CDT_GPU --gres=gpu:gtx2080ti:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash

# cd ~
# source activate_fairseq.sh
# NUM_WORKERS=2
# UPDATE_FREQ=3
# MAX_TOKENS=20000 # 30000 is default for transformer TTS
# FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

# MODEL_NAME=test_vanilla_tts_transformer
# FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
# fairseq-train ${FEATURE_MANIFEST_ROOT} \
#   --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
#   --config-yaml config.yaml --train-subset train --valid-subset dev \
#   --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
#   --task text_to_speech --criterion tacotron2 --arch tts_transformer \
#   --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
#   --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
#   --encoder-normalize-before --decoder-normalize-before \
#   --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#   --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss