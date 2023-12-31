# Train Speech Audio Corrector single speaker dataset 

## Install/setup conda env and fairseq

```bash
conda update -y -n base -c defaults conda
conda env remove -y --name fairseq
conda create -y -n fairseq python=3.8 # python must be <=3.8 for pytorch to work
conda activate fairseq
pip install --upgrade pip
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # works with ILCC cluster 2080Ti GPU
pip install transformers datasets soundfile jupyterlab ipywidgets librosa


conda install ipython # ensures that jupyter can find env python packages
pip install jupyter # ensures that jupyter can find env python packages
#conda install -y -c conda-forge librosa
#pip install -r requirements.txt

# to make env visible to jupyter notebooks @ https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook
conda install ipykernel
conda install nb_conda_kernels # or conda install nb_conda
conda install ipywidgets
python -m ipykernel install --user --name fairseq --display-name "Python (fairseq)"
pip install jupyterlab
pip install torchdistill

# get correct version of gcc (greater than 4.0 but less than 8.0) (for installing apex)
# remove cluster-scripts from PATH i.e.:
#PATH=/opt/mendeleydesktop/bin:/usr/lib64/qt-3.3/bin:/opt/mendeleydesktop/bin:/home/s1785140/miniconda3/bin:/home/s1785140/miniconda3/condabin:/home/s1785140/miniconda3/bin:/usr/local/bin:/usr/lib/jvm/java-1.8.0/bin:/opt/texlive/2019/bin/x86_64-linux:/usr/local/sbin:/usr/bin:/sbin:/usr/pgsql-12/bin:/opt/sicstus-4.0.1/bin
scl enable devtoolset-7 bash

# install apex 
cd
export LD_LIBRARY_PATH=/opt/cuda-10.2.89_440_33/lib64/
export PATH=$PATH:/opt/cuda-10.2.89_440_33/bin
whereis nvcc # check if it works
# activate newest version of gcc
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Setup TTS data 

(from https://github.com/pytorch/fairseq/blob/main/examples/speech_synthesis/docs/ljspeech_example.md)

```bash
cd /home/s1785140/data/LJSpeech-1.1/

mkdir audio_data
mkdir audio_manifest
mkdir feature_manifest

AUDIO_DATA_ROOT=/home/s1785140/data/LJSpeech-1.1/audio_data
AUDIO_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/audio_manifest

python -m examples.speech_synthesis.preprocessing.get_ljspeech_audio_manifest \
  --output-data-root ${AUDIO_DATA_ROOT} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT}

FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT}
  # --ipa-vocab --use-g2p # commented out as we want raw grapeheme inputs for TAC

##################################################################
## create new manifest for faster training via scratch disk data
#cd /home/s1785140/data/LJSpeech-1.1/feature_manifest
## backup old manifest
#if [ ! -f train_original.tsv ]; then
#    echo "train_original.tsv not found!"
#    cp train.tsv train_original.tsv
#fi
#if [ ! -f dev_original.tsv ]; then
#    echo "dev_original.tsv not found!"
#    cp dev.tsv dev_original.tsv
#fi
#if [ ! -f test_original.tsv ]; then
#    echo "test_original.tsv not found!"
#    cp test.tsv test_original.tsv
#fi
## edit path to audio in manifests
#SCRATCH_DISK=scratch_fast # for nodes with faster scratch
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK}\/s1785140\//g" train_original.tsv > train.tsv
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK}\/s1785140\//g" dev_original.tsv > dev.tsv
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK}\/s1785140\//g" test_original.tsv > test.tsv
#
#cd /home/s1785140/data/LJSpeech-1.1/feature_manifest_standardscratch # for nodes with slower scratch
#SCRATCH_DISK=scratch
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK}\/s1785140\//g" train_original.tsv > train.tsv
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK}\/s1785140\//g" dev_original.tsv > dev.tsv
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK}\/s1785140\//g" test_original.tsv > test.tsv
#
## copy log audio data to scratch space (on current node) 
#mkdir -p /disk/${SCRATCH_DISK}/s1785140 #careful sometimes scratch disk is named something else!!!
#rsync -avu /home/s1785140/data/LJSpeech-1.1/feature_manifest/logmelspec80.zip /disk/${SCRATCH_DISK}/s1785140
#
## optional run slurm job to copy data to scratch disk (on all nodes!!!)
## onallnodes /home/s1785140/fairseq/examples/speech_audio_corrector/copy_data_to_scratch.sh
```

```bash

# VCTK audio manifest @ https://github.com/pytorch/fairseq/blob/main/examples/speech_synthesis/docs/vctk_example.md
VCTKDIR=/home/s1785140/data/VCTK_fairseq
mkdir $VCTKDIR
cd $VCTKDIR

mkdir audio_data
mkdir audio_manifest
mkdir feature_manifest

# audio manifest (get_vctk_audio_manifest will download VCTK itself into '$VCTKDIR/audio_data' dir)
AUDIO_DATA_ROOT=$VCTKDIR/audio_data
AUDIO_MANIFEST_ROOT=$VCTKDIR/audio_manifest
python -m examples.speech_synthesis.preprocessing.get_vctk_audio_manifest \
  --output-data-root ${AUDIO_DATA_ROOT} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT}
  
# convert files from flac to wav
vctk_path=/home/s1785140/data/VCTK_fairseq/audio_data/VCTK-Corpus-0.92/wav48_silence_trimmed
cd $vctk_path
find . -name "*_mic2.flac" -print0 | 
    while IFS= read -r -d '' line; do 
        echo "$line"
        basename=${line//_mic2.flac/}
        sox $line ${basename}.wav
        rm $line # remove flac file
    done

# feature manifest, create feature manifest
FEATURE_MANIFEST_ROOT=$VCTKDIR/feature_manifest
python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT}

# note can optionally add SNR or CER filtering when creating feature manifest, look @ https://github.com/pytorch/fairseq/blob/main/examples/speech_synthesis/docs/vctk_example.md

# resample to 16khz for hubert extraction
cd /home/s1785140/data/VCTK_fairseq/audio_data/VCTK-Corpus-0.92
for i in wav48_silence_trimmed/*/*.wav; 
do 
#  echo $i 
  o=wav48_silence_trimmed_16kHz/${i#wav48_silence_trimmed/}
  echo $o
  speaker_dir="$(dirname "${o}")"
#  echo $speaker_dir
  mkdir -p $speaker_dir
#  sox -v 0.95 "$i" -r 16000 "${o%.wav}.wav" #need to reduce vol to stop clipping of samples?
  sox "$i" -r 16000 "${o%.wav}.wav"
done

# extract hubert codes (do on gpu node for better speed)
TYPE=hubert
NUM_CLUSTERS=100
KM_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km${NUM_CLUSTERS}.bin
ACSTC_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/hubert_base_ls960.pt
LAYER=6
MANIFEST=~/fairseq/examples/speech_audio_corrector/manifests/vctk_fairseq.txt # created using notebooks/create_audio_manifest_for_hubert_code_extraction.ipynb
OUT_QUANTIZED_FILE=~/fairseq/examples/speech_audio_corrector/vctk_fairseq_quantized_km${NUM_CLUSTERS}.txt
EXTENSION=".wav"

cd ~/fairseq
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $ACSTC_MODEL_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXTENSION
    
TYPE=hubert
NUM_CLUSTERS=200
KM_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km${NUM_CLUSTERS}.bin
ACSTC_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/hubert_base_ls960.pt
LAYER=6
MANIFEST=~/fairseq/examples/speech_audio_corrector/manifests/vctk_fairseq.txt
OUT_QUANTIZED_FILE=~/fairseq/examples/speech_audio_corrector/vctk_fairseq_quantized_km${NUM_CLUSTERS}.txt
EXTENSION=".wav"

cd ~/fairseq
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $ACSTC_MODEL_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXTENSION
```

To investigate what scratch disks are available on a node get an interactive job on it:

```bash
NODE=duflo; srun --part=ILCC_GPU,CDT_GPU --nodelist=$NODE --pty bash
```

## (Optional) setup hifigan vocoder

Download checkpoint from hifigan repo (e.g. universal V1)

In feature_manifest/config.yaml add 

```yaml
vocoder:
  type: hifigan
  config: /home/s1785140/pretrained_models/hifigan/config.json
  checkpoint: /home/s1785140/pretrained_models/hifigan/g_02500000
```

Then add --vocoder hifigan to model training commands

## Vanilla TTS training command
```bash
NUM_GPUS=2
CPUS_PER_TASK=2
MEM=32000
EXCLUDE=arnold
#EXCLUDE=duflo,arnold
srun --part=ILCC_GPU,CDT_GPU --gres=gpu:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash
#srun --part=ILCC_GPU,CDT_GPU --gres=gpu:gtx2080ti:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash

cd ~
source activate_fairseq.sh
NUM_WORKERS=2
UPDATE_FREQ=3
MAX_TOKENS=20000 # 30000 is default for transformer TTS
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

MODEL_NAME=test_vanilla_tts_transformer
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss
```

## Choosing max tokens and update freq depending on number of GPUs

### default single gpu training for transformer TTS
default 30000 max tokens x 8 update frequency = 240000 tokens per update

### getting to new values for multigpu training

30000 max tokens per device results in OOM
want to reduce to 20000 per device

keep 240000 tokens per update as constant

4 gpus

240000 / (20000*4) = 3 

therefore set update freq to 3

Total tokens desired per update	240000
Num gpus	4

Calculate max tokens for different update freq:
max_tokens_per_gpu = total_tokens_per_update / (update_freq * num_gpus)
==============================
update_freq	max_tokens_per_gpu
==============================
2	        30000
3	        20000             # seems to work fine with 4 2080s/1080s
4	        15000
5	        12000
6	        10000
7	        8571.428571
8	        7500
9	        6666.666667
10	        6000
11	        5454.545455
12	        5000
13	        4615.384615
14	        4285.714286
15	        4000
16	        3750

## Training command (for debugging)
```bash
##################################################################################################
# Get suitable GPU node
NUM_GPUS=4
CPUS_PER_TASK=2
MEM=32000
EXCLUDE=arnold
#EXCLUDE=duflo,arnold
srun --part=ILCC_GPU,CDT_GPU --gres=gpu:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash
#srun --part=ILCC_GPU,CDT_GPU --gres=gpu:gtx2080ti:$NUM_GPUS --cpus-per-task=$CPUS_PER_TASK --mem=$MEM --exclude=$EXCLUDE --pty bash

##################################################################################################
# Set training params
cd ~
source activate_fairseq.sh
NUM_GPUS=4
NUM_WORKERS=2

UPDATE_FREQ=3
MAX_TOKENS=20000 # 30000 is default for transformer TTS
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest

##################################################################################################
# Run training for different experiments
MODEL_NAME=test_sac_normal_masking_maxtokens${MAX_TOKENS}_updatefreq${UPDATE_FREQ}_gpus${NUM_GPUS}
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss  
  #--eval-inference --best-checkpoint-metric mcd_lossmetric mcd_lossv \
  
MODEL_NAME=test_sac_mask_all_speechreps
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss
  
MODEL_NAME=test_randomise_examples
fairseq-train ${FEATURE_MANIFEST_ROOT} \
  --save-dir checkpoints/$MODEL_NAME --tensorboard-logdir tb_logs/$MODEL_NAME \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers $NUM_WORKERS --max-tokens $MAX_TOKENS --max-update 200000 \
  --task speech_audio_corrector --criterion sac_tts --arch sac_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq $UPDATE_FREQ --best-checkpoint-metric loss \
  --randomise-examples-p 1.0
```

## Generate waveform

## Normal TTS speech synthesis 

Example follows https://github.com/pytorch/fairseq/blob/main/examples/speech_synthesis/docs/ljspeech_example.md#inference

### TTS Inference from vanilla transformer TTS model

```bash
cd ~/fairseq

MODEL=test_vanilla_tts_transformer
SAVE_DIR=checkpoints/$MODEL
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
#CHECKPOINT_NAME=avg_last_5
CHECKPOINT_NAME=last
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
OUT_DIR=inference/$MODEL/$CHECKPOINT_NAME
#python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
#  --num-epoch-checkpoints 5 \
#  --output ${CHECKPOINT_PATH}

python -m examples.speech_synthesis.generate_waveform ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task text_to_speech \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --dump-waveforms
  
# copy wavs back to macbook
./copy_fairseq_samples_back_to_macbook.sh # run this script from macbook 
```

### TTS Inference from SAC model (over different sets of data. Test set, utts with test set words OOV in training, and same for dev set.)

```bash
cd ~/fairseq

MODEL=test_sac_normal_masking2
VOCODER=wav_22050hz_hifigan
SAVE_DIR=checkpoints/$MODEL
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
OUT_DIR=inference/$MODEL/$CHECKPOINT_NAME

# optionally form an averaged checkpoint
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 5 \
  --output ${CHECKPOINT_PATH}

# generate entire test set (using random masking just like training)
SPLIT=test
python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --vocoder hifigan \
  --dump-waveforms
mkdir $OUT_DIR/$VOCODER/LJ_TEST_SET
mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/LJ_TEST_SET
  
# generate txt file of utterances
# where we can control which words are swapped to speechreps i.e. "how is <champagne> pronounced"
#SPLIT=train
SPLIT=test
python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --vocoder hifigan \
  --dump-waveforms \
  --batch-size 32 \
  --speechreps-add-mask-tokens \
  --txt-file examples/speech_audio_corrector/test_utts.txt
mkdir $OUT_DIR/$VOCODER/test_utts
mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/test_utts

# TEST SET OOVS (words in test set but not in train)  
SPLIT=test
python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --vocoder hifigan \
  --dump-waveforms \
  --batch-size 32 \
  --speechreps-add-mask-tokens \
  --txt-file examples/speech_audio_corrector/test_utts_test_set_oovs.txt \
  --add-count-to-filename
mkdir $OUT_DIR/$VOCODER/test_oovs
mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/test_oovs
  

# DEV SET OOVS (words in dev set but not in train)
SPLIT=dev 
python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --vocoder hifigan \
  --dump-waveforms \
  --batch-size 32 \
  --speechreps-add-mask-tokens \
  --txt-file examples/speech_audio_corrector/test_utts_dev_set_oovs.txt \
  --add-count-to-filename
mkdir $OUT_DIR/$VOCODER/dev_oovs
mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/dev_oovs
```

### Generate using external speechreps (from VCTK)

```bash
cd ~/fairseq

MODEL=test_sac_normal_masking2
VOCODER=wav_22050hz_hifigan
SAVE_DIR=checkpoints/$MODEL
SPLIT=test
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
NUM=3500
CHECKPOINT_NAME=epoch${NUM}
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint${NUM}.pt
OUT_DIR=inference/$MODEL/${CHECKPOINT_NAME}

python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --results-path $OUT_DIR \
  --vocoder hifigan \
  --dump-waveforms \
  --batch-size 32 \
  --speechreps-add-mask-tokens \
  --txt-file examples/speech_audio_corrector/test_utts_vctk_oovs.txt \
  --add-count-to-filename \
  --use-external-speechreps
```

### Inference from single model at different checkpoints

```bash
cd ~/fairseq

MODEL=test_sac_normal_masking2
VOCODER=wav_22050hz_hifigan
SAVE_DIR=checkpoints/$MODEL
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
SPLIT=test # dataset to get word-aligned speech reps from
TXT_FILE=examples/speech_audio_corrector/test_utts_test_set_oovs.txt

for NUM in 500 1000 1500 2000 2500 3000 3500
do
    CHECKPOINT_NAME=epoch${NUM}
    CHECKPOINT_PATH=${SAVE_DIR}/checkpoint${NUM}.pt
    OUT_DIR=inference/$MODEL/$CHECKPOINT_NAME
  
    # generate LJ TEST set
    python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
      --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
      --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
      --results-path $OUT_DIR \
      --vocoder hifigan \
      --dump-waveforms
    mkdir $OUT_DIR/$VOCODER/LJ_TEST_SET
    mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/LJ_TEST_SET
  
#    # generate carrier sentence test set (words that are in test set but OOV in training set)
#    python -m examples.speech_audio_corrector.generate_waveform_sac ${FEATURE_MANIFEST_ROOT} \
#      --config-yaml config.yaml --gen-subset ${SPLIT} --task speech_audio_corrector \
#      --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
#      --results-path $OUT_DIR \
#      --vocoder hifigan \
#      --dump-waveforms \
#      --batch-size 32 \
#      --speechreps-add-mask-tokens \
#      --txt-file $TXT_FILE \
#      --add-count-to-filename
#    mkdir $OUT_DIR/$VOCODER/test_oovs
#    mv $OUT_DIR/$VOCODER/*.wav $OUT_DIR/$VOCODER/test_oovs
done


```

## Speech Audio Correction speech synthesis

## Setup Speech Reps data

1) obtain discretised speech reps
https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm/speech2unit
either 
a) generate yourself:
```bash
# convert LJSpeech to 16khz 

# create manifests file (look at lj_speech_manifest.txt for an example)
# top line should be path to the folder containing the wav files

#download checkpoints
#hubert: https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
#k-means clusters: https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin

# run speech reps model and quantisation
TYPE=hubert
KM_MODEL_PATH=../../fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km100/hubert_km100.bin
ACSTC_MODEL_PATH=../../fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/hubert_base_ls960.pt
LAYER=6
MANIFEST=filelists/voice_conversion_test.txt
OUT_QUANTIZED_FILE=speech2unit_output/quantized/voice_conversion_test_quantized.txt
EXTENSION=".wav"

CUDA_VISIBLE_DEVICES=9 python ../../fairseq/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $ACSTC_MODEL_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXTENSION
```
b) or use precomputed ones: fairseq/fairseq/models/lexicon_learner/lj_speech_quantized.txt

2) align speech reps at the word level using mfa 
run ipynb script

3) get lookup table (also look at fairseq/examples/lexicon_learner/get_hubert_lookup_table.py)
```python 
import joblib
kmeans_model_path = '../../fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km100/km.bin'
kmeans_model = joblib.load(open(kmeans_model_path, "rb")) # this is just a sklearn model
centroids = kmeans_model.cluster_centers_
```

## Train model



```bash
FEATURE_MANIFEST_ROOT=/home/s1785140/data/LJSpeech-1.1/feature_manifest
MODEL_NAME=test_tac
SAVE_DIR=checkpoints/$MODEL_NAME
fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-tokens 30000 --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss
```


```bash
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-train $DATA \
    --task learn_lexicon \
    --arch lexicon_learner \
    --optimizer adam \
    --batch-size 4 \
    --num-wordtypes 100 \
    --max-examples-per-wordtype 100
```

Commands for fast debugging training of this model (hubert):

```bash
MODEL_NAME=test_hubert
DATA=/home/s1785140/data/ljspeech_hubert_reps/hubert-base/layer-6/word_level_without_padding_idx_offset
cd ~/fairseq
fairseq-train $DATA \
    --tensorboard-logdir tb_logs/$MODEL_NAME \
    --task learn_lexicon_discrete_inputs \
    --arch lexicon_learner_seq2seq \
    --criterion lexicon_learner \
    --sequence-loss-method summariser \
    --optimizer adam \
    --batch-size 2 \
    --padding-index-offset 1 \
    --max-train-wordtypes 10 \
    --min-train-examples-per-wordtype 2 \
    --max-train-examples-per-wordtype 2 \
    --valid-seen-wordtypes 5 \
    --valid-unseen-wordtypes 5 \
    --valid-examples-per-wordtype 2 \
    --valid-subset valid-seen,valid-unseen \
    --save-dir checkpoints/$MODEL_NAME \
    --save-interval 1 --max-epoch 2 \
    --lr 0.001 \
    --cache-all-data \
    --debug-only-include-words-beginning-with b \
    --normalize-out \
    --transformer-mask-outputs \
    --no-save
```

Commands for debugging training of this model (wav2vec2):

```bash
MODEL_NAME=debugging
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-train $DATA \
    --tensorboard-logdir tb_logs/$MODEL_NAME \
    --task learn_lexicon \
    --arch lexicon_learner \
    --criterion lexicon_learner \
    --optimizer adam \
    --batch-size 64 \
    --min-train-examples-per-wordtype 10 \
    --max-train-examples-per-wordtype 10 \
    --valid-seen-wordtypes 100 \
    --valid-unseen-wordtypes 100 \
    --valid-examples-per-wordtype 10 \
    --valid-subset valid-seen,valid-unseen \
    --save-dir checkpoints/$MODEL_NAME \
    --save-interval 1 --max-epoch 2 \
    --lr 0.001 \
    --no-save
```

To submit as a slurm job, prepend the slurm script:

```bash
MODEL_NAME=test_model3
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
../sbatch.sh 2080 fairseq-train $DATA \
    --tensorboard-logdir tb_logs/$MODEL_NAME \
    --task learn_lexicon \
    --arch lexicon_learner \
    --criterion lexicon_learner \
    --optimizer adam \
    --batch-size 8 \
    --max-train-wordtypes 25 \
    --valid-seen-wordtypes 10 \
    --valid-unseen-wordtypes 10 \
    --max-train-examples-per-wordtype 25 \
    --valid-subset valid-seen,valid-unseen \
    --save-interval 1 --max-epoch 2 \
    --save-dir checkpoints/$MODEL_NAME \
    --no-save
```

(GET ME WORKING!) From config file and command line **(NOTE WE ARE USING fairseq-hydra-train now)**:

```bash
DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
fairseq-hydra-train \
    --config-dir /home/s1785140/fairseq/examples/lexicon_learner/config \
    --config-name ALL_ljspeech \
    task.data=$DATA \
    dataset.batch_size=2 \
    --max-num-wordtypes 50 \
    --max-train-examples-per-wordtype 50 \
    --max-epoch 5 \
    --no-save
```

## Generate features for SAC inference

How to generate features for a new dataset.

### Get MFA alignments

Rearrange VCTK file structure for use with MFA

```bash
REARRANGED='/home/s1785140/data/VCTK_fairseq/audio_data/VCTK-Corpus-0.92/wav48_silence_trimmed_rearranged_for_MFA'

cd ~/fairseq/examples/speech_audio_corrector/
python reorganize_vctk_for_montreal.py \
  --input_dir ~/data/VCTK_fairseq/audio_data/VCTK-Corpus-0.92 \
  --output_dir $REARRANGED
```

We need to generate ground truth alignments for speech corpus

```bash 
# update conda
conda update -n base -c defaults conda

#install MFA in new conda env
conda create -n aligner -c conda-forge montreal-forced-aligner
source ~/.bashrc
conda activate aligner

# download models
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa

# run validate and align
REARRANGED='/home/s1785140/data/VCTK_fairseq/audio_data/VCTK-Corpus-0.92/wav48_silence_trimmed_rearranged_for_MFA'
OUTDIR='/home/s1785140/data/VCTK_fairseq/mfa_alignments'
mfa validate $REARRANGED english_mfa english_mfa
mfa align --clean $REARRANGED english_mfa english_mfa $OUTDIR

# keep track of unaligned files (might want to exclude these from audio and features manifest
# @/home/s1785140/Documents/MFA/wav48_silence_trimmed_rearranged_for_MFA_validate_pretrained/unalignable_files.csv
```

# Speech Audio Corrector multispeaker dataset

## Combine different datasets (optional)

In order to train a model that reconstructs both LJspeech and VCTK then we need to make a combined feature manifest that contains
both LJ and VCTK.

We do this by combining train dev and test tsv files using the following script

```bash
mkdir -p /home/s1785140/data/LJ_and_VCTK_fairseq/feature_manifest/
cd examples/speech_audio_corrector

python combine_manifests.py \
    --in_tsvs /home/s1785140/data/LJSpeech-1.1/feature_manifest/train.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/train.tsv \
    --out_tsv /home/s1785140/data/LJ_and_VCTK_fairseq/feature_manifest/train.tsv
    
python combine_manifests.py \
    --in_tsvs /home/s1785140/data/LJSpeech-1.1/feature_manifest/dev.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/dev.tsv \
    --out_tsv /home/s1785140/data/LJ_and_VCTK_fairseq/feature_manifest/dev.tsv
    
python combine_manifests.py \
    --in_tsvs /home/s1785140/data/LJSpeech-1.1/feature_manifest/test.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/test.tsv \
    --out_tsv /home/s1785140/data/LJ_and_VCTK_fairseq/feature_manifest/test.tsv
```

Then backup the original tsvs in the VCTK feature manifest directory 

```bash
mv /home/s1785140/data/VCTK_fairseq/feature_manifest/train.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/train.backup.tsv
mv /home/s1785140/data/VCTK_fairseq/feature_manifest/dev.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/dev.backup.tsv
mv /home/s1785140/data/VCTK_fairseq/feature_manifest/test.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/test.backup.tsv
mv /home/s1785140/data/VCTK_fairseq/feature_manifest/speakers.txt /home/s1785140/data/VCTK_fairseq/feature_manifest/speakers.backup.txt
```

Then we replace the original tsvs with the new combined tsvs. (we do this in the VCTK directory and not in the LJ Speech one
because we want to enable multispeaker training. Which is achieved by defining a list of speakers AKA 'speaker_set_filename' in
config.yaml. This is all done automatically when creating the VCTK feature manifest by using the get_vctk_audio_manifest()
and get_feature_manifest() functions.

```bash
# copy ljspeech and vctk manifests in
cp /home/s1785140/data/LJ_and_VCTK_fairseq/feature_manifest/train.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/
cp /home/s1785140/data/LJ_and_VCTK_fairseq/feature_manifest/dev.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/
cp /home/s1785140/data/LJ_and_VCTK_fairseq/feature_manifest/test.tsv /home/s1785140/data/VCTK_fairseq/feature_manifest/
```

Then, we must also add 'ljspeech' to the speakers.txt file created when creating feature manifest for VCTK. (Note that 
this speakers.txt file is "speaker_set_filename" referred to in config.yaml and 
fairseq.tasks.speech_to_text.SpeechToTextTask._get_speaker_to_id 

```bash
# Add 'ljspeech' to last line of ~/data/VCTK_fairseq/feature_manifest/speakers.txt
echo ljspeech >> /home/s1785140/data/VCTK_fairseq/feature_manifest/speakers.txt
```

## Extract speaker embeddings (xvectors, x-vectors) for multispeaker TTS training

### Using speechbrain 

(Instructions @ https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb or https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)

Install speechbrain

```bash
# update conda
conda update -n base -c defaults conda

#install in new conda env
conda create -n speechbrain python=3.8
source ~/.bashrc
conda activate speechbrain

pip install --upgrade pip
pip install speechbrain
conda install -c conda-forge jupyterlab

# test installation
pytest tests
pytest --doctest-modules speechbrain
```

Get GPU node

```bash
srun --part=ILCC_GPU,CDT_GPU --gres=gpu:1 --cpus-per-task=2 --exclude=arnold,duflo --mem=16000 --pty bash
conda activate speechbrain
```

Run jupyter notebook or python script to extract

```bash
cd ~/fairseq/examples/speech_audio_corrector/
jupyter-lab --no-browser --ip=0.0.0.0
# go to notebooks folder

python extract_speaker_vectors.py --dataset vctk
python extract_speaker_vectors.py --dataset ljspeech
```

## Create speaker emb numpy array that is loaded as pretrained speaker embeddings for TTS or SAC training 

We need to create the file referenced as 'speaker_emb_filename' and is defined in config.yaml

Create speaker embedding numpy array according to the speaker embeddings you've calculated and the order of speakers 
in speakers.txt

```bash
# note the order of speakers in speaker_emb_mat should match the order of speakers in  speakers.txt 
# as the order of the speakers in speakers.txt  defines the order in speaker_to_id
cd ~/fairseq/examples/speech_audio_corrector/
python create_spk_emb_table.py \
  --embeddings_dir /home/s1785140/fairseq/examples/speech_audio_corrector/speaker_embeddings/ \
  --speaker_list /home/s1785140/data/VCTK_fairseq/feature_manifest/speakers.txt \
  --outfile /home/s1785140/data/VCTK_fairseq/feature_manifest/speaker_embeddings.npy 
```

## Set path to 'speaker_emb_path' in config.yaml

echo 'speaker_emb_path: speaker_embeddings.npy' >> /home/s1785140/data/VCTK_fairseq/feature_manifest/config.yaml

## Get HuBERT codes for wavs in corpus 

Instructions @ https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit

Resample speech to that required by Hubert (16Khz)

Use fairseq/examples/speech_audio_corrector/bash_scripts/recursive_resample.sh

LJ Speech
```bash
cd ~/data/LJSpeech-1.1/audio_data/LJSpeech-1.1/
mkdir -p wavs_16kHz
for i in wavs/*.wav; 
do 
  echo $i 
  o=wavs_16kHz/${i#wavs/}
  sox "$i" -r 16000 "${o%.wav}.wav"
done
```

VCTK
```bash
cd ~/data
for i in vqvae_wavernn_trimmed_wavs/*/*.wav; 
do 
#  echo $i 
  o=vqvae_wavernn_trimmed_wavs_16kHz/${i#vqvae_wavernn_trimmed_wavs/}
  echo $o
  speaker_dir="$(dirname "${o}")"
#  echo $speaker_dir
  mkdir -p $speaker_dir
#  sox -v 0.95 "$i" -r 16000 "${o%.wav}.wav" #need to reduce vol to stop clipping of samples?
  sox "$i" -r 16000 "${o%.wav}.wav"
done
```

### Create audio manifests file

create filelists/vctk.txt

Note about the manifest file is a file with paths and length of input audio files. The format of the file is as follows:

Seems to be ok to put 0 for number_of_samples

```bash
<path_of_root_directory_containing_audio_files>
<relative_path_of_audio_file_1>\t<number_of_samples1>
<relative_path_of_audio_file_2>\t<number_of_samples2>
...
```

use create_audio_manifest_for_hubert_code_extraction.ipynb to do this

### Download pretrained hubert and k-means clustering models

https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit#quantization-model

```bash
cd fairseq/examples/textless_nlp/gslm/speech2unit/
mkdir -p pretrained_models/hubert
cd pretrained_models/hubert
# hubert model
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
# kmeans
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin
mv km.bin km100.bin
```

### extract codes

Get k means model
```bash
cd ~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin
mv km.bin km100.bin
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin
mv km.bin km50.bin
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km200/km.bin
mv km.bin km200.bin
```

LJ Speech 
```bash
NUM_CLUSTERS=50  # change to 50 100 or 200
TYPE=hubert
KM_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km${NUM_CLUSTERS}.bin
ACSTC_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/hubert_base_ls960.pt
LAYER=6
MANIFEST=~/fairseq/examples/speech_audio_corrector/manifests/ljspeech.txt
OUT_QUANTIZED_FILE=~/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km${NUM_CLUSTERS}.txt
EXTENSION=".wav"

# CUDA_VISIBLE_DEVICES=9
cd ~/fairseq  
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $ACSTC_MODEL_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXTENSION
    
NUM_CLUSTERS=100  # change to 50 100 or 200
TYPE=hubert
KM_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km${NUM_CLUSTERS}.bin
ACSTC_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/hubert_base_ls960.pt
LAYER=6
MANIFEST=~/fairseq/examples/speech_audio_corrector/manifests/ljspeech.txt
OUT_QUANTIZED_FILE=~/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km${NUM_CLUSTERS}.txt
EXTENSION=".wav"

# CUDA_VISIBLE_DEVICES=9
cd ~/fairseq  
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $ACSTC_MODEL_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXTENSION
    
NUM_CLUSTERS=200  # change to 50 100 or 200
TYPE=hubert
KM_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km${NUM_CLUSTERS}.bin
ACSTC_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/hubert_base_ls960.pt
LAYER=6
MANIFEST=~/fairseq/examples/speech_audio_corrector/manifests/ljspeech.txt
OUT_QUANTIZED_FILE=~/fairseq/examples/speech_audio_corrector/ljspeech_quantized_km${NUM_CLUSTERS}.txt
EXTENSION=".wav"

# CUDA_VISIBLE_DEVICES=9
cd ~/fairseq  
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $ACSTC_MODEL_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXTENSION
```

VCTK
```bash
TYPE=hubert
NUM_CLUSTERS=100
KM_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km${NUM_CLUSTERS}.bin
ACSTC_MODEL_PATH=~/fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/hubert_base_ls960.pt
LAYER=6
MANIFEST=~/fairseq/examples/speech_audio_corrector/manifests/vctk.txt
OUT_QUANTIZED_FILE=~/fairseq/examples/speech_audio_corrector/vctk_quantized_km${NUM_CLUSTERS}.txt
EXTENSION=".wav"

# CUDA_VISIBLE_DEVICES=9
cd ~/fairseq  
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $ACSTC_MODEL_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension $EXTENSION
```

# Low resource alignments for SAC 

Get CTC-segmentation word alignments for SAC-LR (low resource training of SAC)

No need to use mfa alignments

##Install huggingface for getting CTC outputs

```bash
conda update -y -n base -c defaults conda
conda env remove -y --name huggingface
conda create -y -n huggingface python=3.8 # python must be <=3.8 for pytorch to work
conda activate huggingface
pip install --upgrade pip
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Use examples/speech_audio_corrector/notebooks/CTC_word_segmentation_hubert.ipynb

# Using tensorboard from local computer 

## 1) On local

https://stackoverflow.com/questions/38464559/how-to-locally-view-tensorboard-of-remote-server

```bash
ssh -NfL 1337:localhost:1337 username@remote_server_address

# i.e. 
ssh -NfL 1337:localhost:1337 s1785140@escience6.inf.ed.ac.uk
```

## 2) On server

ensure node has internet access (GPU nodes often do not)

```bash
source activate_fairseq.sh
tensorboard \
    --logdir=tb_logs/ \
    --port 1337 --bind_all \
    --max_reload_threads 4 # speed up loading of runs in log dir
```

If tensorboard is loading very slowly (perhaps because too many models in the logdir)
move some log folders out in old_tb_logs dir.


## Tips

If you can see tensorboard logs from another user, change the port number.

# Evaluate model


## Remote jupyter notebook development

Choose A or B:

### A: Using configured server (started yourself on a gpu node)

Instructions adapted from:
https://nero-docs.stanford.edu/jupyter-slurm.html

1. Start up jupyter server on remote GPU node. Make a note of the node you were assigned to ('duflo' in this example). 

```bash
ssh s1785140@escience6.inf.ed.ac.uk # replace s1785140 with your dice username
srun --part=ILCC_GPU,CDT_GPU --gres=gpu:gtx2080ti:1 --cpus-per-task=1 --mem=8000 --pty bash
conda activate fairseq
cd examples/speech_audio_corrector/notebooks/ # (optional) go to project root to change the 'cwd' of jupyter
jupyter-lab --no-browser --ip=0.0.0.0 # or jupyter-lab --no-browser --ip=0.0.0.0
```

2. Find the notebook URL related to the node that you were assigned. For example in the example below the correct link is `http://duflo.inf.ed.ac.uk:8888/?token=95dd3ae95c8d91c466405cbcaf8114e944b85d731b481183`

```bash
[I 16:23:20.776 NotebookApp] Serving notebooks from local directory: /disk/nfs/ostrom/s1785140
[I 16:23:20.777 NotebookApp] Jupyter Notebook 6.4.0 is running at:
[I 16:23:20.777 NotebookApp] http://duflo.inf.ed.ac.uk:8888/?token=95dd3ae95c8d91c466405cbcaf8114e944b85d731b481183
[I 16:23:20.777 NotebookApp]  or http://127.0.0.1:8888/?token=95dd3ae95c8d91c466405cbcaf8114e944b85d731b481183
[I 16:23:20.777 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

3. Either click the link to directly develop in your browser or copy the link and paste into pycharm as a configured jupyter notebook server to develop there.

4. Test if GPU works in jupyter notebook. Enter in cell:

```python
import torch
torch.tensor([1.0, 2.0]).cuda()
```

If you want to make access to the notebook easier from the url without a token string, then set your jupyter notebook password

```bash
jupyter notebook password
```

### B: Using managed server (managed/started by pycharm) NOT SOLVED!!!
First need to setup a remote interpreter for jupyter to run in (https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html)

https://www.jetbrains.com/help/pycharm/configuring-jupyter-notebook.html

NOT SOLVED... some info about how to possibly run pycharm using srun interactive gpu on slurm
https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000489490-Running-remote-interpreter-on-a-cluster-with-srun
https://researchcomputing.princeton.edu/support/knowledge-base/pytorch#jupyter


## Remote debugging

https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html#remote-interpreter

