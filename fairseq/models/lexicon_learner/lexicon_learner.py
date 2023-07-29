import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    register_model,
)
from fairseq.modules import (
    TransformerEncoderLayer,
)

"""
Commands for debugging training of this model:

DATA=/home/s1785140/data/ljspeech_wav2vec2_reps/wav2vec2-large-960h/layer-15/word_level/
TB_LOG_DIR=/home/s1785140/fairseq/tensorboard_logs/
fairseq-train $DATA \
    --tensorboard-logdir $TB_LOG_DIR \
    --task learn_lexicon \
    --arch lexicon_learner \
    --criterion lexicon_learner \
    --optimizer adam \
    --batch-size 2 \
    --max-num-wordtypes 50 \
    --max-train-examples-per-wordtype 50 \
    --max-epoch 5 \
    --no-save
"""

@dataclass
class LexiconLearnerConfig(FairseqDataclass):
    ##############
    # Data
    input_dim: int = field(
        default=1024, metadata={"help": "dimension of input features"}
    )

    ##############
    # Encoder
    enc_num_layers: int = field(
        default=3, metadata={"help": "number of encoder layers"}
    )
    enc_hid_dim: int = field(
        default=128, metadata={"help": "number of hidden dimensions"}
    )
    enc_out_dim: int = field(
        default=128, metadata={"help": "number of output dimensions"}
    )
    enc_dropout_in: float = field(
        default=0.2, metadata={"help": "number of output dimensions"}
    )

    enc_dropout_out: float = field(
        default=0.2, metadata={"help": "number of output dimensions"}
    )

    ##############
    # Decoder


@register_model("lexicon_learner", dataclass=LexiconLearnerConfig)
class LexiconLearner(BaseFairseqModel):
    def __init__(self, cfg: LexiconLearnerConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = LSTMEncoder(cfg)

    @classmethod
    def build_model(cls, cfg: LexiconLearnerConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def forward(self, src_tokens, src_lengths):
        src_tokens = self.encoder(src_tokens, src_lengths)
        return src_tokens


class LSTMEncoder(FairseqEncoder):

    def __init__(self, cfg):
        super().__init__(None)

        self.cfg = cfg

        self.dropout_in = nn.Dropout(p=cfg.enc_dropout_in)
        self.dropout_out = nn.Dropout(p=cfg.enc_dropout_out)

        self.lstm = nn.LSTM(
            input_size=cfg.input_dim, #TODO make input dim dynamic depending on detected dimensions
            hidden_size=cfg.enc_hid_dim,
            num_layers=cfg.enc_num_layers,
            dropout=cfg.enc_dropout_out,
            bidirectional=False,
            batch_first=True,
        )

        self.output_projection = nn.Linear(cfg.enc_hid_dim, cfg.enc_out_dim)

    def forward(
            self,
            src_tokens,
            src_lengths,
    ):
        layer_to_return = 1 # TODO make this a cfg setting?

        src_tokens = self.dropout_in(src_tokens)

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        packed_x = nn.utils.rnn.pack_padded_sequence(
            src_tokens,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False, # TODO could have bad side effects? to perf?
        )

        # Get the output from the LSTM.
        _packed_outputs, (final_timestep_hidden, _final_timestep_cell) = self.lstm(packed_x)

        # NB No need to unpack and then pad this packed seq as we simply need "final timestep hidden",
        # We do not need all timesteps which are returned in "_packed_outputs"


        if self.cfg.enc_num_layers > 1:
            # Only return from one layer of the LSTM
            final_timestep_hidden = final_timestep_hidden.squeeze(0)[layer_to_return,:,:]
        else:
            final_timestep_hidden = final_timestep_hidden.squeeze(0)


        final_timestep_hidden = self.dropout_out(final_timestep_hidden)

        final_timestep_hidden = self.output_projection(final_timestep_hidden)

        # print("XXX", final_timestep_hidden)

        return {
            # this will have shape `(bsz, hidden_dim)`
            'final_timestep_hidden': final_timestep_hidden,
        }
