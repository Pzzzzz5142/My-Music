from fairseq.data.fasta_dataset import FastaDataset
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
    FairseqDecoder,
)
from fairseq.modules.positional_encoding import PositionalEncoding
from fairseq.modules.rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR
import logging

logger = logging.getLogger(__name__)

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

TOKEN_END = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD = TOKEN_END + 1

VOCAB_SIZE = TOKEN_PAD + 1

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int32

TORCH_LABEL_TYPE = torch.long

PREPEND_ZEROS_WIDTH = 4


# MusicTransformer
@register_model("music_transformer")
class MusicTransformer(FairseqLanguageModel):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(self, decoder):
        super().__init__(decoder)
        self.decoder = decoder

    @classmethod
    def build_model(cls, args, task):

        haha(args)

        decoder = HAHA(Dictionary=task.target_dictionary)

        return cls(decoder)


class HAHA(FairseqDecoder):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(
        self,
        n_layers=6,
        num_heads=8,
        d_model=512,
        dim_feedforward=1024,
        dropout=0.1,
        max_sequence=2048,
        rpr=True,
        Dictionary=None,
    ):
        super().__init__(Dictionary)

        self.dummy = DummyDecoder()

        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = max_sequence
        self.rpr = rpr

        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.d_model, self.dropout, self.max_seq
        )

        # Base transformer
        if not self.rpr:
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = nn.Transformer(
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.nlayers,
                num_decoder_layers=0,
                dropout=self.dropout,  # activation=self.ff_activ,
                dim_feedforward=self.d_ff,
                custom_decoder=self.dummy,
            )
        # RPR Transformer
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(
                self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq
            )
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.nlayers,
                num_decoder_layers=0,
                dropout=self.dropout,  # activation=self.ff_activ,
                dim_feedforward=self.d_ff,
                custom_decoder=self.dummy,
                custom_encoder=encoder,
            )

        # Final output is a softmaxed linear layer
        self.Wout = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax = nn.Softmax(dim=-1)

    # forward
    def forward(
        self, x, mask=False, src_lengths=None,
    ):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """

        if mask is True:
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).cuda()
        else:
            mask = None

        x = self.embedding(x)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1, 0, 2)

        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1, 0, 2)

        y = self.Wout(x_out)
        # y = self.softmax(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y, y


# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(
        self,
        tgt,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
    ):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """

        return memory


@register_model_architecture("music_transformer", "music_transformer")
def haha(args):
    pass
