import torch

# The same interface can be used with custom models as well
from fairseq.models.transformer_lm import TransformerLanguageModel
custom_lm = TransformerLanguageModel.from_pretrained('/mnt/zhangyi/checkpoints/transformer_music_new', 'checkpoint_best.pt', tokenizer='moses')
a=custom_lm.sample('', beam=5)
with open("ff.txt",'w') as fl:
    print(a,file=fl)