from fairseq.models.transformer_lm import TransformerLanguageModel
import torch
import numpy as np
import torch.nn.functional as F
from random import randint
from midi_preprocess import encode_midi

custom_lm = (
    TransformerLanguageModel.from_pretrained(
        "/mnt/zhangyi/checkpoints/transformer_music_fs_split_fp16_relative_mixed/", "checkpoint_best.pt",
    )
    .cuda()
    .half()
    .eval()
)
model = custom_lm.models[0]
l = 2048
a = []
s = 1
ss = []
"""
with open("fl.txt", "r") as fl:
    ss = fl.read().strip()
"""
if len(ss) == 0:
    input_sequence = custom_lm.encode(" ".join(encode_midi("primer.mid")))[:-1]
    # input_sequence = [np.random.randint(300)]
else:
    input_sequence = custom_lm.encode(ss)[:-1]
    print(len(input_sequence))
input_tensor = torch.LongTensor(input_sequence).cuda().unsqueeze(0)
print("ok")
a.append(custom_lm.decode(torch.LongTensor(input_sequence).cuda()))
print(input_sequence.shape)
for ind in range(len(input_sequence), l):
    x = model(input_tensor[-2000:, :])[0]
    x = F.softmax(x, dim=2)[:, -1, :]
    distrib = torch.distributions.categorical.Categorical(probs=x[None])
    next_token = distrib.sample()
    input_tensor = torch.cat([input_tensor[:, :], next_token], dim=1)
    a.append(custom_lm.decode(next_token))
    if ind % 100 == 0:
        print("saving {}".format(ind))
        with open("fl.txt", "w") as fl:
            print(" ".join(a), file=fl)
# "Barack Obama (...)"
