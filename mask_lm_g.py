from fairseq.models.transformer_lm import TransformerLanguageModel
import torch
import numpy as np
import torch.nn.functional as F
from random import randint
from midi_preprocess import encode_midi

custom_lm = (
    TransformerLanguageModel.from_pretrained(
        "/mnt/zhangyi/checkpoints/transformer_autoencoders_noise", "checkpoint_best.pt",
    )
    .cuda()
    .eval()
)
model = custom_lm.models[0]
l = 2048
a = []
s = 1
ss = ""

with open("data/mae.test.tokens", "r") as fl:
    ss = fl.readline().strip()

if len(ss) == 0:
    input_sequence = custom_lm.encode(" ".join(encode_midi("primer.mid")))[:-1]
    # input_sequence = [np.random.randint(300)]
else:
    input_sequence = custom_lm.encode(ss)[:-1]
    print(len(input_sequence))
input_tensor = torch.LongTensor(input_sequence).cuda().unsqueeze(0)
print("ok")
a.append(custom_lm.decode(torch.LongTensor(input_sequence).cuda()))
x = model(input_tensor[...,1:1+2048])[0]
y=F.softmax(x,dim=-1)[0]
distrib = torch.distributions.categorical.Categorical(probs=y[0])
next_token = distrib.sample()
los=F.nll_loss(y,input_tensor[0][1:1+2048])
a=custom_lm.decode(x)
# "Barack Obama (...)"
