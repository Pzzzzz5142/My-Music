from fairseq.models.transformer_lm import TransformerLanguageModel
import torch
import numpy as np
import torch.nn.functional as F
from random import randint

custom_lm = (
    TransformerLanguageModel.from_pretrained(
        "/mnt/zhangyi/checkpoints/transformer_music_fs_split_fp16_relative_mixed/",
        "checkpoint_best.pt",
    )
    .cuda()
    .half()
    .eval()
)
model = custom_lm.models[0]
l = 4096
a = []
s = 1
with open("fl.txt", "r") as fl:
    ss = fl.read().strip()
if len(ss) == 0:
    input_sequence = [np.random.randint(300)]
else:
    input_sequence = custom_lm.encode(ss)[:-1]
input_tensor = torch.LongTensor(input_sequence).cuda().unsqueeze(0)
print("ok")
a.append(custom_lm.decode(torch.LongTensor(input_sequence).cuda()))
for ind in range(l):
    x = model(input_tensor)[0][0, -1, :]
    probs = F.softmax(x, dim=0)
    # sample prob distribution for next character
    i = randint(0, s - 1)
    c = torch.multinomial(probs, s)
    c = c[i][None]
    input_tensor = torch.cat([input_tensor[:, :], c[None]], dim=1)
    a.append(custom_lm.decode(c))
    if ind % 100 == 0:
        print("saving {}".format(ind))
        with open("fl.txt", "w") as fl:
            print(" ".join(a), file=fl)
# "Barack Obama (...)"
