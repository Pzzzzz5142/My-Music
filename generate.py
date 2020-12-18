from fairseq.models.transformer_lm import TransformerLanguageModel
import torch
import numpy as np
import torch.nn.functional as F
from random import randint

custom_lm = TransformerLanguageModel.from_pretrained(
    "checkpoints/transformer_music/", "checkpoint_best.pt"
).cuda().eval()
model=custom_lm.models[0]
l = 10000
a=[]
s=3
input_sequence = [np.random.randint(312)]
input_tensor = torch.LongTensor(input_sequence).cuda().unsqueeze(0)
a.append(custom_lm.decode(torch.LongTensor(input_sequence).cuda()))
for ind in range(l):
    x = model(input_tensor)[0][0, -1, :]
    probs = F.softmax(x, dim=0)
    #sample prob distribution for next character
    i=randint(0,s-1)
    c = torch.multinomial(probs,3)
    c=c[i][None]    
    input_tensor = torch.cat([input_tensor[:,-2048:], c[None]], dim=1)
    a.append(custom_lm.decode(c))
    if ind%100==0:
        print('saving {}'.format(ind))
        with open("fl.txt", "w") as fl:
            print(" ".join(a), file=fl)
# "Barack Obama (...)"
