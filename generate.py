from torch._C import dtype
from torch.tensor import Tensor
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.models.transformer_autoencoders import TransformerAutoencoders
import torch
import numpy as np
import torch.nn.functional as F
from random import randint
from midi_preprocess import encode_midi, decode_midi
import utils


def temperature_sampling(x, temperature, topk):
    logits = x.cpu().detach().numpy()[0]
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    if topk == 1:
        prediction = np.argmax(probs)
    else:
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return x.new([prediction]).int()[None]


def topk_sampling(x, topk):
    logits = x.cpu().detach().numpy()[0]
    probs = logits
    if topk == 1:
        prediction = np.argmax(probs)
    else:
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return x.new([prediction]).int()[None]


custom_lm = (
    TransformerLanguageModel.from_pretrained(
        "/mnt/zhangyi/checkpoints/transformer_music_lm_remi_hi", "checkpoint_best.pt",
    )
    .cuda()
    .half()
    .eval()
)
model = custom_lm.models[0]
for i in range(86, 100):
    l = 2048
    a = []
    s = 1
    ss = "Bar_None"
    if len(ss) == 0:
        input_sequence = custom_lm.encode(" ".join(encode_midi("primer.mid")))[:-1]
        with open("data/mae.test.tokens", "r") as fl:
            ss = fl.readline().strip().split()[3 : 3 + 2048]
        prev_input_seq = custom_lm.encode(" ".join(ss))[:-1]
    else:
        input_sequence = custom_lm.encode(ss)[:-1]
        print(len(input_sequence))
    input_tensor = torch.LongTensor(input_sequence).cuda().unsqueeze(0)
    # prev_input_tensor = torch.LongTensor(prev_input_seq).cuda().unsqueeze(0)
    print("ok")
    a = custom_lm.decode(torch.LongTensor(input_sequence).cuda()).split()
    print(input_sequence.shape)
    try:
        flg = 0
        for ind in range(len(input_sequence), l):
            x = model(input_tensor[-2000:, :])[0]
            y = x.clone().detach()
            x = F.softmax(x, dim=2)[:, -1, :]
            if flg:
                xx = F.softmax(y, dim=2)
                xx = xx.topk(1, dim=2)[1]
                flg -= 1
                decode_midi(
                    custom_lm.decode(xx[0, :, 0]).split(), file_path="final2.mid"
                )
            if False:
                distrib = torch.distributions.categorical.Categorical(probs=x[None])
                next_token = distrib.sample()
            elif False:
                next_token = x.topk(1)[1]
            elif False:
                next_token = temperature_sampling(x, 1.2, 5)
            else:
                next_token = topk_sampling(x, 5)
            input_tensor = torch.cat([input_tensor[:, :], next_token], dim=1)
            a.append(custom_lm.decode(next_token))
            if ind % 100 == 0:
                print("saving {}".format(ind))
                with open("fl.txt", "w") as fl:
                    print(" ".join(a), file=fl)
    except Exception as e:
        print(e)
        print("Abort lenght {}".format(len(a)))
    try:
        decode_midi(a, file_path="final.mid")
        decode_midi(ss, file_path="primer2.mid")
    except:
        utils.write_midi(a, None, "top5_hi/{}.mid".format(i), None)
