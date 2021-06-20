from typing import Optional
from pynvml import *
from fastapi import FastAPI
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.models.transformer_autoencoders import TransformerAutoencoders
import torch
import numpy as np
import torch.nn.functional as F
from random import randint
from midi_preprocess import encode_midi, decode_midi
import utils
import base64

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


def compose(dev: int):
    custom_lm = (
        TransformerLanguageModel.from_pretrained(
            "/mnt/zhangyi/checkpoints/transformer_music_lm_remi_cov",
            "checkpoint_best.pt",
        )
        .cuda(dev)
        .half()
        .eval()
    )
    model = custom_lm.models[0]
    l = 1024
    a = []
    ss = "<time_shift,0>"
    input_sequence = custom_lm.encode(ss)[:-1]
    input_tensor = torch.LongTensor(input_sequence).cuda().unsqueeze(0)
    # prev_input_tensor = torch.LongTensor(prev_input_seq).cuda().unsqueeze(0)
    print("ok")
    a = custom_lm.decode(torch.LongTensor(input_sequence).cuda()).split()
    print(input_sequence.shape)
    try:
        for ind in range(len(input_sequence), l):
            x = model(input_tensor[-2000:, :])[0]
            x = F.softmax(x, dim=2)[:, -1, :]
            next_token = topk_sampling(x, 5)
            input_tensor = torch.cat([input_tensor[:, :], next_token], dim=1)
            a.append(custom_lm.decode(next_token))
            if ind % 100 == 0:
                print("saving {}".format(ind))
                with open("fl.txt", "w") as fl:
                    print(" ".join(a), file=fl)
    except Exception as e:
        print("Abort lenght {}".format(len(a)))
    utils.write_midi(a, None, "tmp.mid", None)
    custom_lm.cpu()
    del custom_lm
    with open("tmp.mid", "rb") as fl:
        return base64.b64encode(fl.read())


app = FastAPI()


@app.get("/")
def read_root():
    dev = 0
    if dev == -1:
        return {"success": False, "error": "No enough memory"}
    try:
        thing = compose(dev)
        return {"success": True, "song": str(thing, encoding="utf-8")}
    except:
        return {"success": False, "error": "Compose Error"}