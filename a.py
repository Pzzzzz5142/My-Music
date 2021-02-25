from tqdm import tqdm
from random import randint

with open("data/a.token", "r") as fl:
    total = 2
    thing = []
    cnt = 0
    for line in fl.readlines():
        line = line.strip().split()
        cnt += 1
        thing.append(len(line))
        if len(line) < 2048:
            print(len(line), cnt)
    print(cnt)

