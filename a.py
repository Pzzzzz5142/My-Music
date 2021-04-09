from tqdm import tqdm
from random import randint

with open("data/mae_remi.test.tokens", "r") as fl:
    total = 2
    thing = []
    cnt = 0
    for line in fl.readlines():
        line = line.strip()
        cnt += 1
        ff=open("fl.txt",'w')
        print(line,file=ff)
        break
    print(cnt)

for i in ["test", "train", "valid"]:
    with open("data/extend_2/mae.{}.tokens".format(i), "r") as fl, open(
        "data/df-token/mae.{}.tokens".format(i), "r"
    ) as ff, open("data/mixed/mixed.{}.tokens".format(i), "w") as tt:
        for line in fl.readlines():
            tt.write(line)
        for line in ff.readlines():
            tt.write(line)
