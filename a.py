from tqdm import tqdm
from random import randint

with open("data/mae.train.tokens", "r") as fl:
    total = 2
    thing = [0, 0, 0]
    cnt=0
    for line in tqdm(fl.readlines()):
        i = randint(1, 100)
        if i<20:
            line=line.strip()
            with open("fl.txt",'w') as ff:
                print(line,file=ff)
                cnt=1
                break
    if cnt==1:
        print("Ok")    
