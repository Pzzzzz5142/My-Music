import requests
import os.path as path
import os
from multiprocessing import Pool
from bs4 import BeautifulSoup

url = r"https://thwiki.cc/分类:官方MIDI"
uu = r"https://thwiki.cc"

destdir = "data/df"
workers = 20


class FailMessage(object):
    def __init__(self, name, reason) -> None:
        super().__init__()
        self.name = name
        self.reason = reason

    def __repr__(self) -> str:
        return "Failed file name: {}, Reason: {}".format(self.name, self.reason)

    def __str__(self) -> str:
        return "Failed file name: {}, Reason: {}".format(self.name, self.reason)


def down_single_worker(ind: int, ls: list) -> str:
    print(
        "Worker {} start working. {} files waiting to be downloaded. ".format(
            ind, len(ls)
        )
    )
    succ = 0
    fail = []

    for item in ls:
        try:
            re = requests.get(item)
            x = BeautifulSoup(re.text, "lxml")
            x = x.find("a", class_="internal")["href"]
            re = requests.get(x)
            x = path.basename(x)
            with open(os.path.join(destdir, x), "wb") as fl:
                fl.write(re.content)
            succ += 1
        except Exception as e:
            fail.append(FailMessage(item["href"], str(e)))
    print(
        "Worker {} finished. {} files successed. {} files failes. ".format(
            ind, succ, len(ls) - succ
        )
    )
    if len(ls) - succ > 0:
        print("\n".join(fail))
    return "Worker {} finished. {} files successed. {} files failes. ".format(
        ind, succ, len(ls) - succ
    )


def solve() -> str:
    os.makedirs(destdir, exist_ok=True)

    re = requests.get(url)
    a = BeautifulSoup(re.text, "lxml")
    a = a.find_all("a", {"class": "galleryfilename galleryfilename-truncate"})
    pool = Pool(workers)
    per = (len(a) + workers - 1) // workers
    print("{} files in total. Allocated to {} workers.".format(len(a), workers))

    for i in range(0, len(a), per):
        pool.apply_async(
            down_single_worker,
            (i // per, [uu + item["href"] for item in a[i : i + per]]),
        )

    pool.close()
    pool.join()


if __name__ == "__main__":
    solve()

