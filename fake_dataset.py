from random import randint
from multiprocessing import Pool


class MusicEvent(object):
    def __init__(self) -> None:
        super().__init__()
        self.val: list = getattr(self, "val", [])
        self.range_limit = getattr(self, "range_limit", [[0, 128]])
        self.num = getattr(self, "num", 1)
        self.sep = getattr(self, "sep", ",")
        self.random_init()

    def __str__(self) -> str:
        return "<{}{}{}>".format(
            self.__class__.__name__,
            "," if len(self.val) != 0 else "",
            self.sep.join([str(i) for i in self.val]),
        )

    def __repr__(self) -> str:
        return str(self)

    def random_init(self):
        self.legal_check()

        self.val = [
            randint(self.range_limit[i][0], self.range_limit[i][1])
            for i in range(self.num)
        ]

    def legal_check(self):
        assert self.num == len(
            self.range_limit
        ), "The size of val is different with the size of range_limit"

    def from_token(self, token: str):
        _, val = token[1:-1].split(",")
        self.val = [int(i) for i in val.split(self.sep)]


class NoteOn(MusicEvent):
    def __init__(self) -> None:
        super().__init__()


class NoteDuration(MusicEvent):
    def __init__(self) -> None:
        self.range_limit = [[1, 128]]
        super().__init__()


class Bar(MusicEvent):
    def __init__(self) -> None:
        self.num = 0
        self.range_limit = []
        super().__init__()


class Beats(MusicEvent):
    def __init__(self) -> None:
        self.sep = "/"
        self.num = 2
        self.range_limit = [[1, 5], [1, 512]]
        super().__init__()


def gen_single_worker(num):
    lines = []
    length = 512
    for _ in range(num):
        line = [Beats()]
        cur_tm = line[0].val[1]
        while len(line) < length:
            ii = randint(0, 5)
            if ii < 3:
                if len(line) == length - 1:
                    continue
                line.append(str(NoteOn()))
                Dur = NoteDuration()
                Dur.range_limit[0][1] = min(cur_tm, Dur.range_limit[0][1])
                Dur.random_init()
                cur_tm -= Dur.val[0]
                line.append(Dur)
                if cur_tm == 0:
                    cur_tm = line[0].val[1]
                    if len(line) < length:
                        line.append(Bar())
            else:
                line.append(MusicEvent())
        lines.append(" ".join([str(i) for i in line]))
    return lines


def Generate_fake_dataset():
    lines = []
    pre_name = "fake_one_beat_512.tokens."
    workers = 16
    with open(pre_name + "train", "w") as train, open(
        pre_name + "test", "w"
    ) as test, open(pre_name + "valid", "w") as valid:
        ...
    train = open(pre_name + "train", "a")
    test = open(pre_name + "test", "a")
    valid = open(pre_name + "valid", "a")
    datasets = [[train, 900], [test, 100], [valid, 100]]
    pool = Pool(workers)

    def postprocessing(worker_result):
        num = min(datasets[0][1], len(worker_result))
        datasets[0][1] -= num
        print("\n".join(worker_result[:num]) + "\n", file=datasets[0][0])
        if datasets[0][1] == 0:
            print("搞完一个")
            datasets.pop(0)
            print("\n".join(worker_result[num:]) + "\n", file=datasets[0][0])

    dispatch = (1100 + (workers - 1)) // workers
    for i in range(workers):
        pool.apply_async(
            gen_single_worker,
            (dispatch if i != workers - 1 else 1100 % dispatch,),
            callback=postprocessing,
        )
    pool.close()
    pool.join()


def check_bar():
    with open("fl.txt", "r") as fl:
        line = fl.read().strip().split(" ")
        Dur = 0
        seq = []
        for i in line:
            if "Dur" in i:
                _, tm = i[1:-1].split(",")
                Dur += int(tm)
            elif "Ba" in i:
                seq.append(Dur)
                Dur = 0
            elif "Bea" in i:
                seq.append(i)
        print(seq)


if __name__ == "__main__":
    Generate_fake_dataset()
    check_bar()

