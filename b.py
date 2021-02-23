for i in ["test", "train", "valid"]:
    with open("data/extend_2/mae.{}.tokens".format(i), "r") as fl, open(
        "data/df-token/mae.{}.tokens".format(i), "r"
    ) as ff, open("data/mixed/mixed.{}.tokens".format(i), "w") as tt:
        for line in fl.readlines():
            tt.write(line)
        for line in ff.readlines():
            tt.write(line)
