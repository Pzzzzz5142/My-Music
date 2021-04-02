with open("data-bin/new/dict.txt", "r") as fl:
    ti = [i for i in range(0, 100)]
    oo = [i for i in range(0, 128)]
    of = [i for i in range(0, 128)]
    for line in fl.readlines():
        line, _ = line.strip().split(" ")
        pre, tm = line.strip()[1:-1].split(",")
        tm = int(tm)
        if "time" in line:
            ti.remove(tm)
        elif "on" in line:
            oo.remove(tm)
        elif "off" in line:
            of.remove(tm)

    with open("tt.tt", "w") as ff:
        oo = ["<note_on,{}> 0".format(i) for i in oo]
        of = ["<note_off,{}> 0".format(i) for i in of]
        print("\n".join(oo))
        print("\n".join(of))

