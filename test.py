import unittest
from midi_preprocess import main
import argparse
import os
from datetime import datetime


class Test(unittest.TestCase):
    def get_args(self, args: list):
        parser = argparse.ArgumentParser("midi preprocess parser")

        parser.add_argument(
            "--destdir", metavar="DIR", default="./", help="destination dir"
        )
        parser.add_argument(
            "--other", action="store_true", default=True, help="using maestro or not"
        )
        parser.add_argument("--datadir", metavar="DIR", default="../remi/data/train", help="data dir")
        parser.add_argument("--workers", type=int, default="20")
        parser.add_argument("--prefix", type=str)
        return parser.parse_args(args)

    def test_multi(self):
        pre1 = "a"
        pre2 = "b"
        args1 = self.get_args(["--workers", "1", "--prefix", pre1])
        args2 = self.get_args(["--workers", "20", "--prefix", pre2])

        main(args1)
        main(args2)

        ls = ["test", "train", "valid"]

        content1 = []
        content2 = []

        for sp in ls:
            fl1 = open(f"{pre1}.{sp}.tokens", "r")
            fl2 = open(f"{pre2}.{sp}.tokens", "r")

            content1 += fl1.read().strip().split("\n")
            content2 += fl2.read().strip().split("\n")

            self.assertEqual(len(content1), len(content2))

            fl1.close()
            fl2.close()

            os.remove(f"{pre1}.{sp}.tokens")
            os.remove(f"{pre2}.{sp}.tokens")

        self.assertCountEqual(content1, content2)

    def test_split(self):
        args = self.get_args(["--workers", "20", "--prefix", "a"])

        main(args)

        ls = ["test", "train", "valid"]

        content = []

        for sp in ls:
            with open(f"a.{sp}.tokens", "r") as fl:
                content.append(len(fl.read().strip().split("\n")))

            os.remove(f"a.{sp}.tokens")

        self.assertAlmostEqual(content[1] / content[0], 8, 0)
        self.assertAlmostEqual(content[1] / content[2], 8, 0)

    def test_speed(self):
        pre1 = "a"
        pre2 = "b"
        args1 = self.get_args(
            ["--workers", "20", "--prefix", pre1, "--datadir", "../remi/data/train"]
        )
        args2 = self.get_args(
            ["--workers", "1", "--prefix", pre2, "--datadir", "../remi/data/train"]
        )

        now = datetime.now()
        main(args1)
        t1 = datetime.now() - now
        now = datetime.now()
        main(args2)
        t2 = datetime.now() - now

        print("Multi-processing spent time:", t1)
        print("Single thread spent time:", t2)

        ls = ["test", "train", "valid"]

        for sp in ls:
            os.remove(f"{pre1}.{sp}.tokens")
            os.remove(f"{pre2}.{sp}.tokens")

        self.assertLess(t1, t2)


if __name__ == "__main__":
    unittest.main()
