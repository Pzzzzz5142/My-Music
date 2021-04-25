from tqdm import tqdm
from random import randint

class A(object):
    def __init__(self):
        self.a=self.aa()
    def aa(self):
        return 'aaa'

class B(A):
    def __init__(self):
        super().__init__()
    def aa(self):
        return 'aaaaaa'


a=A()
print(a.a)