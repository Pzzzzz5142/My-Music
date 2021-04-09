import torch
from torch.tensor import Tensor

A = torch.zeros(3, 4, 6)
B = torch.rand(3, 4, 6)

indA = torch.LongTensor([[0, 0], [0, 1], [1, 1]])
indB = torch.LongTensor([[1, 1], [2, 1], [2, 2]])


def indices_copy(A: Tensor, B, indA, indB, inplace=True):
    # To make sure our views below are valid
    assert A.is_contiguous()
    assert B.is_contiguous()

    # Get the size
    size = A.size()

    # Collapse the first two dimensions, so that we index only one
    vA = A.view(size[0] * size[1], size[2])
    vB = B.view(size[0] * size[1], size[2])

    # If we need out of place, clone to get a tensor backed by new memory
    if not inplace:
        vA = vA.clone()

    # Transform the 2D indices into 1D indices in our collapsed dimension
    lin_indA = indA.select(1, 0) * size[1] + indA.select(1, 1)
    lin_indB = indB.select(1, 0) * size[1] + indB.select(1, 1)

    # Read B and write in A
    vA.index_copy_(0, lin_indA, vB.index_select(0, lin_indB))

    return vA.view(size)


print("Inputs")
print(A)
print(B)
print(indA)
print(indB)

indices_copy(A, B, indA, indB)

print("Output inplace")
print(A)


A = torch.zeros(3, 4, 6)
new_A = indices_copy(A, B, indA, indB, inplace=False)

print("Output out of place")
print(new_A)
print("Unmodified A")
print(A)
