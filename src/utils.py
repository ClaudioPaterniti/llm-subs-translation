from typing import Generator
from math import ceil

def balanced_partition(tot: int, n: int = 1, max: int = None) -> Generator[slice, None, None]:
    blocks = ceil(tot/max) if max else n
    q, r = divmod(tot, blocks)
    for i in range(blocks):
        yield slice(i*q + min(i, r), (i+1)*q + min(i+1, r))