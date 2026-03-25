from merkly.mtree import MerkleTree   
from typing import Callable

#choose any hash function that is of type (bytes, bytes) -> bytes
my_hash_function: Callable[[bytes, bytes], bytes] = lambda x, y: x + y

# create a Merkle Tree
mtree = MerkleTree(['a', 'b', 'c', 'd'], my_hash_function)

print(mtree.raw_leaves)
# show original input
assert mtree.raw_leaves == ['a', 'b', 'c', 'd']

print(mtree.leaves)
# hashed leaves
assert mtree.leaves == [b'a', b'b', b'c', b'd']

print(mtree.short_leaves)
# shorted hashed leaves
assert mtree.short_leaves == [b'a', b'b', b'c', b'd']

with open("mtree.txt", "w+") as f:
    f.write (f"{mtree.leaves}")