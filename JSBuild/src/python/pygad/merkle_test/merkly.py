from merkly import MerkleTree   
from typing import Callable

#choose any hash function that is of type (bytes, bytes) -> bytes
my_hash_function: Callable[[bytes, bytes], bytes] = lambda x, y: x + y

# create a Merkle Tree
mtree = MerkleTree(['a', 'b', 'c', 'd'], my_hash_function)

# show original input
assert mtree.raw_leaves == ['a', 'b', 'c', 'd']

# hashed leaves
assert mtree.leaves == [b'a', b'b', b'c', b'd']

# shorted hashed leaves
assert mtree.short_leaves == [b'a', b'b', b'c', b'd']
