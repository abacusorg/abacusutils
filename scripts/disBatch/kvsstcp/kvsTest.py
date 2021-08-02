#!/usr/bin/env python2
import os, sys
from kvsclient import KVSClient

# Connect to the key value store service (its location is the first
# argument).
kvs = KVSClient(sys.argv[1] if len(sys.argv) > 1 else None)

# Find out which process we are (assumes running under SLURM).
rank = int(os.environ['SLURM_PROCID'])

# Example of a function that dynamically and disjointly decomposes the
# work. All participants make the same sequence of calls to this
# function. One and only one participant will see True on the Nth
# invocation, the rest will see False.
#
# "chunk" controls the coarseness of the decomposition.
def own(rank, chunk=1):
    # initialize a global counter.
    if 0 == rank: kvs.put('count', 0)
    checks, lower, upper = 0, -1, -1
    while 1:
        if checks >= upper:
            lower = kvs.get('count')
            upper = lower + chunk
            kvs.put('count', upper)
        yield lower <= checks < upper
        checks += 1

# Here we illustrate splitting a simple loop, but the same approach
# would work with any iterative control structure, as long as it is
# deterministic.
ownCheck = own(rank, chunk=300013)
s = 0
limit = 100000000
for x in xrange(limit):
    if ownCheck.next(): s += 2*x + 1
# Report partial result.
kvs.put('psum', (rank, s))

# One participant gathers the partial results and generates the final
# output.
if 0 == rank:
    s, workers = 0, int(os.environ['SLURM_NTASKS'])
    for p in xrange(workers):
        wrank, ps = kvs.get('psum')
        print('rank %d: %d'%(wrank, ps))
        s += ps
    print('total = %d (%r)' % (s, (limit*limit) == s))
    sys.exit((limit*limit) != s)
