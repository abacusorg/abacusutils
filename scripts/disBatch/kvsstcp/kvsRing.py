#!/usr/bin/python2
from __future__ import print_function

import os, time, sys
from kvsclient import KVSClient

kvs = KVSClient(os.getenv('KVSSTCP_HOST'), os.getenv('KVSSTCP_PORT'))
rank, pop = int(os.getenv('SLURM_PROCID')), int(os.getenv('SLURM_NTASKS'))

myTag = 'data%10d'%rank
pTag = 'data%10d'%((rank+1)%pop)

Loops = 100
for payloadSize in [1<<x for x in range(21)]:
    p = 'a'*(payloadSize)
    t0 = time.time()
    for l in range(Loops):
        kvs.put(myTag, p,  False)
        p = kvs.get(pTag, False)
    t1 = time.time()
    delta = t1 - t0
    print("%20d%20.2e%20.2e%20.2e"%(payloadSize, delta/Loops, pop*Loops*2/delta, pop*payloadSize*Loops*2/delta))

print(kvs.dump())


