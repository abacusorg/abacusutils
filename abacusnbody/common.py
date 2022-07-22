'''Common utility functions.
'''

def maxthreads():
    '''Return the number of logical cores available to this process.
    First tries the affinity mask, then the total number of CPUs,
    then 1 if all else fails.
    '''
    import multiprocessing
    import os
    
    try:
        maxthreads = len(os.sched_getaffinity(0))
    except AttributeError:
        maxthreads = multiprocessing.cpu_count() or 1
        
    return maxthreads
