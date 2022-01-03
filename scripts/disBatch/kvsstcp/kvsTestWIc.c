
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "kvsSupport.h"

int
main(int argc, char *argv[])
{
  int		cinfo[3], rank, nclones, chunk, limit, psum[3];
  int   	i, sum = 0, work = 0;
  time_t	t0;
  char		cchunk[10], climit[10], html[1000];

  // Disable kvs tracing.
  cinfo[2] = 0;
  // Connect to the key/value service.
  kvsconnect(cinfo);

  // Figure out our role in this run.
  slurmme(&rank, &nclones);
  printf("Rank %d of %d.\n", rank, nclones);
      
  // Get run parameters.
  if (rank == 0) {
    // Request input via KVS from a web inteface.
    kvsget(cinfo, "Loop limit?%10s", climit); //TODO: Check null handling.
    limit = atoi(climit);
    kvsget(cinfo, "Chunk?%10s", cchunk);
    chunk = atoi(cchunk);
    // Send some HTML to the web inteface.
    sprintf(html, "<html><h2>Looping %.10s times, in chunks of size %.10s.</h2>", climit, cchunk);
    kvsput(cinfo, "@Run info", html, strlen(html));
    //       Let the clones know.
    kvsput(cinfo, "limit", &limit, sizeof(limit));
    kvsput(cinfo, "chunk", &chunk, sizeof(chunk));
  }
  else {
    kvsview(cinfo, "limit", &limit);
    kvsview(cinfo, "chunk", &chunk);
  }

  // The work in this loop will be split among all participants into
  // blocks of size "chunk".
  t0 = time(0);
  for (i = 0; i < limit; ++i) {
    if (kvscheckown(cinfo, rank, nclones, chunk, "ticket")) {
      sum += 2*i + 1;
      work += 1;
      // Arrange for some clones to be slower than others.
      sleep((rank%2)+1);
    }
  }
  psum[0] = rank;
  psum[1] = work;
  psum[2] = sum;

  // Merge partial results.
  kvsput(cinfo, "psum", psum, sizeof(psum));

  if (rank == 0 ) {
    sum = 0;
    for (i = 0; i < nclones; ++i) {
      kvsget(cinfo, "psum", psum);
      printf("%d %d => %d\n", psum[0], psum[1], psum[2]);
      sum += psum[2];
    }
    printf("Sum of the first %d odd integers: %d\n", limit, sum);
    printf("Elapsed: %d\n", time(0) - t0);
  }
  return 0;
}

