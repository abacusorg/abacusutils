
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "kvsSupport.h"

char payload[2000000];

#define Loops 10000
int
main(int argc, char *argv[])
{
  int			cinfo[3];
  int			l;
  char			myKey[100];
  int			payloadSize = 1;
  double		t0, t1;
  struct timeval	t;
    
  cinfo[2] = 0;
  kvsconnect(cinfo);

  sprintf(myKey, "data%10d", getpid()); // Cheap way to separate
					// multiple runs using the
					// same server.

  for (;payloadSize < (1<<20)+1; payloadSize <<= 1) {
    gettimeofday(&t, 0);
    t0 = t.tv_sec + t.tv_usec/1000000.;
    for (l = 0; l < Loops; ++l) {
      int midpoint = payloadSize>>1;
      payload[midpoint] = 13;
      kvsput(cinfo, myKey, payload, payloadSize);
      payload[midpoint] = 0;
      kvsget(cinfo, myKey, payload);
      assert(payload[midpoint] == 13);
    }
    gettimeofday(&t, 0);
    t1 = t.tv_sec + t.tv_usec/1000000.;
    printf("%20d%20.2e\n", payloadSize, (t1-t0)/Loops);
  }
  return 0;
}

