#include <assert.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#define EVALSTR(x) STR(x)
#define STR(x)  #x
#define LENLEN	10
#define LENFMT "%"EVALSTR(LENLEN)"d"

#define TRACE 1

/* Connect to a KVS server at the given host and port.
 * If either is 0, it is taken from the KVSSTCP_HOST and KVSSTCP_PORT
 * environment variables.
 * Returns a non-negative connection id (file descriptor) on success,
 * or -1 on error (and sets errno).
 */
int kvs_connect(char *host, int port)
{
  char *sport;
  char portbuf[8];

  if (!host)
    host = getenv("KVSSTCP_HOST");
  if (port > 0) {
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    sport = portbuf;
  } else
    sport = getenv("KVSSTCP_PORT");

  struct addrinfo hints = {
    .ai_family = AF_UNSPEC,
    .ai_flags = AI_NUMERICSERV,
    .ai_socktype = SOCK_STREAM,
  };
  struct addrinfo *ai = NULL;
  if (!host || !sport || getaddrinfo(host, sport, &hints, &ai)) {
    errno = ENOENT;
    return -1;
  }

  int s = -1;
  struct addrinfo *aip;
  for (aip = ai; aip; aip = aip->ai_next) {
    s = socket(aip->ai_family, aip->ai_socktype, aip->ai_protocol);
    if (s < 0)
      continue;
    const int one = 1;
    if (setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
      close(s);
      s = -1;
      continue;
    }
    if (connect(s, aip->ai_addr, aip->ai_addrlen) < 0) {
      close(s);
      s = -1;
      continue;
    }
  }

  if (ai)
    freeaddrinfo(ai);
  return s;
}

// This file includes both C and FORTRAN interfaces. It assumes the
// convention that a FORTRAN string is represented in an argument list
// twice: first by a pointer to the bytes of the strings, and second
// by an integer giving the length of the string. The second is
// appended to the list of existing arguments.

// TODO: WHAT IS THE DEFAULT SIZE OF AN INTEGER CONSTANT IN FORTRAN?
// MORE GENERALLY, HOW DO WE ENSURE THE TYPE OF LENGTH PARAMETERS?

// cinfo is an opaque type holding connection info. currently:
// cinfo[0]: socket fd
// cinfo[1]: unused
// cinfo[2]: flags. TRACE (bit 0) traces get/view and put operations.

// Warpper that handles partial readv or writev.
// SIDE EFFECTING!!!
static void
dovec(ssize_t (*func)(), int fd, struct iovec *iov, int iovcnt, ssize_t bytes)
{
  int	vx = 0;

  while (1) {
    ssize_t	done = func(fd, iov+vx, iovcnt-vx);
    if (done <= 0) {
      char	*name = "readv";

      if (func == writev) name = "writev";
      fprintf(stderr, "%s:%d %s failed with %zu bytes left (errno: %d)\n", __FILE__, __LINE__, name, bytes, errno);
      exit(-1);
    }

    if (done == bytes) return;
    bytes -= done;
    for (;vx < iovcnt; ++vx) {
      if (done < iov[vx].iov_len) {
	iov[vx].iov_base += done;
	iov[vx].iov_len -= done;
	break;
      }
      done -= iov[vx].iov_len;
    }
  }
}

static void
gv(char* op, int *cinfo, char *k, void *v, int key_len)
{
  char		encoding[4];
  struct iovec	parts[3];
  int		plen;
  char		sbuf[LENLEN+1];
  int		sfd = cinfo[0];

  if (key_len && (k[0] == '@')) {
    // strip ascii designator---not really needed for get/view anyway.
    --key_len;
    ++k;
  }

  // Send op and key.
  sprintf(sbuf, LENFMT, key_len);
  parts[0].iov_base = op;
  parts[0].iov_len = 4;
  parts[1].iov_base = sbuf;
  parts[1].iov_len = LENLEN;
  parts[2].iov_base = k;
  parts[2].iov_len = key_len;
  plen = 4 + LENLEN + key_len;
  dovec(writev, sfd, parts, 3, plen);
  if (cinfo[2] & TRACE) fprintf(stderr, "%s req: key %.*s, addr %p\n", op, key_len, k, v);

  // Receive reply.
  parts[0].iov_base = encoding;
  parts[0].iov_len = 4;
  parts[1].iov_base = sbuf;
  parts[1].iov_len = LENLEN;
  plen = 4 + LENLEN;
  dovec(readv, sfd, parts, 2, plen);
  sbuf[LENLEN] = 0;
  // TODO: using plain int limits payload to < 2^31 bytes
  plen = atoi(sbuf);

  // Get value.
  parts[0].iov_base = v;
  parts[0].iov_len = plen;
  dovec(readv, sfd, parts, 1, plen);
  if (cinfo[2] & TRACE) fprintf(stderr, "%s rep: key %.*s;  %d bytes\n", op, key_len, k, plen);
}

void
kvsconnect_(int *cinfo)
{
  int s = kvs_connect(0, 0);
  if (s < 0) {
    fprintf(stderr, "Failed to connect to KVS server: %m.\n");
    exit(1);
  }
  *cinfo = s;
}

void
kvsconnect(int *cinfo)
{
  kvsconnect_(cinfo);
}

void 
kvsget_(int *cinfo, char *k, void *v, int key_len)
{
  gv("get_", cinfo, k, v, key_len);
}
void 
kvsget(int *cinfo, char *k, void *v)
{
  int	key_len = (int)strlen(k);
  gv("get_", cinfo, k, v, key_len);
}

void
kvsput_(int *cinfo, char *k, void *v, int *nbytes, int key_len)
{
  char	*encoding = "BINY";
  struct iovec	parts[6];
  int	plen;
  char	sbuf[2][LENLEN+1];
  int	sfd = cinfo[0];

  if (key_len && (k[0] == '@')) {
    // strip ascii designator.
    --key_len;
    ++k;
    encoding = "ASTR";
  }
  // send op, key length, key, encoding, value length
  sprintf(sbuf[0], LENFMT, key_len);
  sprintf(sbuf[1], LENFMT, *nbytes);
  parts[0].iov_base = "put_";
  parts[0].iov_len = 4;
  parts[1].iov_base = sbuf[0];
  parts[1].iov_len = LENLEN;
  parts[2].iov_base = k;
  parts[2].iov_len = key_len;
  parts[3].iov_base = encoding;
  parts[3].iov_len = 4;
  parts[4].iov_base = sbuf[1];
  parts[4].iov_len = LENLEN;
  parts[5].iov_base = v;
  parts[5].iov_len = *nbytes;
  plen = 4 + LENLEN + key_len + 4 + LENLEN + *nbytes;
  dovec(writev, sfd, parts, 6, plen);
  if (cinfo[2] & TRACE) fprintf(stderr, "put: key %.*s; addr %p; %d bytes\n", key_len, k, v, *nbytes);
}
void
kvsput(int *cinfo, char *k, void *v, int nbytes)
{
  int	key_len = (int)strlen(k);
  kvsput_(cinfo, k, v, &nbytes, key_len);
}

void 
kvsview_(int *cinfo, char *k, void *v, int key_len)
{
  gv("view", cinfo, k, v, key_len);
}
void 
kvsview(int *cinfo, char *k, void *v)
{
  int	key_len = (int)strlen(k);
  gv("view", cinfo, k, v, key_len);
}

////////// Utilities

#define MaxKeyLen	1000

// Barrier design note: by special casing "(?? == rank)" the edge case
// of a single worker will use no coordination operations.
void
kvsabarw_(int *cinfo, int *rank, int *nclones, char *key, int *it, int key_len)
{
  // Asymmetric barrier: clone identified by rank "it" waits until all
  // other clones have executed abarrierp for the given key.
  char	buf[MaxKeyLen+10+1];
  int	dummy, i;

  if (*it != *rank) return;
  
  if (key_len > MaxKeyLen) {
    fprintf(stderr, "Implementation restriction: barrier key must be no more than %d bytes.\n", MaxKeyLen);
    exit(-1);
  }

  for (i = 0; *nclones-1; ++i) {
    if (i == *rank) continue;
    sprintf(buf, "%s%10d", key, i);
    kvsget(cinfo, buf, &dummy);
  }
}
void
kvsabarw(int *cinfo, int rank, int nclones, char *key, int it)
{
  int	key_len = (int)strlen(key);
  kvsabarw_(cinfo, &rank, &nclones, key, &it, key_len);
}

void
kvsabarp_(int *cinfo, int *rank, int *nclones, char *key, int *it, int key_len)
{
  char	buf[MaxKeyLen+10+1];

  if (*it == *rank) return;

  if (key_len > MaxKeyLen) {
    fprintf(stderr, "Implementation restriction: barrier key must be no more than %d bytes.\n", MaxKeyLen);
    exit(1);
  }

  sprintf(buf, "%s%10d", key, *rank);
  kvsput(cinfo, buf, &it, sizeof(int));
}
void
kvsabarp(int *cinfo, int rank, int nclones, char *key, int it)
{
  int	key_len = (int)strlen(key);
  kvsabarp_(cinfo, &rank, &nclones, key, &it, key_len);
}

int
kvscheckown_(int *cinfo, int *rank, int *nclones, int *chunk, char *ticketKey, int ticketKey_len)
{
  static int high = -1, low, serializer = 0;
  int intsize = sizeof(int), result, ticket;

  if (*rank==0 && serializer==0) {
    ticket = 0;
    kvsput_(cinfo, ticketKey, &ticket, &intsize, ticketKey_len);
  }

  if (serializer >= high) {
    kvsget_(cinfo, ticketKey, &ticket, ticketKey_len);
    low = ticket;
    high = low + *chunk;
    ticket = high;
    kvsput_(cinfo, ticketKey, &ticket, &intsize, ticketKey_len);
  }
  result = (low <= serializer) && (serializer < high);
  serializer += 1;
  return result;
}
int
kvscheckown(int *cinfo, int rank, int nclones, int chunk, char *ticketKey)
{
  int	ticketKey_len = (int)strlen(ticketKey);
  return kvscheckown_(cinfo, &rank, &nclones, &chunk, ticketKey, ticketKey_len);
}

void
kvsgentasks_(int *cinfo, int *rank, int *nclones, char *key, int *b, int *e, int *s, int *p, int key_len)
{
  // Generate a series of task values bound to the given key. Append
  // to the end of the list a number of poison pills sufficient to end
  // processing on all clones.
  int	i, intsize=sizeof(int);

  if (*rank != 0 ) return;
  for (i = *b; *e; i += *s) {
    kvsput_(cinfo, key, &i, &intsize, key_len);
  }

  for (i = 0; i < *nclones; ++i) {
    kvsput_(cinfo, key, p, &intsize, key_len);
  }
}
void
kvsgentasks(int *cinfo, int rank, int nclones, char *key, int b, int e, int s, int p)
{
  int	key_len = (int)strlen(key);
  kvsgentasks_(cinfo, &rank, &nclones, key, &b, &e, &s, &p, key_len);
}

void
slurmme_(int *rank, int *nclones)
{
  char	*cp;
  // Determine rank and number of clones for a SLURM job.
  // Batch queueing systems typically use environment variables to set
  // a given processes rank and to indicate how many processes overall
  // are involved in a run. Code very similar to this would likely work
  // for LSF, PBS, SGE, etc.
  cp = getenv("SLURM_PROCID");
  if (!cp) {
    fprintf(stderr, "Environment variable \"SLURM_PROCID\" is not set.\n");
    exit(-1);
  }
  *rank = atoi(cp);
  cp = getenv("SLURM_NTASKS");
  if (!cp) {
    fprintf(stderr, "Environment variable \"SLURM_NTASKS\" is not set.\n");
    exit(-1);
  }
  *nclones = atoi(cp);
}
void
slurmme(int *rank, int *nclones)
{
  slurmme_(rank, nclones);
}
