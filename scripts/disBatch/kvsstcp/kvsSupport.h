/* C API */
int kvs_connect(char *host, int port);

/* Fortran-compatible API */
void kvsconnect(int *cinfo);
void kvsget(int *cinfo, char *k, void *v);
void kvsput(int *cinfo, char *k, void *v, int nbytes);
void kvsview(int *cinfo, char *k, void *v);

////////// Utilities
void kvsabarw(int *cinfo, int rank, int nclones, char *key, int it);
void kvsabarp(int *cinfo, int rank, int nclones, char *key, int it);
int  kvscheckown(int *cinfo, int rank, int nclones, int chunk, char *ticketKey);
void kvsgentasks(int *cinfo, int rank, int nclones, char *key, int b, int e, int s, int p);
void slurmme(int *rank, int *nclones);
