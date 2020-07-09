/*

An example C program to use with pipe_asdf.py.  Reads "N" and "x_com"
over the pipe and prints the first and last few values to demonstrate
sanity.

Usage
-----
$ ./pipe_asdf.py halo_info_000.asdf -f N -f x_com | ./client

*/

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main(int argc, char *argv[]){

    uint32_t *N = NULL;
    float *x_com = NULL;

    int64_t n_N;
    int width;

    // Read metadata and data for N
    size_t nread;
    nread = fread(&n_N, sizeof(int64_t), 1, stdin);
    assert(nread == 1);
    nread = fread(&width, sizeof(int), 1, stdin);
    assert(nread == 1);
    assert(width == 4);

    N = malloc(n_N*width);
    assert(N != NULL);

    nread = fread(N, sizeof(uint32_t), n_N, stdin);
    assert(nread == n_N);

    int64_t n_xcom;
    // Read metadata and data for x_com
    nread = fread(&n_xcom, sizeof(int64_t), 1, stdin);
    assert(nread == 1);
    nread = fread(&width, sizeof(int), 1, stdin);
    assert(nread == 1);
    assert(width == 4);

    x_com = malloc((int64_t)n_xcom*width);
    assert(x_com != NULL);

    nread = fread(x_com, sizeof(float), n_xcom, stdin);
    assert(nread == n_xcom);

    printf("First and last 5 N:\n");
    for(int i = 0; i < 5; i++)
        printf("%u\n",N[i]);
    for(int i = 0; i < 5; i++)
        printf("%u\n",N[n_N-i-1]);

    printf("First and last 5 x_com:\n");
    for(int i = 0; i < 5; i++)
        printf("(%f,%f,%f)\n",x_com[3*i],x_com[3*i + 1],x_com[3*i + 2]);
    for(int i = 0; i < 5; i++)
        printf("(%f,%f,%f)\n",x_com[n_xcom-1-3*i-2],x_com[n_xcom-1-3*i-1],x_com[n_xcom-1-3*i]);

    // Make sure no more data left on stream
    assert(fgetc(stdin) == EOF);

    free(N);
    free(x_com);

    return 0;
}
