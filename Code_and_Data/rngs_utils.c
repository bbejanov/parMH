
#include <stdlib.h>
#include <stdarg.h>
#include "rngs_utils.h"

/** define global integer constrant */
enum { 
    MAX_NUM_GARS = 100         /** maximum number of streams */
};

static RngStream gars[MAX_NUM_GARS];   /** an array of all streams */
static int max_valid_gar = -1;         /** highest index of a valid stream */

/** creates a new stream for generating double precision values */
static RngStream init_stream() 
{
    RngStream g;
    g = RngStream_CreateStream("");
    /** increase precision of stream for generating double precision */
    RngStream_IncreasedPrecis(g, 1);
    return g;
}

/** returns the stream of the given index (zero-based) */
/** creates the stream if necessary */
RngStream get_gar(int w) 
{
    if (w < 0) {
        printf("Invalid stream index (must be > 0)\n");
        return NULL;
    }
    if (w >= MAX_NUM_GARS) {
        printf("Too many streams (max allowed = %d)\n", MAX_NUM_GARS);
        return NULL;
    }
    while (w > max_valid_gar) {
        ++max_valid_gar;
        gars[max_valid_gar] = init_stream();
    }

    return gars[w];
}

void check_alloc(int num, ...)
{
    int i;
    void *ptr;
    va_list ptr_list;
    va_start(ptr_list, num);
    for(i=0; i<num; ++i) {
        ptr = va_arg(ptr_list, void *);
        if( ptr == NULL ) 
            goto error_alloc;
    }
    va_end(ptr_list);
    return;
error_alloc:
    printf("Failed to allocate memory.\n");
    exit(EXIT_FAILURE);  /* Unreachable */
    return;
}

void urand(const int *which_gar, int *n, double *vals) 
{
    RngStream gar;
    int i;

    /** get a stream **/
    gar = get_gar(which_gar[0]);
    if (gar == NULL) { 
        n[0]=0; 
        return;
    }

    for(i=0 ; i<n[0]; ++i) 
      vals[i] = RngStream_RandU01(gar);

   return;
} 

void rngs_set_random_seed()
{
    unsigned long seed[6] = {0, 0, 0, 0, 0, 0};
    srand(time(NULL));
    while( seed[0] + seed[1] + seed[2] == 0 ) {
        seed[0] = rand() % 4294967087;
        seed[1] = rand() % 4294967087;
        seed[2] = rand() % 4294967087;
    }
    while( seed[3] + seed[4] + seed[5] == 0 ) {
        seed[3] = rand() % 4294944443;
        seed[4] = rand() % 4294944443;
        seed[5] = rand() % 4294944443;
    }
    RngStream_SetPackageSeed(seed);
}

