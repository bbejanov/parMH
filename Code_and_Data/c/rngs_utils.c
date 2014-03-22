
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <math.h>
#include <assert.h>
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

void mnrand0(const int *which_gar, int *n, int *vals, 
        const int *num_bins, const double *prob_bins)
{
    double cs[num_bins[0]];
    int b, i, j;
    
    RngStream gar;
    /** get a stream **/
    gar = get_gar(which_gar[0]);
    if (gar == NULL) { 
        n[0]=0; 
        return;
    }
    assert(prob_bins[0] >= 0.0);
    cs[0] = prob_bins[0];
    for(i=1; i<num_bins[0]; ++i) {
        assert(prob_bins[i] >= 0.0);
        cs[i] = cs[i-1] + prob_bins[i];
    }
    if ( fabs(cs[num_bins[0]-1] - 1.0) > 1e-14 ) {
        for(i=0; i<num_bins[0]; ++i) {
            cs[i] /= cs[num_bins[0]-1];
        }
    }
    for(i=0 ; i<n[0]; ++i) {
        double U = RngStream_RandU01(gar);
        for(j=0; U > cs[j]; ++j) 
            {}
        vals[i] = j;
    }
    return;    
}

void mnrand(const int *which_gar, int *n, int *vals, 
        const int *num_bins, const double *prob_bins)
{
    int i;
    mnrand0(which_gar, n, vals, num_bins, prob_bins);
    for(i=0; i<n[0]; ++i) ++vals[i];
}
