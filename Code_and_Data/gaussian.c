
#include <RngStream.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mkl_lapack.h>


#ifndef DEBUG
#define DEBUG (0!=0)
#endif

enum { MAX_NUM_GARS = 100};

static RngStream gars[MAX_NUM_GARS];
static int num_gars = -1;

RngStream init_stream() 
{
    RngStream g;
    g = RngStream_CreateStream("");
    RngStream_IncreasedPrecis(g, 1);
    return g;
}

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
    while (w > num_gars) {
        ++num_gars;
        gars[num_gars] = init_stream();
    }

    return gars[w];
}

void rstdnorm(int *which_gar, int *n, double *rvals) 
{
    RngStream gar;
    double u, v, s;
    int i;

    gar = get_gar(which_gar[0]);
    if (gar == NULL) { 
        n[0]=0; 
        return;
    }

    for(i=0; i<n[0];) {
        u = RngStream_RandU01(gar)*2.0 - 1.0;
        v = RngStream_RandU01(gar)*2.0 - 1.0;
        s = u*u + v*v;
        if ((s == 0.0) || (s >= 1.0)) 
            continue;
        s = sqrt( -2.0*log(s)/s );
        rvals[i] = u*s;
        rvals[i+1] = v*s;
        i += 2;
    }
}

void rnorm(int *which_gar, int *n, double *rvals, double *mu, double *sigma)
{
    int i;
    rstdnorm(which_gar, n, rvals);
    for(i=0; i<n[0]; ++i) 
        rvals[i] = mu[0] + rvals[i] * sigma[0];
}



void chol(int *n, double *mat) 
{
    int info;
    dpotrf("U", n, mat, n, &info);

    if ( info < 0 ) {
        printf("Argument %i is illegal\n", -info);
    } else if (info > 0) {
        printf("The %i leading minor is not positive definite\n", info);
    } else {
        int i;
        for(i=0; i<n[0]; ++i) 
            memset(mat + i*n[0]*sizeof(double) + i + 1, 0, (n[0]-i-1)*sizeof(double));
    }
}


