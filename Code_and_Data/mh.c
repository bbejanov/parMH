
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <mkl.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>

#include "mh.h"
#include "rngs_utils.h"
#include "gaussian.h"

static const int ione = 1;
static const int wgar = 0;

#define MKL_MALLOC(x)   mkl_malloc(x, 64)
#define MKL_FREE(x)     mkl_free(x)

/** The Random-Walk proposal 
 *     Generates a new point from a multi-variate normal distribution 
 *     with mean the current point and given variance 
 **/ 
void rw_prop_rand(
    const int *which_gar,
    int *n,
    const int *dim, const double *current, double *new, 
    const double *sqrt_var)
{
    mvn_rand(which_gar, n, new, dim, current, sqrt_var);
}

void target_example_1(const int *n, const int *dim, 
    const double *points, double *lp, void *tdata)
{    
    double sqv[4];
    const double mean[2] = {-7.0, 11.0};
    const double var[4] = { 4.0, -1.0, 
                           -1.0,  2.0 };
    assert(dim[0]==2);
    mvn_sqrt(dim, var, sqv);
    mvn_logpdf(n, points, lp, dim, mean, sqv);
}


void rwmh_chain(const int *n, const int *dim, const double *start, 
    target_logpdf_t t_lpdf, void *tdata,
    const double *prop_var,
    double *chain, 
    int *accepted)
{
    double *prop_sqv = (double *) MKL_MALLOC(dim[0]*dim[0]*sizeof(double));
    assert(prop_sqv != NULL);
    mvn_sqrt(dim, prop_var, prop_sqv);
    
    /** Nomenclature
     *    current - current point of the chain
     *    lp_current - the logpdf of the target at the current point
     *    proposed - proposed point
     *    lp_proposed - the logpdf of the proposed
     **/
    
    double *current = (double *) MKL_MALLOC(dim[0]*sizeof(double));
    assert(current != NULL);
    double lp_current;
    
    double *proposed = (double *) MKL_MALLOC(dim[0]*sizeof(double));
    assert(proposed != NULL);
    double lp_proposed;
    
    #define T_LPDF(p,lp) (*t_lpdf)(&ione, dim, p, &lp, tdata)
    #define P_RAND(c,n) {                                   \
        int nn=1;                                           \
        rw_prop_rand(&wgar, &nn, dim, c, n, prop_sqv);      \
        assert(nn>0);                                       \
    }
    #define U_RAND(v) {         \
        int nn=1;               \
        urand(&wgar, &nn, &v);  \
        assert(nn>0);           \
    }

    int i, nn;
    double U; 
 
    memcpy(current, start, dim[0]*sizeof(double));
    T_LPDF(current, lp_current);
    for(i=0; i<n[0]; ++i) {
        P_RAND(current, proposed);
        T_LPDF(proposed, lp_proposed);
        U_RAND(U);
        if( log(U) < lp_proposed - lp_current ) {
            accepted[i] = 1;
            dcopy(dim, proposed, &ione, current, &ione);
            lp_current = lp_proposed;
        } else {
            accepted[i] = 0;
        }
        dcopy(dim, current, &ione, chain+i, n);
    }
    
    MKL_FREE(proposed);
    MKL_FREE(current);
    MKL_FREE(prop_sqv);
    return;    
}










