
#include <assert.h>

#include <stdlib.h>
#include <string.h>   /** for memcpy */
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include <mkl.h>
#include <mkl_lapack.h>
#include <mkl_blas.h>

#ifndef DEBUG
#define DEBUG (0!=0)
#endif

#include "rngs_utils.h"

inline static double SQR(double x) { return x*x; }

static const int ione = 1;
static const double dzero=0.0;
static const double done=1.0;
static const double dmone=-1.0;

#define MKL_MALLOC(x)   mkl_malloc(x, 64)
#define MKL_FREE(x)     mkl_free(x)

static void box_muller_pair(RngStream gar, double *u, double *v)
{
    double s;
    do {
        u[0] = RngStream_RandU01(gar)*2.0 - 1.0;
        v[0] = RngStream_RandU01(gar)*2.0 - 1.0;
        s = u[0]*u[0] + v[0]*v[0];
    } while( (s>=1.0) || (s==0.0) );
    s = sqrt( -2.0*log(s)/s );
    u[0] *= s;
    v[0] *= s;
}

void nstd_rand(const int *which_gar, int *n, double *rvals) 
{
    RngStream gar = get_gar(which_gar[0]);
    if (gar == NULL) { 
        n[0]=0; 
        return;
    }
    /** the box-muller transformation produces a pair of normals */
    int i; // i must exist outside the loop
    for(i=0; 1+i<n[0]; i+=2) {
        box_muller_pair(gar, rvals+i, rvals+i+1);
    }
    if ( i < n[0] ) {
        /** n is odd, so we need one more */
        double dummy;
        box_muller_pair(gar, rvals+i, &dummy);
    }
}

void    n_rand(const int *which_gar, int *n, double *rvals, 
               const double *mean, const double *var)
{
    double sigma = sqrt(var[0]);
    nstd_rand(which_gar, n, rvals);
    for(int i=0; i<n[0]; ++i) 
        rvals[i] = mean[0] + rvals[i] * sigma;
}

void n_logpdf(const int *n, const double *x, double *logpdf, 
              const int *k, const double *mean, const double *var)
{
    /* 
     * pdf is 1/sqrt(2*pi*var)*exp(-(x-mean)^2/(2*var))
     * we return log of it
     */    
    double add, mul;
    if(k[0] == 1) {
        /* mean and var are single numbers that apply to all x */
        assert(var[0] > 0.0);
        add = -0.5 * log(2.0*M_PI*var[0]);
        mul = -0.5/var[0];
        if (mean[0] == 0.0)
            for(int i=0; i<n[0]; ++i)
                logpdf[i] = mul * SQR(x[i]) + add;
        else
            for(int i=0; i<n[0]; ++i)
                logpdf[i] = mul * SQR(x[i]-mean[0]) + add;
    } else {
        /* mean and var are vectors of the same length as x */ 
        assert(k[0]==n[0]);
        for(int i=0; i<n[0]; ++i) {
            assert(var[i]>0.0);
            add = -0.5 * log(2.0*M_PI*var[i]);
            mul = -0.5 / var[i];
            logpdf[i] = mul * SQR(x[i]-mean[i]) + add;
        }
    }
    return;
}

void nstd_logpdf(const int *n, const double *x, double *logpdf) {
    n_logpdf(n, x, logpdf, &ione, &dzero, &done);
}

void n_pdf(const int *n, const double *x, double *pdf, 
        const int *k, const double *mean, const double *var)
{
    n_logpdf(n,x,pdf,k,mean,var);
    for(int i=0; i<n[0]; ++i)
        pdf[i] = exp(pdf[i]);
}
              
void nstd_pdf(const int *n, const double *x, double *pdf) {
    n_pdf(n, x, pdf, &ione, &dzero, &done);
}


/***************************************************************************/
/** The multivariate normal distribution
 * 
 *  - we do not allocate/free memory; the caller must manage their own RAM 
 *****/

/** returns the memory location of the (i,j)-th element of n-by-m matrix 
 *  in column-major order, i.e. matrix is stored in memory like this
 *    (0,0) (1,0) ... (n-1,0) (0,1) ... (n-1,1) (0,2) ...
 **/
inline static int cm_index(int i, int j, int n) { return i + j*n; }

void mvn_sqrt(const int *dim, const double *var, double *sqrt_var) 
{
    int info;
    if ( var != sqrt_var ) {
        memcpy(sqrt_var, var, dim[0]*dim[0]*sizeof(double));
    }
    dpotrf("U", dim, sqrt_var, dim, &info);
    if ( info < 0 ) {
        printf("Argument %i is illegal\n", -info);
    } else if (info > 0) {
        printf("The %i leading minor is not positive definite\n", info);
    } else {
        /** set the lower-triangle to 0.0 **/
        int i, j;        
        for(i=0; i<dim[0]-1; ++i) {
            memset(sqrt_var+cm_index(i,i,dim[0])+1, 0, sizeof(double)*(dim[0]-i-1));
            /* for(j=i+1; j<dim[0]; ++j)
                sqrt_var[cm_index(j,i,dim[0])] = 0.0; */
        }
    }
}

 
void mvn_rand(const int *which_gar, int *n, double *rvals, 
        const int *dim, const double *mean, const double *sqrt_var)
{
    /** start by generating n*dim standard normals **/
    int ndim = n[0]*dim[0];   
    nstd_rand(which_gar, &ndim, rvals);
    if(ndim==0) { /* indicates error */
        n[0] = 0;
        return;
    }    
    /** adjust the mean and the covariance.  
     *  we need to do 
     *        rvals = mean + rvals * sqrt_var
     *  rvals is n-by-dim
     *  mean is 1-by-dim, so we need singleton expansion 
     *  sqrt_var is dim-by-dim
     **/
#if 1 
    dtrmm("R", "U", "N", "N", n, dim, &done, sqrt_var, dim, rvals, n);
    for(int j=0; j<dim[0]; ++j)
        for(int i=0; i<n[0]; ++i) 
            rvals[i+j*n[0]] += mean[j];
    return;
#else 
    /* one row at a time, little temp memory, multiple calls to blas */
    double *tmp_vec = (double*)MKL_MALLOC(dim[0]*sizeof(double));
    assert(tmp_vec != NULL);  /** dim is expected to be small (~200 max) */
    for(int i=0; i<n[0]; ++i) {
        /** this loop does (in Matlab notation) 
         *     tmp_vec = rvals(i,:)
         *     rvals(i,:) = mean
         **/
        dcopy(dim, rvals+i, n, tmp_vec, &ione);
        dcopy(dim, mean, &ione, rvals+i, n);
        /* for(j=0; j<dim[0]; ++j) {
            tmp_vec[j] = rvals[cm_index(i,j,n[0])];
            rvals[cm_index(i,j,n[0])] = mean[j];
        } */
        /** this blas call does (in Matlab notation) 
         *      rvals(i,:) = tmp_vec * sqrt_var + rvals(i,:)
         **/        
        dgemm("N", "N", &ione, dim, dim, &done, tmp_vec, &ione, sqrt_var, 
                dim, &done, rvals+cm_index(i,0,n[0]), n);
    }
    MKL_FREE(tmp_vec);
#endif

    return;
}


void mvn_logpdf(const int *n, const double *x, double *logpdf,
        const int *dim, const double *mean, const double *sqrt_var)
{
    double a = -0.5*log(2.0*M_PI)*dim[0];
    double b = 0.0;
    double c;
    for(int i=0; i<dim[0]; ++i) 
        b += log(sqrt_var[cm_index(i,i,dim[0])]);
    /* one row at a time, little temp memory, multiple calls to blas :-( */
    /**allocate memory */
    double *tmp_vec = (double*)MKL_MALLOC(dim[0]*sizeof(double));
    assert(tmp_vec != NULL);  
    for(int i=0; i<n[0]; ++i) 
    /** compute logpdf(i)=a-b-0.5*(x(i,:)-mean)*inv(var)*(x(i,:)-mean)' */
    {
        /** tmp_vec = x(i,:) - mean **/
        dcopy(dim, x+i, n, tmp_vec, &ione);
        daxpy(dim, &dmone, mean, &ione, tmp_vec, &ione);
       /** solve Y*sqrt_var=tmp_vec for Y
         * the answer goes back into tmp_vec, so there is no Y **/
        dtrsm("R", "U", "N", "N", &ione, dim, &done, sqrt_var, 
                dim, tmp_vec, &ione);
        c = ddot(dim, tmp_vec, &ione, tmp_vec, &ione);
        logpdf[i] = a-b-0.5*c;
    }
    MKL_FREE(tmp_vec);
    return;
}

void mvn_pdf(const int *n, const double *x, double *pdf,
        const int *dim, const double *mean, const double *sqrt_var)
{
    mvn_logpdf(n, x, pdf, dim, mean, sqrt_var);
    for(int i=0; i<n[0]; ++i)
        pdf[i] = exp(pdf[i]);
}



/* Themultivariate normal mixture, a.k.a. mixture of mvn */

void mvnm_normalize_weights(const int *num_modes,
        const double * weights, double *pr)
{
    /** normalize the weights to add up to unity */
    double aux = 0.0;
    for(int i=0; i<num_modes[0]; ++i) {
        pr[i] = weights[i];
        assert(pr[i] >= 0.0); /** weights may be zero, but not negative*/
        aux += pr[i];
    }
    assert(aux > 0.0);   /** at least one weight must be non-zero */
    for(int i=0; i<num_modes[0]; ++i) 
        pr[i] /= aux;
}            

void mvnm_pdf(const int *n, const double *x, double *pdf, 
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *sqrt_vars)
{
    if( n[0] <= 0 ) return;
    
    /** normalize the weights to add up to unity */
    double pr[num_modes[0]];
    mvnm_normalize_weights(num_modes, weights, pr);
    
    /** allocate a temporary array */
    double *tmp = (double*) MKL_MALLOC(n[0]*sizeof(double));
    assert( tmp != NULL );
    /** weighted sum of the individual multivariate normals */
    const int step_mean = dim[0];
    const int step_var = dim[0]*dim[0];
    memset(pdf, 0, n[0]*sizeof(double));
    for(int i=0; i<num_modes[0]; ++i) {
        mvn_pdf(n, x, tmp, dim, means, sqrt_vars);
        daxpy(n, pr+i, tmp, &ione, pdf, &ione);
        means += step_mean;
        sqrt_vars += step_var;
    }
    MKL_FREE(tmp);
    return;    
}

void mvnm_logpdf(const int *n, const double *x, double *logpdf, 
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *sqrt_vars)
{
    mvnm_pdf(n, x, logpdf, num_modes, weights, dim, means, sqrt_vars);
    for(int i=0; i<n[0]; ++i) {
        logpdf[i] = log(logpdf[i]);
    }
}

void mvnm_rand(const int *which_gar, int *n, double *rvals,
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *sqrt_vars)
{
    double pr[num_modes[0]];
    mvnm_normalize_weights(num_modes, weights, pr);

    const int step_mean = dim[0];
    const int step_var = dim[0]*dim[0];    
    for(int i=0; i<n[0]; ++i) {
        int bin, none=1;
        double tmp[dim[0]];
        mnrand0(which_gar, &none, &bin, num_modes, pr);
        if(none == 0) {
            n[0] = i;
            return;
        }
        mvn_rand(which_gar, &none, tmp,
            dim, means+bin*step_mean, sqrt_vars+bin*step_var);
        if (none == 0) {
            n[0] = i;
            return;
        }
        dcopy(dim, tmp, &ione, rvals+i, n);
    }
}


