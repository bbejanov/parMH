#ifndef __MH_H
#define __MH_H

typedef void (*target_logpdf_t)(const int *n, const int *dim, 
    const double *points, double *lp, void *tdata);


void rwmh_chain(const int *n, const int *d, const double *start, 
    target_logpdf_t t_lpdf, void *tdata,
    const double *prop_var,
    double *chain, int *accepted);
        

/** The example_1 is a bivariate normal distribution with
 *    mean = [ -7,  11 ]
 *    covariance = [ 4 -1; -1 2 ]
 **/
void target_example_1(const int *n, const int *dim, 
    const double *points, double *lp, void *tdata);
void rand_example_1(const int *wgar, int *n, double *points,
    const int *dim, void *tdata);

/** The example_2 is a mixture of two bivariate normals with
 * the following
 * mu_0 = [ -3, 0 ]  cov_0 = [ 1 0; 0 1]  with prob = 0.8
 * mu_1 = [  1, 4 ]  cov_1 = [ 4 0; 0 4]  with prob = 0.2
 **/
void target_example_2(const int *n, const int *dim, 
    const double *points, double *lp, void *tdata);
void rand_example_2(const int *wgar, int *n, double *points,
    const int *dim, void *tdata);

#endif
