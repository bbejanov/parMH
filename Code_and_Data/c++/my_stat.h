#ifndef __MY_STAT_H
#define __MY_STAT_H

#ifdef __cplusplus
extern "C" {
#endif

/** all functions here are defined with the restrictions of R */

void set_random_seed_from_time();
void how_many_streams(int *count);

void urand(const int *gar, int *n, double *pts);
void catrand(const int *gar, int *n, int *vals, const int *k, const double *p);
void catrand1(const int *gar, int *n, int *vals, const int *k, const double *p);

/* Gaussian normal distribution */
/** naming:
 *   xyz_rand(const int *w, const int *n, double *rvals, ...) 
 *         generates n random values using stream w and returns them in rvals
 *         n returns the number of actually generated values (n=0 means error)
 * 
 *   xyz_logpdf(const int *n, const double *x, double *...)
 *         computes the natural log of the p.d.f.
 *   
 *   xyz_pdf(const int *n, const double *x, double *...)
 *         computes the p.d.f.
 *   xyz_logpdf(const int *n, const double *x, double *...)
 *         computes the natural log of the p.d.f.
 * 
 * 
 */

void    n_rand(const int *which_gar, int *n, double *rvals, 
               const double *mean, const double *var);
void nstd_rand(const int *which_gar, int *n, double *rvals);

void    n_logpdf(const int *n, const double *x, double *logpdf, 
                const double *mean, const double *var);
void nstd_logpdf(const int *n, const double *x, double *logpdf);

void    n_pdf(const int *n, const double *x, double *pdf, 
                const double *mean, const double *var);
void nstd_pdf(const int *n, const double *x, double *pdf);



/* The multivariate normal distribution **/
/** NOTES:
 *    - all matrices are stored column-major, i.e. the Fortran way 
 *    - deviates are row-vectors, e.g. mvn_rand returns in rvals a
 *       matrix with n rows and dim columns, where each row is a random point
 *    - mean must be a vector of length dim
 *    - the correlation structure is given by the square-root of the variance 
 *      matrix. This can be obtained by calling mvn_sqrt.
 *    - var must be a dim-by-dim s.p.d. matrix, sqrt_var is the 
 *      upper-triangular Cholesky decomposition, s.t. 
 *           transpose(sqrt_var) * sqrt_var = var
 *    - we do not allocate/deallocate memory, user must manage their own ram
 * 
 **/

/** works correctly when sqrt_var==var, replacing the matrix with its chol */
void mvn_sqrt(const int *dim, const double *var, double *sqrt_var);

void mvn_rand(const int *which_gar, int *n, double *rvals, 
        const int *dim, const double *mu, const double *sigma);
void mvn_logpdf(const int *n, const double *x, double *logpdf,
        const int *dim, const double *mu, const double *sigma);
void mvn_pdf(const int *n, const double *x, double *pdf,
        const int *dim, const double *mu, const double *sigma);

/* The multivariate normal mixture */

void mvnm_rand(const int *which_gar, int *n, double *rvals,
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *covs);
void mvnm_pdf(const int *n, const double *x, double *pdf, 
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *covs);
void mvnm_logpdf(const int *n, const double *x, double *logpdf, 
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *covs);


/** lets run some rwmh examples **/

void rwmh_Example1(int *n, const double *start, double *chain, int *accepted);
void rwmh_Example2(int *n, const double *start, double *chain, int *accepted);

void rwmh_prefetch_Example1(int *n, int *h, const double *start, double *chain, int *accepted);
void rwmh_prefetch_Example2(int *n, int *h, const double *start, double *chain, int *accepted);


#ifdef __cplusplus
}
#endif


#endif
