
#include <cstdlib>
#include <cstring>

#include <iostream>

#include "my_stat.h"
#include "statdistros.hpp"

void set_random_seed_from_time()
{
    MyRngStream::RngS_set_random_seed();
}

void how_many_streams(int *count)
{
    count[0] = MyRngStream::CountStreams();
}

void urand(const int *which_gar, int *n, double *vals) 
{
    MyRngStream G(*which_gar);
    for(int i=0; i<n[0]; ++i) 
        vals[i] = G.uRand(); 
    return;
} 

void catrand(const int *which_gar, int *n, int *vals, const int *k, const double *p)
{
    Categorical C(k[0], p, which_gar[0]);
    C.sample(n[0], vals);
}

void catrand1(const int *which_gar, int *n, int *vals, const int *k, const double *p)
{
    Categorical C(k[0], p, which_gar[0]);
    for(int i=0; i<n[0]; ++i)
        vals[i] = 1+C.rand();
}

/*****  Univariate Gaussian *******/

void n_rand(const int *which_gar, int *n, double *rvals, 
               const double *mean, const double *var)
{
    Gaussian G(mean[0], sqrt(var[0]), which_gar[0]);
    G.sample(n[0], rvals);
}
void nstd_rand(const int *which_gar, int *n, double *rvals)
{
    Gaussian G(which_gar[0]);
    G.sample(n[0], rvals);
}

void    n_logpdf(const int *n, const double *x, double *logpdf, 
                const double *mean, const double *var)
{
    Gaussian G(mean[0], sqrt(var[0]), 0);
    G.sample_logpdf(n[0], x, logpdf);
}
void nstd_logpdf(const int *n, const double *x, double *logpdf)
{
    Gaussian G(0);
    G.sample_logpdf(n[0], x, logpdf);
}

void    n_pdf(const int *n, const double *x, double *pdf, 
                const double *mean, const double *var)
{
    Gaussian G(mean[0], sqrt(var[0]), 0);
    G.sample_pdf(n[0], x, pdf);
}
void nstd_pdf(const int *n, const double *x, double *pdf)
{
    Gaussian G(0);
    G.sample_pdf(n[0], x, pdf);
}

/*****  Multivariate Gaussian *******/

void mvn_sqrt(const int *dim, const double *var, double *sqrt_var)
{
    MultivariateGaussian MVG(dim[0], 0);
    memcpy(sqrt_var, var, dim[0]*dim[0]*sizeof(double));
    MVG.SqrtMatrix(sqrt_var);
}

void mvn_rand(const int *which_gar, int *n, double *rvals, 
        const int *dim, const double *mu, const double *sigma)
{
    MultivariateGaussian MVG(dim[0], mu, sigma, which_gar[0]);
    MVG.sample(n[0], rvals);
}
        
void mvn_logpdf(const int *n, const double *x, double *logpdf,
        const int *dim, const double *mu, const double *sigma)
{
    MultivariateGaussian MVG(dim[0], mu, sigma, 0);
    MVG.sample_logpdf(n[0], x, logpdf);
}
        
void mvn_pdf(const int *n, const double *x, double *pdf,
        const int *dim, const double *mu, const double *sigma)
{
    MultivariateGaussian MVG(dim[0], mu, sigma, 0);
    MVG.sample_pdf(n[0], x, pdf);
}

/****** Mixture of Multivariate Gaussian *********/

void mvnm_rand(const int *which_gar, int *n, double *rvals,
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *covs)
{
    MixtureMVN A(num_modes[0], weights, dim[0], means, covs, which_gar[0]);
    A.sample(n[0], rvals);
}
        
void mvnm_pdf(const int *n, const double *x, double *pdf, 
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *covs)
{
    MixtureMVN A(num_modes[0], weights, dim[0], means, covs);
    A.sample_pdf(n[0], x, pdf);
}

void mvnm_logpdf(const int *n, const double *x, double *logpdf, 
        const int *num_modes, const double *weights, 
        const int *dim, const double *means, const double *covs)
{
    MixtureMVN A(num_modes[0], weights, dim[0], means, covs);
    A.sample_logpdf(n[0], x, logpdf);
}

/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/




