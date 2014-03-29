
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <mkl.h>
#include <mkl_lapack.h>

#include "statmh.hpp"
#include "statdistros.hpp"
#include "mhexamples.hpp"

Example1Target::Example1Target ()
    : TargetDistribution(2), MVG(2, 0)
{
    static double m[2] = {-7.0, 11.0};
    static double c[4] = { 4.0, -1.0, -1.0,  2.0 };
    MVG.SetCovariance(c);
    MVG.SetMean(m);
}

double Example1Target::pdf(int n, const double *pts, double *vals, int pftag) const
 {
    assert((n==1) || (vals!=0));
    double ret;
    if(vals==0) vals = &ret;
    MVG.sample_pdf(n, pts, vals);
    return vals[0];
}

double Example1Target::logpdf(int n, const double *pts, double *vals, int pftag) const
{
    assert((n==1) || (vals!=0));
    double ret;
    if(vals==0) vals = &ret;
    MVG.sample_logpdf(n, pts, vals);
    return vals[0];
}



extern "C"
void rwmh_Example1(int *n, const double *start, double *chain, double *logpi, int *accepted)
{
    rwmh_Example<RWMHChain, Example1Target>(n, start, chain, logpi, accepted);
}

extern "C"
void rwmh_prefetch_Example1(int *n, const double *start, double *chain, double *logpi, int *accepted)
{
    rwmh_Example<PrefetchRWMHChain, Example1Target>(n, start, chain, logpi, accepted);
}


/***************************************************************************/


double Example2Target::pdf(int n, const double *pts, double *vals, int pftag) const
{
    assert((n==1) || (vals!=0));
    double ret;
    if(vals==0) vals = &ret;
    M.sample_pdf(n, pts, vals);
    return vals[0];
}

double Example2Target::logpdf(int n, const double *pts, double *vals, int pftag) const
{
    assert((n==1) || (vals!=0));
    double ret;
    if(vals==0) vals = &ret;
    M.sample_logpdf(n, pts, vals);
    return vals[0];
}

const double Example2Target::prob[2] = { 0.8, 0.2 };
const double Example2Target::means[6] = {
        -3.0, 0.0, 5.0,
        -1.0, 2.0, 5.0
};
const double Example2Target::covs[18] = {
        1.0, -0.13, 0.7,
       -0.13, 1.0, 0.0,
         0.7, 0.0, 1.0,
         4.0, 0.0, -1.5,
         0.0, 4.0, 0.0,
         -1.5, 0.0, 2.0
};

extern "C"
void rwmh_Example2(int *n, const double *start, double *chain, double *logpi, int *accepted)
{
    rwmh_Example<RWMHChain, Example2Target>(n, start, chain, logpi, accepted);
}

extern "C"
void rwmh_prefetch_Example2(int *n, const double *start, double *chain, double *logpi, int *accepted)
{
    rwmh_Example<PrefetchRWMHChain, Example2Target>(n, start, chain, logpi, accepted);
}


/***************************************************************************/

void Example3Target::work_for_delay(int p) const
 {
    MyRngStream S(p);
    static const int work_size = 1000;
    double *A = new double[work_size*work_size];
    int *ipiv = new int[work_size];
    int info;
    for(int i=0; i<work_size*work_size; ++i)
        A[i] = S.uRand();
    dgetrf(&work_size, &work_size, A, &work_size, ipiv, &info);
    delete[] ipiv;
    delete[] A;
}

Example3Target::Example3Target () : TargetDistribution(dim_factor*3)
{
    const int i_1 = 1;
    double prob[2] = { 0.8, 0.2}; 
    double *means = new double[dim*2];
    double *covs = new double[dim*dim*2];
    memset(means, 0, dim*2*sizeof(double));
    memset(covs, 0, dim*dim*2*sizeof(double));
    for(int p=0; p<2; ++p) {
        double *m = means + p*dim;
        double *c = covs + p*dim*dim;
        const double *m2 = Example2Target::means+p*3;
        const double *c2 = Example2Target::covs+p*3*3; 
        for(int f=0; f<dim_factor; ++f) {
            for(int i=0; i<3; ++i)
                m[f*3+i] = m2[i];
            for(int i=0; i<3; ++i)
                for(int j=0; j<3; ++j)
                /** row is f*3+i and column is f*3+j */
                /** each column has dim rows */
                    c[dim*(f*3+j)+f*3+i] = c2[3*j+i];
        }
    }
    M = new MixtureMVN(2, prob, dim, means, covs, 0 );
    delete[] means;
    delete[] covs;
}
    
double Example3Target::pdf(int n, const double *pts, double *vals, int pftag) const
{
    assert((n==1) || (vals!=0));
    double ret;
    if(vals==0) vals = &ret;
    M->sample_pdf(n, pts, vals);
    work_for_delay(pftag);
    return vals[0];
}
    
double Example3Target::logpdf(int n, const double *pts, double *vals, int pftag) const
{
    assert((n==1) || (vals!=0));
    double ret;
    if(vals==0) vals = &ret;
    M->sample_logpdf(n, pts, vals);
    work_for_delay(pftag);
    return vals[0];
}

extern "C"
void rwmh_Example3(int *n, const double *start, double *chain, double *logpi, int *accepted)
{
    rwmh_Example<RWMHChain, Example3Target>(n, start, chain, logpi, accepted);
}

extern "C"
void rwmh_prefetch_Example3(int *n, const double *start, double *chain, double *logpi, int *accepted)
{
    rwmh_Example<PrefetchRWMHChain, Example3Target>(n, start, chain, logpi, accepted);
}


