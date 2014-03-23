
#include <cstdlib>
#include <cassert>
#include <cstring>

#include "statmh.hpp"
#include "statdistros.hpp"

struct Example1Target
        : public TargetDistribution
{
    MultivariateGaussian MVG;

    Example1Target ()
        : TargetDistribution(2), MVG(2, 0)
    {
        static double m[2] = {-7.0, 11.0};
        static double c[4] = { 4.0, -1.0, -1.0,  2.0 };
        MVG.SetCovariance(c);
        MVG.SetMean(m);
    }

    double pdf(int n, const double *pts, double *vals=NULL) {
        assert((n==1) || (vals!=0));
        double ret;
        if(vals==0) vals = &ret;
        MVG.sample_pdf(n, pts, vals);
        return vals[0];
    }
    
    double logpdf(int n, const double *pts, double *vals=NULL) {
        assert((n==1) || (vals!=0));
        double ret;
        if(vals==0) vals = &ret;
        MVG.sample_logpdf(n, pts, vals);
        return vals[0];
    }
};

template <class ET> 
void rwmh_Example(int *n, const double *start, double *chain, int *accepted)
{
    RWMHChain RWMH;
    RWMH.PI = new ET;
    RWMH.dim = RWMH.PI->dim;
    RWMH.Q = new RandomWalkProposalDistribution(RWMH.PI->dim);
    RWMH.run(*n, start, chain, accepted);
    delete RWMH.PI;
    delete RWMH.Q;
}
 
extern "C"
void rwmh_Example1(int *n, const double *start, double *chain, int *accepted)
{
    rwmh_Example<Example1Target>(n, start, chain, accepted);
}

/***************************************************************************/

struct Example2Target
        : public TargetDistribution
{
    MixtureMVN M;

    static const double prob[2];
    static const double means[6];
    static const double covs[18];

    Example2Target ()
        : TargetDistribution(3),
        M(2, prob, 3, means, covs, 0 )
    {}

    double pdf(int n, const double *pts, double *vals=NULL) {
        assert((n==1) || (vals!=0));
        double ret;
        if(vals==0) vals = &ret;
        M.sample_pdf(n, pts, vals);
        return vals[0];
    }
    
    double logpdf(int n, const double *pts, double *vals=NULL) {
        assert((n==1) || (vals!=0));
        double ret;
        if(vals==0) vals = &ret;
        M.sample_logpdf(n, pts, vals);
        return vals[0];
    }
};

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
void rwmh_Example2(int *n, const double *start, double *chain, int *accepted)
{
    rwmh_Example<Example2Target>(n, start, chain, accepted);
}





