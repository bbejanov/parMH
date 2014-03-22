
#include <cstdlib>
#include <cassert>
#include <cstring>

#include "statmh.hpp"
#include "statdistros.hpp"

struct Example1Target
        : public TargetDistribution
{
    double mean[2];
    double cov[4];

    Example1Target () {
        dim = 2;
        static double m[2] = {-7.0, 11.0};
        static double c[4] = { 4.0, -1.0, -1.0,  2.0 };
        memcpy(mean, m, dim*sizeof(double));
        memcpy(cov, c, dim*dim*sizeof(double));
    }

    double pdf(int n, const double *pts, double *vals=NULL) {
        assert((n==1) || (vals!=0));
        double ret;
        if(vals==0) vals = &ret;
        MultivariateGaussian MVG(dim, mean, cov, 0);
        MVG.sample_pdf(n, pts, vals);
        return vals[0];
    }
    
    double logpdf(int n, const double *pts, double *vals=NULL) {
        assert((n==1) || (vals!=0));
        double ret;
        if(vals==0) vals = &ret;
        MultivariateGaussian MVG(dim, mean, cov, 0);
        MVG.sample_logpdf(n, pts, vals);
        return vals[0];
    }
};

extern "C"
    void rwmh_Example1(int n, const double *start, double *chain, int *accepted);

void rwmh_Example1(int n, const double *start, double *chain, int *accepted)
{
    RWMHChain RWMH;
    RWMH.PI = new Example1Target;
    RWMH.dim = RWMH.PI->dim;
    RWMH.Q = new RandomWalkProposalDistribution(RWMH.PI->dim);
    RWMH.run(n, start, chain, accepted);
    delete RWMH.PI;
    delete RWMH.Q;
}

