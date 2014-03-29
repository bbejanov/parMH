#ifndef __MHEXAMPLES_HPP
#define __MHEXAMPLES_HPP

#include "statmh.hpp"
#include "statdistros.hpp"

struct Example1Target : public TargetDistribution
{
    MultivariateGaussian MVG;
    Example1Target (); 
    virtual double pdf(int n, const double *pts, double *vals=NULL, int pftag=1) const;
    virtual double logpdf(int n, const double *pts, double *vals=NULL, int pftag=1) const;
};

struct Example2Target : public TargetDistribution
{
    MixtureMVN M;
    static const double prob[2];
    static const double means[6];
    static const double covs[18];
    Example2Target () : TargetDistribution(3),
        M(2, prob, 3, means, covs, 0 ) {}
    virtual double pdf(int n, const double *pts, double *vals=NULL, int pftag=1) const; 
    virtual double logpdf(int n, const double *pts, double *vals=NULL, int pftag=1) const; 
};

struct Example3Target : public TargetDistribution
{
public:

    friend int main(int, char **);

    MixtureMVN *M;
    static const int dim_factor = 5;
    Example3Target (); 
    virtual ~Example3Target() {  delete M; }
    virtual double pdf(int n, const double *pts, double *vals=NULL, int pftag=1) const;
    virtual double logpdf(int n, const double *pts, double *vals=NULL, int pftag=1) const;
private:
    void work_for_delay(int p) const;
};

/***********************************************************************/


template <class RWCLASS, class EXAMPLETARGET> 
void rwmh_Example(int *n, const double *start, double *chain, double *logpi, int *accepted)
{
    RWCLASS RWMH;
    RWMH.PI = new EXAMPLETARGET;
    RWMH.dim = RWMH.PI->dim;
    RWMH.Q = new RandomWalkProposalDistribution(RWMH.PI->dim);
    RWMH.run(*n, start, chain, logpi, accepted);
    delete RWMH.PI;
    delete RWMH.Q;
}

extern "C" {

void rwmh_Example1(int *n, const double *start, double *chain, double *logpi, int *accepted);
void rwmh_Example2(int *n, const double *start, double *chain, double *logpi, int *accepted);
void rwmh_Example3(int *n, const double *start, double *chain, double *logpi, int *accepted);

void rwmh_prefetch_Example1(int *n, const double *start, double *chain, double *logpi, int *accepted);
void rwmh_prefetch_Example2(int *n, const double *start, double *chain, double *logpi, int *accepted);
void rwmh_prefetch_Example3(int *n, const double *start, double *chain, double *logpi, int *accepted);

}


#endif
