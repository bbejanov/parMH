#ifndef __STATMH_H
#define __STATMH_H

#include <cstdlib>
#include <stdexcept>

#include "statdistros.hpp"

struct TargetDistribution
{
    int dim;
    virtual double    pdf(int n, const double *pts, double *vals=NULL) =0;
    virtual double logpdf(int n, const double *pts, double *vals=NULL) =0;
    explicit TargetDistribution(int d=0) : dim(d) {}
};

struct ProposalDistribution
{
    int dim;
    explicit ProposalDistribution(int d=0) : dim(d) {}
    virtual double urand() =0;
    // current is always a single point,
    // proposed will return n points in n-by-dim matrix
    // val must be valid for n!=1
    virtual double    pdf(int n, const double *current, const double *proposed, double *val=NULL) =0;
    virtual double logpdf(int n, const double *current, const double *proposed, double *val=NULL) =0;
    virtual void   sample(int n, const double *current,       double *proposed) =0;
};

struct RandomWalkProposalDistribution
        : ProposalDistribution, MultivariateGaussian
{
    explicit RandomWalkProposalDistribution(int d, int Idx=0) :
        ProposalDistribution(d), 
        MultivariateGaussian(d, Idx)
        { }
        
    double urand() { return G.uRand(); }
    void   sample(int n, const double *current,       double *proposed) {
        MultivariateGaussian::SetMean(current);
        MultivariateGaussian::sample(n, proposed);
    }
    double    pdf(int n, const double *current, const double *proposed, double *vals=NULL) {
        assert((n==1) || (vals!=0));
        double ret;
        if(vals==0) vals = &ret;
        MultivariateGaussian::SetMean(current);
        MultivariateGaussian::sample_pdf(n, proposed, vals);
        return vals[0];
    }
    double logpdf(int n, const double *current, const double *proposed, double *vals=NULL) {
        assert((n==1) || (vals!=0));
        double ret;
        if(vals==0) vals = &ret;
        MultivariateGaussian::SetMean(current);
        MultivariateGaussian::sample_logpdf(n, proposed, vals);
        return vals[0];
   }
};
    

struct MHChain
{
    int dim;
    int have;
    double *chain;
    int *accepted;
    TargetDistribution    *PI;
    ProposalDistribution  *Q;

    MHChain()
        : dim(1), have(0), chain(0), accepted(0), PI(0), Q(0) {};
    // run may allocate memory, so we must clean it
    ~MHChain()
        { if (have>0) { delete[]chain; delete[] accepted; } }

    // the last two arguments allow user to give us memory
    virtual void run(int n, const double *start, double *c=NULL, int *a=NULL)
        =0;
        
protected:
    // if overloading, make sure to call the parent's
    virtual void check_run_args
            (int n, const double *start, double *c=NULL, int *a=NULL)
            throw(std::logic_error, std::invalid_argument) ;
};

struct RWMHChain : public MHChain
{
    void run(int n, const double *start, double *c=NULL, int *a=NULL);
};

#endif
