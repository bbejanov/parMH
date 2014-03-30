#ifndef __STATMH_H
#define __STATMH_H

#include <cstdlib>
#include <stdexcept>

#include "statdistros.hpp"

struct TargetDistribution
{
    int dim;
    virtual double    pdf(int n, const double *pts, double *vals=NULL, int pftag=1) const =0;
    virtual double logpdf(int n, const double *pts, double *vals=NULL, int pftag=1) const =0;
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
        { SetStandardMeanScaleCovariance(pow(0.1,2.0/d)); }

    double urand() { return G.uRand(); }
    void   sample(int n, const double *current,       double *proposed);
    double    pdf(int n, const double *current, const double *proposed, double *vals=NULL);
    double logpdf(int n, const double *current, const double *proposed, double *vals=NULL);
};


class MHChain
{
public:
    int dim;
    int have;
    double *chain;
    double *logpi;
    int *accepted;
    const TargetDistribution    *PI;
    ProposalDistribution  *Q;

    MHChain()
        : dim(1), have(0), chain(0), logpi(0), accepted(0), PI(0), Q(0) {};
    // run may allocate memory, so we must clean it
    ~MHChain()
        { if (have>0) { delete[]chain; delete[] logpi; delete[] accepted; } }

    // the last two arguments allow user to give us memory
    virtual void run(int n, const double *start,
                    double *c=NULL, double *l=NULL, int *a=NULL
        )=0;

protected:
    // if overloading, make sure to call the parent's
    virtual void check_run_args
            (int n, const double *start,
                double *c=NULL, double *l=NULL, int *a=NULL)
            throw(std::logic_error, std::invalid_argument) ;
};

class RWMHChain : public MHChain
{
public:
    RWMHChain() : MHChain() {}
    virtual void run(int n, const double *start,
                double *c=NULL, double *l=NULL, int *a=NULL);
};

class PrefetchRWMHChain : public RWMHChain
{
public:
    PrefetchRWMHChain();
    virtual ~PrefetchRWMHChain ();
    virtual void run(int n, const double *start,
            double *c=NULL, double *l=NULL, int *a=NULL);
            
    /********** prefetching stuff ***************/
    virtual void prefetch(const double *current, double logpi_current);
    void prefetch_set_alpha_const(double alpha);
    void prefetch_set_alpha_vector(const double *alpha);

    enum {
        FULL,       // build the full tree
        STATIC,     // acceptance probability is constant (set by user)
        DYNAMIC     // the realized unform values are taken into
                    // account, probabilities are estimated dynamically
    } pref_type;

    int      pref_h;            // maximum dept of tree
    int      pref_evals;        // maximum number of points to prefetch
    int     *pref_selected;
    double **pref_points;    
    double  *pref_logpi;
    double  *pref_prob;   /** probability of reaching node */
    double  *pref_alpha;  /** the acceptance probability */

    /** these variables are not needed for the algorithm, but for monitoring */
    int     *pref_at_step;      // how many steps did we actually make
    
protected:
    void prefetch_build_tree(const double *root);
    void prefetch_compute_target(double root_target);
    void prefetch_compute_probabilities();
    void prefetch_select_poitns();
    void prefetch_print_tree();
private: 
    void free_points();
    void alloc_points();
    int prefetch_find_best_point(int c);
};

class PrefetchRWMHChainOMP : public PrefetchRWMHChain
{
    virtual void prefetch(const double *current, double logpi_current);
private:
    void prefetch_compute_target_omp(double logpi_root);
};

class PrefetchRWMHChainCilk : public PrefetchRWMHChain
{
    virtual void prefetch(const double *current, double logpi_current);
private:
    void prefetch_compute_target_cilk(int c, int s);
};
    


#endif
