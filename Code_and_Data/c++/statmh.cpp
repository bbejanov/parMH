
#include <stdexcept>

#include <cmath>

#include <mkl.h>
#include <mkl_blas.h>

#include "statmh.hpp"

void   RandomWalkProposalDistribution::sample(int n, const double *current,
                    double *proposed)
{
    MultivariateGaussian::SetMean(current);
    MultivariateGaussian::sample(n, proposed);
}

double RandomWalkProposalDistribution::pdf(int n, const double *current,
                    const double *proposed, double *vals)
{
    assert((n==1) || (vals!=0));
    double ret;
    if(vals==0) vals = &ret;
    MultivariateGaussian::SetMean(current);
    MultivariateGaussian::sample_pdf(n, proposed, vals);
    return vals[0];
}

double RandomWalkProposalDistribution::logpdf(int n, const double *current,
                    const double *proposed, double *vals)
{
    assert((n==1) || (vals!=0));
    double ret;
    if(vals==0) vals = &ret;
    MultivariateGaussian::SetMean(current);
    MultivariateGaussian::sample_logpdf(n, proposed, vals);
    return vals[0];
}

/*************************************************************************/

void MHChain::check_run_args(int n, const double *start,
        double *c, double *l, int *a)
    throw(std::logic_error, std::invalid_argument)
{
    if((PI==0) || (Q==0)) {
        throw std::logic_error(
            "MHChain.run: must set target and proposals first");
    }
    if( (PI->dim!=dim) || (Q->dim!=dim) ) {
        throw std::invalid_argument(
            "MHChain.run: dimensions of chain, target and proposal must agree");
    }
    if(have > 0) {
        delete[] chain;
        delete[] logpi;
        delete[] accepted;
        have = 0;
    }
    if((c != 0) && (l != 0) && (a != 0)) {
        chain = c;
        logpi = l;
        accepted = a;
    } else if ((c==0) && (l==0) && (a==0)) {
        chain = new double[dim*n];
        logpi = new double[n];
        accepted = new int[n];
        have = n;
    } else {
        throw std::invalid_argument(
            "MHChain.run: must give either all of a, l and c or none of them");
    }
}

void RWMHChain::run(int n, const double *start, double *c, double *l, int *a)
{
    check_run_args(n, start, c, l, a);

    const int i_1 = 1;
    double current[dim], proposed[dim];
    dcopy(&dim, start, &i_1, current, &i_1);

    double lp_proposed = 0;
    double lp_current = PI->logpdf(1, current);

    for(int i=0; i<n; ++i) {
        Q->sample(1, current, proposed);
        lp_proposed = PI->logpdf(1, proposed);
        double U = Q->urand();
        if( log(U) < lp_proposed - lp_current ) {
            accepted[i] = 1;
            dcopy(&dim, proposed, &i_1, current, &i_1);
            lp_current = lp_proposed;
        } else {
            accepted[i] = 0;
        }
        dcopy(&dim, current, &i_1, chain+i, &n);
        logpi[i] = lp_current;
    }
}

/***************************************************************************/

PrefetchRWMHChain::PrefetchRWMHChain() : RWMHChain(),
        pref_points(0), pref_logpi(0), pref_at_step(0),
        pref_h(0), pref_type(FULL), pref_alpha(0), pref_prob(0),
        pref_evals(0), pref_selected(0)
        {}
        
PrefetchRWMHChain::~PrefetchRWMHChain () {
    free_points();
    if(pref_at_step!=0) delete[] pref_at_step;
    if(pref_alpha!=0) delete[] pref_alpha;
}


void PrefetchRWMHChain::free_points() {
    if(pref_points!=0) {
        delete[] pref_points[0];
        delete[] pref_points;
    }
    if(pref_logpi!=0)       delete[] pref_logpi;
    if(pref_prob!=0)        delete[] pref_prob;
    if(pref_selected!=0)    delete[] pref_selected;
}

void PrefetchRWMHChain::alloc_points() {
    if (pref_h>0) {
        int pow2n = 1 << pref_h;
        pref_points = (double **) ::operator new (pow2n*sizeof(double*));
        pref_points[0] = new double[dim*pow2n];
        for(int i=1; i<pow2n; ++i) 
            pref_points[i] = pref_points[i-1]+dim;
        pref_logpi = new double[pow2n];
        pref_prob  = new double[pow2n];
        pref_selected = new int[pow2n];
    }
}

void PrefetchRWMHChain::run(int n, const double *start, double *ch, double *l, int *a)
{
    if (pref_h < 2) {
        RWMHChain::run(n, start, ch, l, a);
        return;
    }

    check_run_args(n, start, ch, l, a);
    this->pref_at_step = new int[n];
    memset(this->pref_at_step, 0, n*sizeof(int));

    this->free_points();
    this->alloc_points();

    double lpi_c = PI->logpdf(1, start, 0, 1<<pref_h);
    int c; /* index of current point */
    int p; /* index of proposed point */
    int s; /* step within the current prefetching */

    if( (pref_evals <= 0) || (pref_evals >= (1<<pref_h)-1) ) {
        pref_evals = 0;
        pref_type = FULL;
    }

    if( pref_type == STATIC ) {
        prefetch_compute_probabilities();
        prefetch_select_poitns();
    }

    this->prefetch(start, lpi_c);
    s = 0;
    c = 0;
    for(int i=0; i<n; ++i) {
        s = s + 1;
        p = c + (1 << (s-1));
        if( (s > pref_h) || (pref_points[p] == NULL) ) {
            this->prefetch(pref_points[c], pref_logpi[c]);
            pref_at_step[i] = s-1;
            c = 0;
            s = 1;
            p = 1;
        }
        double U = Q->urand();
        if( log(U) < pref_logpi[p] - pref_logpi[c] ) {
            accepted[i] = 1;
            c = p;
        } else {
            accepted[i] = 0;
        }
        const int i_1 = 1;
        dcopy(&dim, pref_points[c], &i_1, chain+i, &n);
        logpi[i] = pref_logpi[c];
    }
}

void PrefetchRWMHChain::prefetch_compute_probabilities()
{
    pref_prob[1] = 1.0;
    int top_c = 1 << (pref_h-1);
    for(int c=1; c < top_c; ++c) {
        int s = log2(c);
        int acc = c + (1<<(s+1));
        int rej = acc - (1<<s);
        pref_prob[acc] = pref_prob[c] * pref_alpha[s];
        pref_prob[rej] = pref_prob[c] * (1.0-pref_alpha[s]);
    }
}

int PrefetchRWMHChain::prefetch_find_best_point(int c)
{
    int s = log2(c);
    if(s>=pref_h)
        return 0;
    if(pref_points[c]==0)
        return c;
    int acc = c + (1<<(s+1));
    int rej = acc - (1<<s);
    int k1 = prefetch_find_best_point(acc);
    int k2 = prefetch_find_best_point(rej);
    if( pref_prob[k1] >= pref_prob[k2] )
        return k1;
    else
        return k2;
}

void PrefetchRWMHChain::prefetch_select_poitns()
{
    const int npts = (1<<pref_h);
    /** start with nothing selected */
    for(int i=1; i<npts; ++i) {
        pref_points[i] = NULL;
    }
    pref_prob[0] = 0.0;  /** sentinel */
    for(int i=0; i<pref_evals; ++i) {
        int k = prefetch_find_best_point(1);        
        pref_points[k] = pref_points[0] + k*dim;
        pref_selected[i] = k;
        assert(k>0 && k<npts);
    }
    prefetch_print_tree();
}

void PrefetchRWMHChain::prefetch_set_alpha_const(double alpha)
{
    if(pref_h==0) return;
    std::cout << "Target acceptance rate = " << alpha << std::endl;
    if(pref_alpha==0) pref_alpha = new double[pref_h];
    for(int i=0; i<pref_h; ++i) pref_alpha[i] = alpha;
}

void PrefetchRWMHChain::prefetch_set_alpha_vector(const double *alpha)
{
    if(pref_h==0) return;
    if(pref_alpha==0) pref_alpha = new double[pref_h];
    const int i_1 = 1;
    dcopy(&pref_h, alpha, &i_1, pref_alpha, &i_1);
}

void PrefetchRWMHChain::prefetch(const double *current, double logpi_current)
{
    prefetch_build_tree(current);
    prefetch_compute_target(logpi_current);
}

void PrefetchRWMHChain::prefetch_build_tree(const double *root)
{
    int i_1 = 1;
    if(pref_h==0) return;
    dcopy(&dim, root, &i_1, pref_points[0], &i_1);
    
    if( pref_type == DYNAMIC ) {
        prefetch_compute_probabilities();
        prefetch_select_poitns();
    }
    for(int c=0; c<(1<<(pref_h-1)); ++c) {
        if(pref_points[c]!=0) {
            for(int s= c?int(log2(c))+1:0; s<pref_h; ++s) {
                int k = c + (1<<s);
                if(pref_points[k]!=0) 
                    Q->sample(1, pref_points[c], pref_points[k]);
            }
        }
    }
}

void PrefetchRWMHChain::prefetch_compute_target(double root_target)
{
    pref_logpi[0] = root_target;
    for(int k=1; k < (1<<pref_h); ++k) {
        if(pref_points[k]!=0) {
            pref_logpi[k] = PI->logpdf(1, pref_points[k], 0, k);
        }
    }
}

void PrefetchRWMHChain::prefetch_print_tree()
{
    int depth;
    double D = 0.0;
    std::clog << "prefetching tree:\t";
    for(int k=1; k<(1<<pref_h); ++k) {
        if(pref_points[k]==0) {
            std::clog << ". ";        
        } else {    
            depth = 1 + log2(k);
            D = D + pref_prob[k];
            std::clog << k << " ";
        }
    }
    std::clog << "\n\tdepth = " << depth
            << "\texpected steps = " << D << std::endl;
}

/**************************************************************************/

void PrefetchRWMHChainOMP::prefetch(const double *current, double logpi_current)
{
    prefetch_build_tree(current);
    prefetch_compute_target_omp(logpi_current);
}

void PrefetchRWMHChainOMP::prefetch_compute_target_omp(double logpi_root)
{
    pref_logpi[0] = logpi_root;
    if((pref_selected==0) || (pref_evals==0)) {
        #pragma omp parallel for
        for(int k=1; k < (1<<pref_h); ++k) {
            if(pref_points[k]!=0) {
                pref_logpi[k] = PI->logpdf(1, pref_points[k], 0, k);
            }
        }
    } else {
        #pragma omp parallel for
        for(int i=0; i<pref_evals; ++i) {
            int k = pref_selected[i];
            pref_logpi[k] = PI->logpdf(1, pref_points[k], 0, k);
        }
    }
}

void PrefetchRWMHChainCilk::prefetch(const double *current, double logpi_current)
{
    prefetch_build_tree(current);
    pref_logpi[0] = logpi_current;
    prefetch_compute_target_cilk(0, 0);
}

#include <cilk/cilk.h>
void PrefetchRWMHChainCilk::prefetch_compute_target_cilk(int c, int s)
{
    if(pref_points[c]==NULL) return;
    if(s == pref_h) {
        if(c>0) pref_logpi[c] = PI->logpdf(1, pref_points[c], 0, c);
    } else { 
        cilk_spawn prefetch_compute_target_cilk(c + (1<<s), s+1);
        cilk_spawn prefetch_compute_target_cilk(c         , s+1);
    }
    cilk_sync;
}


