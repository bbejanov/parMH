
#include <stdexcept>

#include <cmath>

#include <mkl.h>
#include <mkl_blas.h>

#include "statmh.hpp"

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

void PrefetchRWMHChain::free_points() {
    if(pref_points!=0) {
        delete[] pref_points[0];
        delete[] pref_points;
    }
    if(pref_logpi!=0) 
        delete[] pref_logpi;
}

void PrefetchRWMHChain::alloc_points() {
    if (h>0) {
        int pow2n = 1 << h;
        pref_points = (double **) ::operator new (pow2n*sizeof(double*));
        pref_points[0] = new double[dim*pow2n];
        for(int i=1; i<pow2n; ++i) 
            pref_points[i] = pref_points[i-1]+dim;
        pref_logpi = new double[pow2n];
    }
}

void PrefetchRWMHChain::run(int n, const double *start, double *ch, double *l, int *a)
{
    
    if (h < 2) {
        RWMHChain::run(n, start, ch, l, a);
        return;
    }           
    
    check_run_args(n, start, ch, l, a);
    this->pref_at_step = new int[n];
    memset(this->pref_at_step, 0, n*sizeof(int));

    this->free_points();
    this->alloc_points();

    double lpi_c = PI->logpdf(1, start, 0, 1<<h);
    int c; /* index of current point */
    int p; /* index of proposed point */
    int s; /* step within the current prefetching */

    this->prefetch(start, lpi_c);
    s = 0;
    c = 0;
    for(int i=0; i<n; ++i) {
        s = s + 1;
        p = c + (1 << (s-1));
        if( (s > h) || (pref_points[p] == NULL) ) {
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

void PrefetchRWMHChain::prefetch(const double *current, double logpi_c)
{
    int i_1 = 1;
    if(h==0) return;
    dcopy(&dim, current, &i_1, pref_points[0], &i_1);
    
    for(int c=0; c<(1<<(h-1)); ++c) {
        for(int s= c?int(log2(c))+1:0; s<h; ++s) {
            int k = c + (1<<s);
            Q->sample(1, pref_points[c], pref_points[k]);
            //~ std::cout << "c = " << c << ", s = "
                    //~ << (1+s) << ", k = " << k << std::endl;
        }
    }
    pref_logpi[0] = logpi_c;
    for(int k=1; k<(1 << h); ++k)
        pref_logpi[k] = PI->logpdf(1, pref_points[k], 0, k);
}

int PrefetchRWMHChain::h = 0;




void PrefetchRWMHChainOMP::prefetch(const double *current, double logpi_c)
{
    int i_1 = 1;
    if(h==0) return;
    dcopy(&dim, current, &i_1, pref_points[0], &i_1);
    
    for(int c=0; c<(1<<(h-1)); ++c) {
        for(int s= c?int(log2(c))+1:0; s<h; ++s) {
            int k = c + (1<<s);
            Q->sample(1, pref_points[c], pref_points[k]);
        }
    }
    pref_logpi[0] = logpi_c;
    
    #pragma omp parallel for
    for(int k=1; k<(1 << h); ++k)
        pref_logpi[k] = PI->logpdf(1, pref_points[k], 0, k);
}

void PrefetchRWMHChainCilk::prefetch(const double *current, double logpi_c)
{
    int i_1 = 1;
    if(h==0) return;
    dcopy(&dim, current, &i_1, pref_points[0], &i_1);
    
    for(int c=0; c<(1<<(h-1)); ++c) {
        for(int s= c?int(log2(c))+1:0; s<h; ++s) {
            int k = c + (1<<s);
            Q->sample(1, pref_points[c], pref_points[k]);
        }
    }

    prefetch_logpi_cilk(0, 0);
}

#include "cilk/cilk.h"
void PrefetchRWMHChainCilk::prefetch_logpi_cilk(int c, int s)
{
    if (s == h) {
        pref_logpi[c] = PI->logpdf(1, pref_points[c], 0, c);
    } else { 
        cilk_spawn prefetch_logpi_cilk(c + (1<<s), s+1);
        cilk_spawn prefetch_logpi_cilk(c         , s+1);
    }
    cilk_sync;
}


