
#include "statmh.hpp"

#include <stdexcept>

#include <cmath>
#include <mkl.h>
#include <mkl_blas.h>


void MHChain::check_run_args(int n, const double *start, double *c, int *a)
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
        delete[] accepted;
        have = 0;
    }
    if((c != 0) && (a != 0)) {
        chain = c;
        accepted = a;
    } else if ((c==0) && (a==0)) {
        chain = new double[dim*n];
        accepted = new int[n];
        have = n;
    } else {
        throw std::invalid_argument(
            "MHChain.run: must give either both c and a or none of them");
    }
}

void RWMHChain::run(int n, const double *start, double *c, int *a)
{
    check_run_args(n, start, c, a);

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
    }
}

void PrefetchRWMHChain::prefetch(const double *current, double logpi_c)
{
    int i_1 = 1;
    if(h==0) return;
    dcopy(&dim, current, &i_1, points[0], &i_1);
    
    for(int c=0; c<(1<<(h-1)); ++c) {
        for(int s= c?int(log2(c))+1:0; s<h; ++s) {
            int k = c + (1<<s);
            Q->sample(1, points[c], points[k]);
            //~ std::cout << "c = " << c << ", s = "
                    //~ << (1+s) << ", k = " << k << std::endl;
        }
    }
    logpi_vals[0] = logpi_c;
    for(int k=1; k<(1 << h); ++k)
        logpi_vals[k] = PI->logpdf(1, points[k]);
}

void PrefetchRWMHChain::run(int n, const double *start, double *ch, int *a)
{
    
    if (h == 0) {
        RWMHChain::run(n,start,ch,a);
        return;
    }           
    
    check_run_args(n, start, ch, a);

    this->free();
    this->malloc();

    double lpi_c = PI->logpdf(1, start);
    int c; /* index of current point */
    int p; /* index of proposed point */
    int s; /* step within the current prefetching */

    prefetch(start, lpi_c);
    s = 0;
    c = 0;
    for(int i=0; i<n; ++i) {
        if( s >= h ) {
            prefetch(points[c], logpi_vals[c]);
            c = 0;
            s = 0;
        }
        s = s + 1;
        p = c + (1 << (s-1));
        double U = Q->urand();
        if( log(U) < logpi_vals[p] - logpi_vals[c] ) {
            accepted[i] = 1;
            c = p;
        } else {
            accepted[i] = 0;
        }
        const int i_1 = 1;
        dcopy(&dim, points[c], &i_1, chain+i, &n);
    }
}

