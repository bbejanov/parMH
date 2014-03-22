
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

