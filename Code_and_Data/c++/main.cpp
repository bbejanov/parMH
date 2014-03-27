


#include <cstdlib>
#include <cstdio>
#include "my_stat.h"
#include "statmh.hpp"
#include <iostream>

void print_matrix(const char *name, const double *mat, int n, int m) {
    int i, j;
    printf("%s:\n", name);
    for(i=0; i<n; ++i) {
        for(j=0; j<m; ++j)
            printf("%10g ", mat[n*j+i]);        
        printf("\n");
    }
    printf("\n");
}

void print_chain(const char *name, const double *mat, 
                    const int *acc, int n, int m) {
    int i, j;
    printf("%s:\n", name);
    for(i=0; i<n; ++i) {
        printf("%2i ", acc[i]);
        for(j=0; j<m; ++j)
            printf("%10g ", mat[n*j+i]);        
        printf("\n");
    }
    printf("\n");
}

enum { DIM = 3, N = 10, K = 2};
const int izero=0;
int main(int argc, char **argv)
{
    int dim = DIM;
    int n = N;
    double var[DIM*DIM] = { 4.0, -1.0, 0.0, 
                     -1.0,  2.0, -0.5,
                      0.0, -0.5, 7.0};
    double sqrt_var[DIM*DIM];
    double mean[DIM] = { -1.0, 5.0, 0.0 };    
    double rand_vals[N*DIM];
    double lp_vals[N];

    set_random_seed_from_time();
if(0) {    
    print_matrix("var", var, dim, dim);  
    mvn_sqrt(&dim, var, sqrt_var);
    print_matrix("sqrt(var)", sqrt_var, dim, dim);  
    print_matrix("mean", mean, 1, dim);
    
    mvn_rand(&izero, &n, rand_vals, &dim, mean, var);
    print_matrix("Random numbers", rand_vals, n, dim);

    mvn_logpdf(&n, rand_vals, lp_vals, &dim, mean, var);
    print_matrix("lp values", lp_vals, n, 1);
} else if (0) {
    n = 20000;
    int k = K;
    double prob[K] = {0.8, 0.2};
    double means[K*DIM] = { -3.0, 0.0, 5.0, 1.0, 4.0, 5.0 };
    double covs[K*DIM*DIM] = {
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0
    };    
    double *pts = new double[n*dim+n];
    double *pdf = pts + n*dim;
    mvnm_rand(&izero, &n, pts, &k, prob, &dim, means, covs);
    mvnm_pdf(&n, pts, pdf, &k, prob, &dim, means, covs);
    print_matrix("mvnm values", pts, n, dim+1);
    delete[] pts;
} else if(1) {
    int h = 3;
    n = 10000;
    dim = 3;
    double *chain = new double[n*dim];
    int *accepted = new int[n];
    double start[3] = { -8.0, -4.0, -6.0 }; 
    //~ rwmh_Example2(&n, start, chain, accepted);
    rwmh_prefetch_Example2(&n, &h, start, chain, accepted);
    print_chain("Example2", chain, accepted, n, dim);
    delete [] chain;
    delete [] accepted;
} else {
    
    PrefetchRWMHChain P;
    P.dim = 3;
    double pt[3] = { 1.0, 0.0, -1.0} ;
    P.h = 1;
    P.malloc();
    P.prefetch(pt, 0.0);
    P.free();
    std::cout << std::endl;
    P.h = 2;
    P.malloc();
    P.prefetch(pt, 0.0);
    P.free();
    std::cout << std::endl;
    P.h = 3;
    P.malloc();
    P.prefetch(pt, 0.0);
    P.free();
    std::cout << std::endl;
    P.h = 4;
    P.malloc();
    P.prefetch(pt, 0.0);
    std::cout << std::endl;
}
    
    return 0;
}

