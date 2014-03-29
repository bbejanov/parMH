


#include <cstdlib>
#include <cstdio>
#include <ctime>
#include "my_stat.h"
#include "statmh.hpp"
#include <iostream>

#include <mkl.h>
#include <mkl_lapack.h>

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
    n = argc>2 ? atol(argv[2]) : 50000;
    dim = 15;
    double *chain = new double[n*dim+n];
    double *logpi = chain + n*dim;
    int *accepted = new int[n];
    double start[dim];
    for(int i=0; i<dim; i+=3) {
        start[i]=-8.0; start[i+1]=6.5;  start[i+2]=-1.0;
    }
    //~ rwmh_Example2(&n, start, chain, accepted);
    PrefetchRWMHChain::h =  argc>1 ? atol(argv[1]) : 3;
    std::clog << "h = " << PrefetchRWMHChain::h << std::endl; 
    rwmh_prefetch_Example3(&n, start, chain, logpi, accepted);
    print_chain("Example3", chain, accepted, n, dim+1);
    delete [] chain;
    delete [] accepted;
    std::clog << "Streams used = " << MyRngStream::CountStreams() << std::endl;
} else {
    int work_size = argc>1 ? atol(argv[1]) : 200;
    int repeat = 5000 ; /* + int(10*S.uRand()-5); */ 
    double start = (double) clock();
    for(int i=0; i<repeat; ++i) {
        MyRngStream S(i % 5);
        double *A = new double[work_size*work_size];
        int *ipiv = new int[work_size];
        int info;
        for(int i=0; i<work_size*work_size; ++i)
            A[i] = S.uRand();
        dgetrf(&work_size, &work_size, A, &work_size, ipiv, &info);
        delete[] ipiv;
        delete[] A;
    }
    double total_time = ((double) clock() - start) / (double) CLOCKS_PER_SEC;
    std::clog << "Total time = " << total_time << std::endl
            << "Average time = " << total_time / repeat << std::endl
            << "Streams used = " << MyRngStream::CountStreams() << std::endl;

}
    
    return 0;
}

