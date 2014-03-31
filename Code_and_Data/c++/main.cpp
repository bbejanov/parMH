


#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <iostream>

#include <mkl.h>
#include <mkl_lapack.h>

// #include "my_stat.h"
#include "statmh.hpp"
#include "statdistros.hpp"
#include "mhexamples.hpp"


#include <sys/time.h>
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

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

void print_chain(const char *name,
            const double *chain,
            const double *logpi,
            const int *pref_steps, 
            const int *acc,
            int n, int dim) {
    int i, j;
    printf("%s:\n", name);
    for(i=0; i<n; ++i) {
        for(j=0; j<dim; ++j)
            printf("%10g ", chain[n*j+i]);
        printf("%10g ", logpi[i]);
        printf("%2i ", acc[i]);
        if( pref_steps != 0 ) printf("%2i ", pref_steps[i]);         
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    // set_random_seed_from_time();
    MyRngStream::RngS_set_random_seed();
        
    /** Put a small default for number of points */
    int n = argc>1 ? atol(argv[1]) : 1000; 
    /** NOTE: if h<2, then PrefetchRWMHChain degenerates to RWMHChain */
    int pref_h = argc>2 ? atol(argv[2]) : 1;
    int pref_evals = argc>3 ? atol(argv[3]) : 0;
    int par = argc>4 ? atol(argv[4]) : 1;

    RWMHChain *MyMH;    
    switch (par) {
        case 0:
            MyMH = new RWMHChain;
            std::clog << "RWMHChain (no prefetching)" << std::endl;
            break;
        case 1: 
            MyMH = new PrefetchRWMHChain;
            std::clog << "PrefetchRWMHChain (sequential)" << std::endl;
            break;
        case 2: 
            MyMH = new PrefetchRWMHChainOMP;
            std::clog << "PrefetchRWMHChainOMP (OpenMP)" << std::endl;
            /*{
                double x = 0.0;
                #pragma omp parallel for reduction (+:x)
                for(int i=0; i<10000; ++i)
                    x += log1p(i);
            }*/
            break;
        case 3: 
            MyMH = new PrefetchRWMHChainCilk;
            std::clog << "PrefetchRWMHChainCilk (Cilk)" << std::endl;
            /*{
                double x = 0.0;
                #pragma omp parallel for reduction (+:x)
                for(int i=0; i<10000; ++i)
                    x += log1p(i);
            }*/
            break;
        default:
            std::cerr << "valid third arguments are 1==seq, 2==omp" << std::endl;
            return EXIT_FAILURE;
    }
    if(par>0) {
        dynamic_cast<PrefetchRWMHChain*>(MyMH)->pref_h = pref_h;
    }
    MyMH->PI = new Example3Target;

    int dim = MyMH->dim = MyMH->PI->dim;
    MyMH->Q = new RandomWalkProposalDistribution(dim);
    double start_point[dim];
    for(int i=0; i<dim; ++i) start_point[i] = -7.0;  /** why not? **/
    
    double start_time, full_time;

    start_time = get_wall_time();
    for(int i=0; i<20; ++i)
        static_cast<const Example3Target *>(MyMH->PI)->work_for_delay(0);
    full_time = get_wall_time() - start_time;
    std::clog << "avg time of work_for_delay() = " << full_time / 20.0 << "sec." << std::endl;

    if(par>0) {
        PrefetchRWMHChain *P = dynamic_cast<PrefetchRWMHChain*>(MyMH);
        P->pref_type = PrefetchRWMHChain::STATIC;
        P->prefetch_set_alpha_const(0.234);
        P->pref_evals = pref_evals;
    }
    
    start_time = get_wall_time();
    MyMH->run(n, start_point);
    full_time = get_wall_time() - start_time;

    std::clog << "h = " << pref_h << std::endl
        << "n = " << n << std::endl
        << "time = " << full_time << "sec." << std::endl
        << "avg time per step = " << full_time / n << "sec." << std::endl;
        
/*
    print_chain("Full Prefetching", MyMH->chain, MyMH->logpi,
        par==0 ? NULL : dynamic_cast<PrefetchRWMHChain*>(MyMH)->pref_at_step,
        MyMH->accepted, n, dim);
*/
    {
        double averageAccRate = 0.0;
        for(int i=0; i<n; ++i)
            averageAccRate += MyMH->accepted[i];
        averageAccRate = averageAccRate / (double) n;
        std::clog << "average acceptance rate = " << averageAccRate << std::endl;
    }

    if (par > 0) {
        PrefetchRWMHChain *P = dynamic_cast<PrefetchRWMHChain*>(MyMH);
        if(P->pref_at_step!=0) {
            double sum_s=0.0, max_s=-INFINITY, min_s=INFINITY, count_s=0.0;
            for(int i=0; i<n; ++i) {
                int s = P->pref_at_step[i];
                if(s > 0) {
                    sum_s += s;
                    count_s += 1.0;
                    if( s > max_s ) max_s = s;
                    if( s < min_s ) min_s = s;
                }
            }
            std::clog << "successful prefetching steps: " // << std::endl
                << "\tmin=" << min_s << "\tavg=" << sum_s / count_s
                << "\tmax=" << max_s << std::endl;
        }
    }

    delete MyMH->PI;
    delete MyMH->Q;
    delete MyMH;
        
    return EXIT_SUCCESS;
}

