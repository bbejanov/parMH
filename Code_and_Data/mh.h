#ifndef __MH_H
#define __MH_H

typedef void (*target_logpdf_t)(const int *n, const int *dim, 
    const double *points, double *lp, void *tdata);


void rwmh_chain(const int *n, const int *d, const double *start, 
    target_logpdf_t t_lpdf, void *tdata,
    const double *prop_var,
    double *chain, int *accepted);
        

void target_example_1(const int *n, const int *dim, 
    const double *points, double *lp, void *tdata);
        
#endif
