#ifndef __RNGS_UTILS_H
#define __RNGS_UTILS_H

#include <RngStream.h>


void rngs_set_random_seed();

RngStream get_gar(int w);

/** uniform random in [0,1] */
void urand(const int *which_gar, int *n, double *vals);

/** multinomial random with given bin-probabilities **/
/* NOTE: bins are numbered starting from 1 (human friendly) */
void mnrand(const int *which_gar, int *n, int *bin, 
        const int *num_bins, const double *prob_bins);        
/* the next one returns 0-based bins, for C programmers */
void mnrand0(const int *which_gar, int *n, int *bin, 
        const int *num_bins, const double *prob_bins);

#endif
