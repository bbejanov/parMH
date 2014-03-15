#ifndef __RNGS_UTILS_H
#define __RNGS_UTILS_H

#include <RngStream.h>


void rngs_set_random_seed();

RngStream get_gar(int w);

void urand(const int *which_gar, int *n, double *vals);

#endif
