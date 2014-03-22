/*
 * main.c
 * 
 * Copyright 2014 Boyan Bejanov <bejb@alvin.bbejanov.dlinkddns.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */


#include <stdio.h>
#include "mh.h"
#include "rngs_utils.h"
#include "gaussian.h"

#include <mkl.h>

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

enum { DIM = 3, N = 10 };
static const int izero = 0; 

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

    rngs_set_random_seed();
if(0) {
    print_matrix("var", var, dim, dim);  
    mvn_sqrt(&dim, var, sqrt_var);
    print_matrix("sqrt(var)", sqrt_var, dim, dim);  
    print_matrix("mean", mean, 1, dim);
    
    mvn_rand(&izero, &n, rand_vals, &dim, mean, sqrt_var);
    print_matrix("Random numbers", rand_vals, n, dim);

    mvn_logpdf(&n, rand_vals, lp_vals, &dim, mean, sqrt_var);
    print_matrix("lp values", lp_vals, n, 1);
}
    n = 20000;
    dim = 2;
    double *chain = (double*) malloc(n*(dim+1)*sizeof(double));
    double start[2] = { 4.0, -4.0};
    int *accepted = (int *) malloc(n*sizeof(int));
    double prop_var[4] = { 0.2, 0.0, 0.0, 0.2 };
    
    rwmh_chain(&n, &dim, start, target_example_2,  NULL, 
        prop_var, chain, accepted);
    target_example_2(&n, &dim, chain, chain+n*dim, NULL);
    print_chain("Chain", chain, accepted, n, 1+dim);

    rand_example_2(&izero, &n, chain, &dim, NULL);
    target_example_2(&n, &dim, chain, chain+n*dim, NULL);
    print_chain("Direct", chain, accepted, n, 1+dim);
    

    free(chain);
    free(accepted);
    return 0;
}

