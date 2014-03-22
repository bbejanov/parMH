/*
 * untitled.cxx
 * 
 * Copyright 2014 Unknown <bejb@gg-m.bbejanov.dlinkddns.com>
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


#include <cstdlib>
#include <cstdio>
#include "my_stat.h"

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
    
    print_matrix("var", var, dim, dim);  
    mvn_sqrt(&dim, var, sqrt_var);
    print_matrix("sqrt(var)", sqrt_var, dim, dim);  
    print_matrix("mean", mean, 1, dim);
    
    mvn_rand(&izero, &n, rand_vals, &dim, mean, var);
    print_matrix("Random numbers", rand_vals, n, dim);

    mvn_logpdf(&n, rand_vals, lp_vals, &dim, mean, var);
    print_matrix("lp values", lp_vals, n, 1);

    
    return 0;
}

