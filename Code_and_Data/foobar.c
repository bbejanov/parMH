
#include <RngStream.h>
#include <stdio.h>

#ifndef DEBUG
#define DEBUG 0!=0
#endif

enum { MAX_NUM_GARS = 100 };

RngStream init_stream() 
{
    RngStream g;
    g = RngStream_CreateStream("");
    RngStream_IncreasedPrecis(g, 1);
    return g;
}


/* 
    urand - uses RngStreams to generate random numbers
            from multiple independent streams
    which_gar - in - stream number, zero-based
    num_vals  - in - number of values to generate
    vals      - out - array of size n containing the generated values
*/
void urand(int *which_gar, int *num_vals, double *vals) 
{
   static int num_gars = -1; 
   static RngStream gar[MAX_NUM_GARS];

   int w = which_gar[0];
   int n = num_vals[0];
   int i;

   if(DEBUG) printf("w = %d, n = %d, v[0] = %g\n", w, n, v[0]);

   if (w < 0) {
      printf("Invalid stream index (must be > 0)\n");
      return;
   }

   if (w >= MAX_NUM_GARS) {
      printf("Too many streams (max streams = %d)\n", MAX_NUM_GARS);
      return;
   }

   while ( w > num_gars ) {
      ++num_gars;
      if(DEBUG) printf("Initializing stream %d\n", num_gars);
      gar[num_gars] = init_stream();
   }

   for(i=0 ; i<n; ++i) 
      vals[i] = RngStream_RandU01(gar[w]);

   return;
} 

