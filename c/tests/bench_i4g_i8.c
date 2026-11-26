/* COLI_I8 kernel vs the f32 grouped kernel. Not a test gate.
 *   OMP_NUM_THREADS=1 ./tests/bench_i4g_i8
 * Prints seconds for 8 calls. The int8 result is not the f32 result. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../quant.h"

static uint32_t rs=3;
static uint32_t xr(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

static void time_one(int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    uint8_t *q4=malloc((size_t)O*rb);
    float *sc=malloc((size_t)O*ng*sizeof(float));
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float));
    for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(xr()&255);
    for(int i=0;i<O*ng;i++) sc[i]=0.01f;
    for(int i=0;i<S*I;i++) x[i]=(float)((int)(xr()%200)-100)/50.f;
    const int REPS=8;
    double ft[3], it[3];
    for(int r=0;r<3;r++){
        double t0=now(); for(int k=0;k<REPS;k++) matmul_i4_grouped(y,x,q4,sc,S,I,O,gs); ft[r]=now()-t0;
        t0=now(); for(int k=0;k<REPS;k++) matmul_i4_grouped_i8(y,x,q4,sc,S,I,O,gs); it[r]=now()-t0;
        printf("S=%d run %d  f32 %.4f s  i8 %.4f s  %.2fx\n", S, r, ft[r], it[r], ft[r]/it[r]);
    }
    free(q4);free(sc);free(x);free(y);
    (void)ft;(void)it;
}

int main(void){
    time_one(1, 2048, 2048, 64);
    time_one(16, 2048, 512, 64);
    return 0;
}
