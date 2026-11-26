/* COLI_I8's pair-layout grouped int4 x int8 dot, against a scalar int32 reference.
 * The weights are the on-disk nibble order (no planar repack). Integer sums are
 * exact, so AVX2 maddubs must match the scalar sum bit for bit, including a
 * partial last group and an odd I. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../quant.h"

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while (0)

static uint32_t rs=1;
static uint32_t xr(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }

static void ref(float *y, const float *x, const uint8_t *q4, const float *scale,
                int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    int8_t *xq=(int8_t*)malloc((size_t)S*I);
    float *sx=(float*)malloc((size_t)S*sizeof(float));
    for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const int8_t *xr=xq+(int64_t)s*I; float a=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs, glen=gs; if(base+glen>I) glen=I-base;
                int32_t d=0, sm=0;
                for(int i=0;i<glen;i++){
                    int aidx=base+i; uint8_t b=w[aidx>>1];
                    int u=(aidx&1)?(int)(b>>4):(int)(b&15);
                    d+=(int32_t)u*xr[aidx]; sm+=xr[aidx];
                }
                a=i4p_fmaf((float)(d-8*sm), scl[g], a);
            }
            y[(int64_t)s*O+o]=a*sx[s];
        }
    }
    free(xq); free(sx);
}

static int check(const char *name, int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    uint8_t *q4=malloc((size_t)O*rb);
    float *scale=malloc((size_t)O*ng*sizeof(float));
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float));
    float *r=malloc((size_t)S*O*sizeof(float));
    if(!q4||!scale||!x||!y||!r){ fprintf(stderr,"OOM\n"); return 1; }
    for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(xr()&255);
    for(int i=0;i<O*ng;i++) scale[i]=(float)((xr()%500)+1)/1000.f * ((xr()&1)?1.f:-1.f);
    for(int i=0;i<S*I;i++) x[i]=(float)((int)(xr()%2001)-1000)/50.f;
    matmul_i4_grouped_i8(y,x,q4,scale,S,I,O,gs);
    ref(r,x,q4,scale,S,I,O,gs);
    int bad=0;
    for(int i=0;i<S*O;i++) if(y[i]!=r[i]){
        if(bad<3) fprintf(stderr,"%s [%d] got %.9g want %.9g\n", name, i, (double)y[i], (double)r[i]);
        bad++;
    }
    free(q4);free(scale);free(x);free(y);free(r);
    if(bad){ fprintf(stderr,"%s FAIL %d\n", name, bad); return 1; }
    printf("  %-40s ok\n", name);
    return 0;
}

int main(void){
    int fail=0;
    fail|=check("gs64 S=1", 1, 256, 4, 64);
    fail|=check("gs64 S=4", 4, 256, 8, 64);
    fail|=check("gs64 S=5 tail", 5, 200, 3, 64);
    fail|=check("gs64 odd I", 4, 201, 3, 64);
    fail|=check("gs64 S=8 I=2048", 8, 2048, 16, 64);
    fail|=check("gs128 S=3", 3, 256, 4, 128);
    if(fail){ fprintf(stderr,"test_i4g_i8 FAIL\n"); return 1; }
    printf("test_i4g_i8 ok\n");
    return 0;
}
