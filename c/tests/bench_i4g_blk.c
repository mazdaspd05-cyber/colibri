/* Single-thread: per-row AVX2 grouped int4 vs the blocked path that decodes
 * each group once and reuses it across prompt rows. Not a test gate.
 *
 *   make tests/bench_i4g_blk && OMP_NUM_THREADS=1 ./tests/bench_i4g_blk
 *
 * The "old" kernel is the per-row loop this path replaced for S>=4. The "new"
 * kernel is matmul_i4_grouped. They must match bit for bit; the bench checks
 * that before timing. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <time.h>

static uint32_t rs=0xA5A5u;
static uint32_t xr(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }

/* The pre-change per-row AVX2 body, one output row at a time. */
static void matmul_i4_grouped_row(float *y, const float *x, const uint8_t *q4, const float *scale,
                                  int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs, glen=gs; if(base+glen>I) glen=I-base;
                float sc=scl[g]; int i=base;
                const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
                __m256 acc=_mm256_setzero_ps();
                for(; i+16<=base+glen; i+=16){
                    __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                    __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), w0, acc);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc);
                }
                a=fmaf(hsum256(acc),sc,a);
                for(; i+1<base+glen; i+=2){
                    uint8_t byte=w[i>>1];
                    a=i4g_step2(a,xs[i],(float)((int)(byte&0xF)-8),xs[i+1],(float)((int)(byte>>4)-8),sc);
                }
                if(i<base+glen){ uint8_t byte=w[i>>1]; a=i4g_step1(a,xs[i],(float)((int)(byte&0xF)-8),sc); }
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

static double bench_now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void){
#ifndef __AVX2__
    fprintf(stderr,"bench_i4g_blk needs AVX2\n"); return 1;
#else
    const int S=32, I=2048, O=1024, gs=64;
    const int REPS=8;
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    uint8_t *q4=malloc((size_t)O*rb);
    float *scale=malloc((size_t)O*ng*sizeof(float));
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y0=malloc((size_t)S*O*sizeof(float));
    float *y1=malloc((size_t)S*O*sizeof(float));
    if(!q4||!scale||!x||!y0||!y1){ fprintf(stderr,"OOM\n"); return 1; }
    for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(xr()&255);
    for(int i=0;i<O*ng;i++) scale[i]=(float)((xr()%1000)+1)/5000.f;
    for(int i=0;i<S*I;i++) x[i]=(float)((int)(xr()%2001)-1000)/1000.f;
    matmul_i4_grouped_row(y0,x,q4,scale,S,I,O,gs);
    matmul_i4_grouped(y1,x,q4,scale,S,I,O,gs);
    int bad=0;
    for(int i=0;i<S*O;i++) if(y0[i]!=y1[i]) bad++;
    if(bad){ fprintf(stderr,"MISMATCH %d/%d\n", bad, S*O); return 1; }
    printf("bit-exact (%d outputs)\n", S*O);
    double oldt[3], newt[3];
    for(int r=0;r<3;r++){
        double t0=bench_now(); for(int k=0;k<REPS;k++) matmul_i4_grouped_row(y0,x,q4,scale,S,I,O,gs); oldt[r]=bench_now()-t0;
        t0=bench_now(); for(int k=0;k<REPS;k++) matmul_i4_grouped(y1,x,q4,scale,S,I,O,gs); newt[r]=bench_now()-t0;
        printf("run %d  row %.4f s   blocked %.4f s   %.2fx\n", r, oldt[r], newt[r], oldt[r]/newt[r]);
    }
    /* median of three */
    double os[3]={oldt[0],oldt[1],oldt[2]}, ns[3]={newt[0],newt[1],newt[2]};
    for(int i=0;i<3;i++) for(int j=i+1;j<3;j++){ if(os[j]<os[i]){ double t=os[i]; os[i]=os[j]; os[j]=t; } if(ns[j]<ns[i]){ double t=ns[i]; ns[i]=ns[j]; ns[j]=t; } }
    printf("median row %.4f s  blocked %.4f s  %.2fx\n", os[1], ns[1], os[1]/ns[1]);
    return 0;
#endif
}
