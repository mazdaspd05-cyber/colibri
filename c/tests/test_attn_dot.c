/* attn_dot / attn_axpy. Lengths under 64 match a scalar loop bit for bit
 * (glm_tiny). Longer dots stay close to a double sum; the axpy matches fmaf
 * per element. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../quant.h"
#include "../attn_simd.h"

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while (0)

static float scalar_dot(const float *a, const float *b, int n){
    float s=0.f; for(int i=0;i<n;i++) s+=a[i]*b[i]; return s;
}

static int test_short(void){
    float a[32], b[32], y[32], y2[32];
    for(int i=0;i<32;i++){ a[i]=(float)(i-16)/7.f; b[i]=(float)(i%5-2); y[i]=0.1f*(i+1); y2[i]=y[i]; }
    for(int n=1;n<=32;n++){
        CHECK(attn_dot(a,b,n)==scalar_dot(a,b,n));
        attn_axpy(y, 0.25f, b, n);
        for(int i=0;i<n;i++) y2[i]+=0.25f*b[i];
        for(int i=0;i<n;i++) CHECK(y[i]==y2[i]);
    }
    printf("  n<64 bit-exact vs scalar\n");
    return 0;
}

static int test_long(void){
    const int n=192;
    float *a=malloc(n*sizeof(float)), *b=malloc(n*sizeof(float));
    double ref=0;
    for(int i=0;i<n;i++){ a[i]=(float)((i*17)%50-25)/9.f; b[i]=(float)((i*13)%40-20)/6.f; ref+=(double)a[i]*(double)b[i]; }
    float got=attn_dot(a,b,n);
    double rel=fabs((double)got-ref)/(fabs(ref)+1.0);
    printf("  n=192 rel vs double %.3g\n", rel);
    CHECK(rel<1e-5);
    float *y=malloc(n*sizeof(float)), *x=malloc(n*sizeof(float));
    for(int i=0;i<n;i++){ y[i]=(float)(i%7)*0.1f; x[i]=a[i]; }
    attn_axpy(y, 0.5f, x, n);
    for(int i=0;i<n;i++){
        float want=fmaf(0.5f, x[i], (float)(i%7)*0.1f);
        CHECK(y[i]==want);
    }
    free(a);free(b);free(y);free(x);
    printf("  axpy n=192 matches fmaf per element\n");
    return 0;
}

int main(void){
    if(test_short()||test_long()){ fprintf(stderr,"test_attn_dot FAIL\n"); return 1; }
    printf("test_attn_dot ok\n");
    return 0;
}
