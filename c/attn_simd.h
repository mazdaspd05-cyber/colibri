/* Attention score dot and value mix. Four AVX2 accumulators for n>=64.
 * Shorter lengths stay the scalar loop so glm_tiny's head dims do not move.
 * Include after quant.h (hsum256, immintrin). */
#ifndef COLI_ATTN_SIMD_H
#define COLI_ATTN_SIMD_H
static inline float attn_dot(const float *a, const float *b, int n){
#if defined(__AVX2__)
    if(n>=64){
        int i=0;
        __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
        for(; i+32<=n; i+=32){
            a0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    a0);
            a1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  a1);
            a2=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), a2);
            a3=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), a3);
        }
        __m256 acc=_mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));
        for(; i+8<=n; i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
        float s=hsum256(acc);
        for(; i<n; i++) s+=a[i]*b[i];
        return s;
    }
#endif
    float s=0.f; for(int i=0;i<n;i++) s+=a[i]*b[i];
    return s;
}
static inline void attn_axpy(float *y, float a, const float *x, int n){
#if defined(__AVX2__)
    if(n>=64){
        __m256 va=_mm256_set1_ps(a); int i=0;
        for(; i+8<=n; i+=8)
            _mm256_storeu_ps(y+i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x+i), _mm256_loadu_ps(y+i)));
        for(; i<n; i++) y[i]+=a*x[i];
        return;
    }
#endif
    for(int i=0;i<n;i++) y[i]+=a*x[i];
}
static inline void attn_scale(float *y, float a, int n){
#if defined(__AVX2__)
    if(n>=64){
        __m256 va=_mm256_set1_ps(a); int i=0;
        for(; i+8<=n; i+=8) _mm256_storeu_ps(y+i, _mm256_mul_ps(va, _mm256_loadu_ps(y+i)));
        for(; i<n; i++) y[i]*=a;
        return;
    }
#endif
    for(int i=0;i<n;i++) y[i]*=a;
}
#endif
