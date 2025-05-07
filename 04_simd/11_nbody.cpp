#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

constexpr int VL = 16;
constexpr int N = 16;

int main()
{
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];

  for (int i = 0; i < N; ++i)
  {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0.0f;
  }

  const __m512i lane = _mm512_set_epi32(
      15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

  for (int i = 0; i < N; ++i)
  {
    const __m512 xi = _mm512_set1_ps(x[i]);
    const __m512 yi = _mm512_set1_ps(y[i]);

    float fxi = 0.0f, fyi = 0.0f;

    for (int j = 0; j < N; j += VL)
    {
      const __m512 xj = _mm512_loadu_ps(&x[j]);
      const __m512 yj = _mm512_loadu_ps(&y[j]);
      const __m512 mj = _mm512_loadu_ps(&m[j]);

      const __m512 rx = _mm512_sub_ps(xi, xj);
      const __m512 ry = _mm512_sub_ps(yi, yj);

      const __m512 r2 = _mm512_fmadd_ps(ry, ry, _mm512_mul_ps(rx, rx));
      const __m512 inv_r = _mm512_rsqrt14_ps(r2);
      const __m512 inv_r3 = _mm512_mul_ps(inv_r, _mm512_mul_ps(inv_r, inv_r));

      __m512 s = _mm512_mul_ps(mj, inv_r3);

      const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(j), lane);
      const __mmask16 self = _mm512_cmpeq_epi32_mask(idx, _mm512_set1_epi32(i));
      s = _mm512_maskz_mov_ps(~self, s);

      const __m512 fxv = _mm512_mul_ps(rx, s);
      const __m512 fyv = _mm512_mul_ps(ry, s);

      fxi -= _mm512_reduce_add_ps(fxv);
      fyi -= _mm512_reduce_add_ps(fyv);
    }
    fx[i] = fxi;
    fy[i] = fyi;
    printf("%2d  % .6f % .6f\n", i, fx[i], fy[i]);
  }
  return 0;
}
