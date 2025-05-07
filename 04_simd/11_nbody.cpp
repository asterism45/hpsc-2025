#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main()
{
  constexpr int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];

  for (int i = 0; i < N; ++i)
  {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0.0f;
  }

  const __m512 zero_ps = _mm512_setzero_ps();
  const __m512i idx_base = _mm512_set_epi32(
      15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

  for (int i = 0; i < N; ++i)
  {
    const __m512 xi = _mm512_set1_ps(x[i]);
    const __m512 yi = _mm512_set1_ps(y[i]);
    __m512 fx_vec = zero_ps;
    __m512 fy_vec = zero_ps;

    const __m512 xj = _mm512_load_ps(x);
    const __m512 yj = _mm512_load_ps(y);
    const __m512 mj = _mm512_load_ps(m);

    const __m512 rx = _mm512_sub_ps(xi, xj);
    const __m512 ry = _mm512_sub_ps(yi, yj);
    const __m512 r2 = _mm512_fmadd_ps(ry, ry, _mm512_mul_ps(rx, rx));

    __m512 inv_r = _mm512_rsqrt14_ps(r2);
    __m512 inv_r2 = _mm512_mul_ps(inv_r, inv_r);
    __m512 inv_r3 = _mm512_mul_ps(inv_r, inv_r2);

    __m512 coef = _mm512_mul_ps(mj, inv_r3);

    __m512i idx_vec = _mm512_add_epi32(idx_base, _mm512_set1_epi32(0));
    __mmask16 self = _mm512_cmpeq_epi32_mask(idx_vec, _mm512_set1_epi32(i));

    coef = _mm512_mask_blend_ps(self, coef, zero_ps);

    fx_vec = _mm512_fmadd_ps(coef, rx, fx_vec);
    fy_vec = _mm512_fmadd_ps(coef, ry, fy_vec);

    fx[i] = -_mm512_reduce_add_ps(fx_vec);
    fy[i] = -_mm512_reduce_add_ps(fy_vec);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
  return 0;
}
