// bucket_sort_gpu.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __managed__ int pos = 0;

__global__ void init_bucket(int *bucket, int range)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range)
    bucket[i] = 0;
}

__global__ void count_keys(const int *key, int *bucket, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void expand_keys(int *key, int *bucket, int range)
{
  int val = blockIdx.x;
  int cnt = bucket[val];
  while (cnt-- > 0)
  {
    int idx = atomicAdd(&pos, 1);
    key[idx] = val;
  }
}

int main()
{
  const int n = 50;
  const int range = 5;
  int *key, *bucket;

  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i = 0; i < n; ++i)
  {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  int B = 256;
  init_bucket<<<(range + B - 1) / B, B>>>(bucket, range);

  count_keys<<<(n + B - 1) / B, B>>>(key, bucket, n);

  expand_keys<<<range, 1>>>(key, bucket, range);

  cudaDeviceSynchronize();

  for (int i = 0; i < n; ++i)
    printf("%d ", key[i]);
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  return 0;
}
