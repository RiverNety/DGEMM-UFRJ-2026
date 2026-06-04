#include <x86intrin.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define UNROLL 8
#define BLOCKSIZE 32
#define ALIGNMENT 64 

// gcc -mavx2 -mfma -o dgemm.exe '3.2. dgemm parallel e fma.c'
void dgemm_parallel_fma (size_t n, double* A, double* B, double* C)
{
    for (size_t i = 0; i < n; i += UNROLL * 4)
    {
        for (size_t j = 0; j < n; ++j)
        {
            __m256d c[UNROLL];
            for (int r = 0; r < UNROLL; r++)
            {
                c[r] = _mm256_load_pd(C + i + r * 4 + j * n);
            }

            for (size_t k = 0; k < n; k++)
            {
                __m256d bb = _mm256_broadcast_sd(B + k + j * n);

                for (int r = 0; r < UNROLL; r++)
                {
                    __m256d aa = _mm256_load_pd(A + k * n + r * 4 + i);
                    c[r] = _mm256_fmadd_pd(aa, bb, c[r]);
                }
            }
            for (int r = 0; r < UNROLL; r++)
            {
                _mm256_store_pd(C + i + r * 4 + j * n, c[r]);
            }
        }
    }
}

void inicializar_1(size_t n, double* a) {
    for (size_t i = 0; i < n * n; i++) {
        a[i] = 1;
    }
}

void inicializar_0(size_t n, double* a) {
    for (size_t i = 0; i < n * n; i++) {
        a[i] = 0;
    }
}

int main(int argc, char* argv[]) {

    size_t n = 1000;

    double* a = (double*)_mm_malloc(n * n * sizeof(double),ALIGNMENT);
    double* b = (double*)_mm_malloc(n * n * sizeof(double),ALIGNMENT);
    double* c = (double*)_mm_malloc(n * n * sizeof(double),ALIGNMENT);


    inicializar_1(n, a);
    inicializar_1(n, b);

    double times[5];

    for(int i = 0; i < 5; i++) {
        inicializar_0(n, c);
        clock_t start = clock();
        dgemm_parallel_fma(n, a, b, c);
        clock_t stop = clock();

        double elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC * 1000;
        times[i] = elapsed_time;
    }


    printf("\nvalores de tempo:\n");
    for(int i = 0; i < 5; i++) {
        printf(" %d: %.0f ms\n", i + 1, times[i]);
    }

    //da problema?

    _mm_free(a);
    _mm_free(b);
    _mm_free(c);

    return 0;
}