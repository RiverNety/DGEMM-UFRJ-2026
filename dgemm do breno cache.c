#include <x86intrin.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define UNROLL 8
#define BLOCKSIZE 32
#define ALIGNMENT 64 


void do_block (size_t n, size_t si, size_t sj, size_t sk,
               double *A, double *B, double *C)
{
    // The i-loop stride must be UNROLL * 4 = 16 for AVX2 (4 doubles per register)
    for (size_t i = si; i < si + BLOCKSIZE; i += UNROLL * 4)
    {
        for (size_t j = sj; j < sj + BLOCKSIZE; j++)
        {
            // Declare UNROLL (4) AVX2 registers for C[i..i+15][j]
            __m256d c[UNROLL];

            // Load the initial values of C[i..i+15][j]
            for (int r = 0; r < UNROLL; r++)
            {
                // Each load handles 4 doubles: offset r * 4
                c[r] = _mm256_load_pd(C + i + r * 4 + j * n);
            }

            for (size_t k = sk; k < sk + BLOCKSIZE; k++)
            {
                // Broadcast B[k][j] (B[k + j*n])
                __m256d bb = _mm256_broadcast_sd(B + k + j * n);

                for (int r = 0; r < UNROLL; r++)
                {
                    // Load A[i + r*4 .. i + r*4 + 3][k]
                    __m256d aa = _mm256_load_pd(A + k * n + r * 4 + i);

                    // Fused Multiply-Add: c[r] += aa * bb
                    c[r] = _mm256_fmadd_pd(aa, bb, c[r]);
                }
            }

            // Store the final accumulated values back into C
            for (int r = 0; r < UNROLL; r++)
            {
                _mm256_store_pd(C + i + r * 4 + j * n, c[r]);
            }
        }
    }
}

double dgemm (size_t n, double* A, double* B, double* C)
{
    for (size_t sj = 0; sj < n; sj += BLOCKSIZE)
        for (size_t si = 0; si < n; si += BLOCKSIZE)
            for (size_t sk = 0; sk < n; sk += BLOCKSIZE)
                do_block(n, si, sj, sk, A, B, C);
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
        dgemm(n, a, b, c);
        clock_t stop = clock();

        double elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC * 1000;
        times[i] = elapsed_time;
    }


    printf("\nvalores de tempo:\n");
    for(int i = 0; i < 5; i++) {
        printf(" %d: %.0f ms\n", i + 1, times[i]);
    }

    //da problema?

    free(a);
    free(b);
    free(c);

    return 0;
}