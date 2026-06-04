#include <x86intrin.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define UNROLL 4
#define BLOCKSIZE 32
#define ALIGNMENT 64 

// gcc -mavx512f -mavx512dq -mfma -o dgemm.exe "3.1. dgemm parallel com 512 avx.( nao tem suporte)c"
void dgemm_parallel (int n, double* A, double* B, double* C)
{
    for ( int i = 0; i < n; i += UNROLL * 8 ) {
        for ( int j = 0; j < n; j++ ) {
            __m512d c[UNROLL];
            for ( int x = 0; x < UNROLL; x++ )
                c[x] = _mm512_load_pd(C + i + x * 8 + j * n);

            for( int k = 0; k < n; k++ ) {
                __m512d b = _mm512_set1_pd(B[k + j * n]);
                for (int x = 0; x < UNROLL; x++)
                    c[x] = _mm512_fmadd_pd(_mm512_load_pd(A + n * k + x * 8 + i), b, c[x]);
            }

            for ( int x = 0; x < UNROLL; x++ )
                _mm512_store_pd(C + i + x * 8 + j * n, c[x]);
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

    size_t n = 1024;

    double* a = (double*)_mm_malloc(n * n * sizeof(double),ALIGNMENT);
    double* b = (double*)_mm_malloc(n * n * sizeof(double),ALIGNMENT);
    double* c = (double*)_mm_malloc(n * n * sizeof(double),ALIGNMENT);


    inicializar_1(n, a);
    inicializar_1(n, b);

    double times[5];

    for(int i = 0; i < 5; i++) {
        inicializar_0(n, c);
        clock_t start = clock();
        dgemm_parallel(n, a, b, c);
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