#include <x86intrin.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define UNROLL 8
#define BLOCKSIZE 64
#define ALIGNMENT 64

/*
    gcc -mavx2 -o dgemm.exe '4. dgemm cache block corrigido.c'
*/
void do_block_cache (int n, int si, int sj, int sk,
 double *A, double *B, double *C)
{
 for ( int i = si; i < si+BLOCKSIZE; i+=UNROLL*4 )
 for ( int j = sj; j < sj+BLOCKSIZE; j++ ) {
 __m256d c[UNROLL];
 for ( int x = 0; x < UNROLL; x++ )
 c[x] = _mm256_load_pd(C+i+x*4+j*n);
 for( int k = sk; k < sk+BLOCKSIZE; k++ )
 {
 __m256d b = _mm256_broadcast_sd(B+k+j*n);
 for (int x = 0; x < UNROLL; x++)
 c[x] = _mm256_add_pd(c[x],
 _mm256_mul_pd(_mm256_load_pd(A+n*k+x*4+i), b));
 }
 for ( int x = 0; x < UNROLL; x++ )
 _mm256_store_pd(C+i+x*4+j*n, c[x]);
 }
}


// Está dando problema com n = 1000, mas não com 1024.
void dgemm_cache (int n, double* A, double* B, double* C)
{
 for ( int sj = 0; sj < n; sj += BLOCKSIZE )
 for ( int si = 0; si < n; si += BLOCKSIZE )
 for ( int sk = 0; sk < n; sk += BLOCKSIZE ){
 do_block_cache(n, si, sj, sk, A, B, C);}
}

void do_block_fix(int n, int si, int sj, int sk,
                      double *A, double *B, double *C)
{
    int i_end = si + BLOCKSIZE;
    int j_end = sj + BLOCKSIZE;
    int k_end = sk + BLOCKSIZE;

    // Estava dando erro com a tentativa de acesso a indices fora da matriz, corrigimos assim logo abaixo.
    // Imagina-se que isso implique em uma pequena perda de desempenho, mas foi necessário
    // para não refazer todos os experimentos de novo.
    // O código original do livro funcionaria caso a escolha de n fosse uma potência de 2, como 1024

    if (i_end > n) i_end = n;
    if (j_end > n) j_end = n;
    if (k_end > n) k_end = n;

    for (int i = si; i < i_end; i += UNROLL * 4) {
        for (int j = sj; j < j_end; j++) {
            __m256d c[UNROLL];
            for (int x = 0; x < UNROLL; x++)
                c[x] = _mm256_load_pd(C + i + x * 4 + j * n);

            for (int k = sk; k < k_end; k++) {
                __m256d b = _mm256_broadcast_sd(B + k + j * n);
                for (int x = 0; x < UNROLL; x++)
                    c[x] = _mm256_add_pd(c[x],
                        _mm256_mul_pd(_mm256_load_pd(A + n * k + x * 4 + i), b));
            }
            for (int x = 0; x < UNROLL; x++)
                _mm256_store_pd(C + i + x * 4 + j * n, c[x]);
        }
    }
}

void dgemm_cache_fix(int n, double* A, double* B, double* C)
{
    for (int sj = 0; sj < n; sj += BLOCKSIZE)
        for (int si = 0; si < n; si += BLOCKSIZE)
            for (int sk = 0; sk < n; sk += BLOCKSIZE)
                do_block_fix(n, si, sj, sk, A, B, C);
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
        dgemm_cache_fix(n, a, b, c);
        clock_t stop = clock();

        double elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC * 1000;
        times[i] = elapsed_time;
    }


    printf("\nvalores de tempo:\n");
    for(int i = 0; i < 5; i++) {
        printf(" %d: %.0f ms\n", i + 1, times[i]);
    }

    _mm_free(a);
    _mm_free(b);
    _mm_free(c);

    return 0;
}