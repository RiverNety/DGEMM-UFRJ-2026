#include <x86intrin.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define UNROLL 8
#define BLOCKSIZE 32
#define ALIGNMENT 64 

// gcc -mavx2 -o dgemm.exe '4.2. dgemm cache block livro anterior.c'
void do_block (int n, int si, int sj, int sk,
 double *A, double *B, double *C)
{
 for ( int i = si; i < si+BLOCKSIZE; i+=UNROLL*4 )
 for ( int j = sj; j < sj+BLOCKSIZE; j++ ) {
 __m256d c[4];
 for ( int x = 0; x < UNROLL; x++ )
 c[x] = _mm256_load_pd(C+i+x*4+j*n);
 for( int k = sk; k < sk+BLOCKSIZE; k++ )
 {
 __m256d b = _mm256_broadcast_sd(B+k+j*n);
 for (int x = 0; x < UNROLL; x++)
 c[x] = _mm256_add_pd(c[x], /* c[x]+=A[i][k]*b */
 _mm256_mul_pd(_mm256_load_pd(A+n*k+x*4+i), b));
 }
 for ( int x = 0; x < UNROLL; x++ )
 _mm256_store_pd(C+i+x*4+j*n, c[x]);
 }

}
void dgemm (int n, double* A, double* B, double* C)
{
 for ( int sj = 0; sj < n; sj += BLOCKSIZE )
 for ( int si = 0; si < n; si += BLOCKSIZE )
 for ( int sk = 0; sk < n; sk += BLOCKSIZE )
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

    _mm_free(a);
    _mm_free(b);
    _mm_free(c);

    return 0;
}