#include <chrono>
#include "../include/utils.h"
#include <algorithm>

#define NUM_RUNS 2

#define CHECK(name) \
  std::cout << "checking " << #name << std::endl;		\
  initialize(refC, Ref::M * Ref::N);				\
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);		\
  if (!ref.checkRef(refC)){					\
    std::cerr << #name << ": check ref failed!" << std::endl;	\
  };								
  
#define TIME(name) \
  for (int i = 0; i < 1; i++)						\
    {									\
      name(A, B, C, M, N, K);						\
    }									\
  std::chrono::duration<double, std::milli> time_##name(0);		\
  for (int i = 0; i < NUM_RUNS; i++)					\
    {									\
      initialize(C, M * N);						\
      auto start_time_ ## name = std::chrono::high_resolution_clock::now(); \
      name(A, B, C, M, N, K);						\
      auto end_time_ ## name = std::chrono::high_resolution_clock::now(); \
      time_ ## name += end_time_ ## name - start_time_ ## name;		\
    }									\
std::chrono::duration<double, std::milli> duration_ ## name = time_ ## name/float(NUM_RUNS); \
  std::cout << "Time taken for GEMM (CPU," << #name <<"): " << duration_ ## name.count() << "ms" << std::endl; 


// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
void gemm_cpu_o0(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
	C[i * N + j]  += A[i * K + k]  * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler flags. So, you can use o3's code for that part
void gemm_cpu_o1(float* A, float* B, float *C, int M, int N, int K) {

	// we want to increment along contiguous memory locations, since i is always multiplied by something, 
	// we want it on the outside
	for (int i = 0; i < M; i++) {

		// we do increment by just k in one place, but we also jump in k steps of N in another, so it'll go in the middle
		for (int k = 0; k < K; k++) {

			// since we increment by just j in two places, placing it on the innermost loop will hopefully keep those arrays in the cache
			for (int j = 0; j < N; j++) {
				C[i * N + j]  += A[i * K + k]  * B[k * N + j];
			}
		}
	}
}

void gemm_cpu_o2(float* A, float* B, float *C, int M, int N, int K) {
	// cache size of the colab machine is 
	// L1d cache:                               32 KiB (1 instance)
	// L1i cache:                               32 KiB (1 instance)
	// We care about the L1d cache in this instance (although they are identical here),
	// so 32KiB is our size. The machine is a 64-bit machine, so each int is 4 bytes
	// we have three arrays we want to keep in the cache each array traverses
	// step * step * sizeof(float) each inner loop iteration
	// so we want (step^2 * 4) * 3 < 32KiB -> step < 51.6

	int step = 40;
	
	// spec says to only tile inner two loops
	for (int i = 0; i < M; i++) {
		for (int jj = 0; jj < N; jj += step) {
			for (int kk = 0; kk < K; kk += step) {
				for (int k = kk; k < std::min(kk + step, K); k++) {
					for (int j = jj; j < std::min(jj + step, N); j++) {
						C[i * N + j] += A[i * K + k] * B[k * N + j];
					}
				}
			}
		}
	}
}

void gemm_cpu_o3(float* A, float* B, float *C, int M, int N, int K) {
	int step = 50;
	
	// Parallelize the outer loop(s) using OpenMP
	#pragma omp parallel for
	for (int i = 0; i < M; i++) {
		#pragma omp parallel for
		for (int kk = 0; kk < K; kk += step) {
			#pragma omp parallel for
			for (int jj = 0; jj < N; jj += step) {

				#pragma omp parallel for
				for (int k = kk; k < std::min(kk + step, K); k++) {
					// vectorize the inner loop
					#pragma omp simd
					for (int j = jj; j < std::min(jj + step, N); j++) {
						C[i * N + j]  += A[i * K + k]  * B[k * N + j];
					}
				}
			}
		}
	}
}


int main(int argc, char* argv[]) {
	if (argc < 3) {
	  std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
	  return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	// Check if the kernel results are correct
	// note that even if the correctness check fails all optimized kernels will run.
	// We are not exiting the program at failure at this point.
	// It is a good idea to add more correctness checks to your code.
	// We may (at discretion) verify that your code is correct.
	float* refC = new float[Ref::M * Ref::N]();
	auto ref = Ref();
	CHECK(gemm_cpu_o0)
	CHECK(gemm_cpu_o1)
	CHECK(gemm_cpu_o2)
	CHECK(gemm_cpu_o3)
	delete[] refC;
	
	TIME(gemm_cpu_o0)
	TIME(gemm_cpu_o1)
	TIME(gemm_cpu_o2)
	TIME(gemm_cpu_o3)

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}