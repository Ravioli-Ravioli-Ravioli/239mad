#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>

#define RNC 100

__global__ void kernel_1t1e(int A[RNC][RNC], int B[RNC][RNC], int C[RNC][RNC], int size){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if (i < size && j < size){
	   A[i][j] = B[i][j] + C[i][j];
   }
}

__global__ void kernel_1t1r(int A[RNC][RNC], int B[RNC][RNC], int C[RNC][RNC], int size){
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   for (int k = 0; k < size; k++){
	if (j < size){
	     A[k][j] = B[k][j] + C[k][j];	
	}
   }
}

__global__ void kernel_1t1c(int A[RNC][RNC], int B[RNC][RNC], int C[RNC][RNC], int size){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   for (int k = 0; k < size; k++){
        if (i < size){
             A[i][k] = B[i][k] + C[i][k]; 
        }
   }
}

int main(void) {
  int nDevices, i, j, B[RNC][RNC], C[RNC][RNC], A[RNC][RNC], (*A_d)[RNC], (*B_d)[RNC], (*C_d)[RNC] ;

//////////////////////////////////////Print device properties

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  MaxThreadPerBlock: %d\n",
           prop.maxThreadsPerBlock);
//    printf("  MultiProcessorCount: %d\n",
//           prop.multiProcessorCount);
//    printf("  ClockRate: %d\n",
//           prop.clockRate);
    printf("  MaxThreadsDim0: %d\n",
           prop.maxThreadsDim[0]);
    printf("  MaxThreadsDim1: %d\n",
           prop.maxThreadsDim[1]);
    printf("  MaxThreadsDim2: %d\n",
           prop.maxThreadsDim[2]);
    printf("  MaxGridSize: %d\n",
           prop.maxGridSize[1]);
//    printf("  Memory Clock Rate (KHz): %d\n",
//           prop.memoryClockRate);
//    printf("  Memory Bus Width (bits): %d\n",
//           prop.memoryBusWidth);
//    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
//           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
/////////////////////////////////////Populate First Matrix
   srand(1);
   for (i = 0; i < RNC; i++){
      for (j = 0; j < RNC; j++) {
         B[i][j] = rand() % 100 + 1; 
         printf("%d  ",B[i][j]);
      }
      printf("\n");
   }
   printf("\n");
/////////////////////////////////////Populate Second Matrix
   for (i = 0; i < RNC; i++){
      for (j = 0; j < RNC; j++) {
         C[i][j] = rand() % 100 + 1;
         printf("%d  ",C[i][j]);
      }
      printf("\n");
   }
   printf("\n");
   printf("===============================");
   printf("\n");
/////////////////////////////////////Allocate memory in the device
   
   cudaMalloc((void**) &A_d, (RNC*RNC)*sizeof(int));
   cudaMalloc((void**) &B_d, (RNC*RNC)*sizeof(int));
   cudaMalloc((void**) &C_d, (RNC*RNC)*sizeof(int));

////////////////////////////////////Mem copy from host to device
   cudaMemcpy(A_d, A, (RNC*RNC)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(B_d, B, (RNC*RNC)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(C_d, C, (RNC*RNC)*sizeof(int), cudaMemcpyHostToDevice);

   dim3 threadsPerBlock(RNC, RNC);
   dim3 numBlocks(RNC / threadsPerBlock.x, RNC / threadsPerBlock.y);

   cudaEvent_t start, stop;
   float elapsed = 0;

////////////////////////////////////ThreadAll
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   kernel_1t1e<<<numBlocks,threadsPerBlock>>>(A_d, B_d, C_d, RNC);
   cudaEventRecord(stop, 0);

   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("GPU Run TIme threadsall %.2f ms \n", elapsed);
////////////////////////////////////Thread Row
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   kernel_1t1r<<<numBlocks,threadsPerBlock>>>(A_d, B_d, C_d, RNC);
   cudaEventRecord(stop, 0);

   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("GPU Run TIme threadsrow %.2f ms \n", elapsed);
////////////////////////////////////Thread Column
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   kernel_1t1c<<<numBlocks,threadsPerBlock>>>(A_d, B_d, C_d, RNC);
   cudaEventRecord(stop, 0);

   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("GPU Run TIme threadscol %.2f ms \n", elapsed);
//////////////////////////////////////Mem Copy
   cudaMemcpy(A, A_d, (RNC*RNC)*sizeof(int), cudaMemcpyDeviceToHost);

/////////////////////////////////////Print matrix A
/*
   for (i = 0; i < RNC; i++){
      for (j = 0; j < RNC; j++) {
         printf("%d  ", A[i][j]);
      }
      printf("\n");
   }
   printf("\n");
*/
/////////////////////////////////////Free up memory
   cudaFree(A_d); 
   cudaFree(B_d); 
   cudaFree(C_d);
}
