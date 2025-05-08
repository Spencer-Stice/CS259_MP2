#include <iostream>
#include <string>
#include "dnn.hpp"
#include <iomanip>  // for std::setw

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

// #ifndef Tnn
//   //Tiling Sizes
//   #define Tnn 32
//   #define Tn  16
//   #define Ti  16
//   printf("defining Tnn\n");
//   #define Ty  8
//   #define Tx  8
// #endif

#define NYPAD   (Ny + Ky - 1)
#define NXPAD   (Nx + Kx - 1)

#define NYSCL   Ny
#define NXSCL   Nx

#define SYNAPSE_SIZE  (Ky * Kx * Nn * Ni)



VTYPE (*synapse)[Ky][Kx][Nn][Ni];

VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                    VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
  for(int yy = 0; yy < Ky; ++yy) {
    for(int xx = 0; xx < Kx; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }
  for(int yy = 0; yy < NYPAD; ++yy) {
    for(int xx = 0; xx < NXPAD; ++xx) {      
      for(int ni = 0; ni < Ni; ++ni) {
        neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
}

void convolution_layer_blocked(
                              VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                              VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                              VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  int c1=0,c2=0;
  VTYPE sum[Nn]={0};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy/Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx/Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              for (int n = nn; n < nn + Tn; n++) {
                sum[n] = 0;
              }

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni -Ti+1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc=0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc+=sv*nv;
                      }
                      sum[n]+=sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++; 
          }
          yout++;
        }
      }
    }
  }
}

void  convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                               VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                               VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}

__device__ VTYPE transfer_gpu(VTYPE i) {
  return (i > 0) ? i : i / 4;
}


// synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])aligned_malloc(64,  SYNAPSE_SIZE * sizeof(VTYPE));
// neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni]) aligned_malloc(64, NYPAD * NXPAD * Ni * sizeof(VTYPE));
// neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_malloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));

// #define BLOCK_X 8
// #define BLOCK_Y 8
// #define BLOCK_Z 16

__global__
void convKernel(
    const VTYPE* __restrict__ synapse,
    const VTYPE* __restrict__ neuron_i,
          VTYPE*       neuron_n)
{
    int sum_count = 0;
    __shared__ VTYPE sh_input[BLOCK_Y][BLOCK_X][BLOCK_Z]; // input patch
    __shared__ VTYPE sh_syn[Ky][Kx][BLOCK_Z][BLOCK_Z];                 // filter tile
    
    int gx = blockIdx.x * Tx + threadIdx.x; // global x
    int gy = blockIdx.y * Ty + threadIdx.y; // global y
    int gz = blockIdx.z * blockDim.z + threadIdx.z; // global z
    int tx = threadIdx.x; // thread x
    int ty = threadIdx.y; // thread y
    int tz = threadIdx.z; // thread z

    VTYPE sum = 0;
    for (int ii = 0; ii < Ni; ii += BLOCK_Z){
      int ic = ii + tz;

      // Load input into shared memory
      sh_input[ty][tx][tz] = neuron_i[ic + gx * (Ni) + gy * (NXPAD*Ni)];


      if (tx < Kx && ty < Ky) {
        for (int in = 0; in < BLOCK_Z; in++){
          // [Ky][Kx][Nn][Ni]
          int iic = ii + in;
         
          // Load synapse into shared memory
          sh_syn[ty][tx][tz][in] = synapse[ty * (Kx * Nn * Ni) + tx * (Nn * Ni) + gz * (Ni) + iic];
            
        }
      }

      __syncthreads();
      
      if (tx < BLOCK_X - 2 && ty < BLOCK_Y - 2){
        for(int in = 0; in < BLOCK_Z; in++){
          #pragma unroll
          for (int ky = 0; ky < Ky; ky++){
            #pragma unroll
            for (int kx = 0; kx < Kx; kx++){
              VTYPE syn_val = sh_syn[ky][kx][tz][in];
              VTYPE inp_val = sh_input[ty + ky][tx + kx][in];

              sum += syn_val * inp_val;
  
            }
          }
        }
      }
      
      __syncthreads();

  }
  if (tx < BLOCK_X - 2 && ty < BLOCK_Y - 2) {
    //[NYSCL][NXSCL][Nn]
    int out_idx = gy * (NXSCL * Nn) + gx *(Nn) + gz;
    neuron_n[out_idx] = transfer_gpu(sum);
  }
}




int main(const int argc, const char** argv) {

  cudaFree(nullptr); 


  synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])aligned_malloc(64,  SYNAPSE_SIZE * sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni]) aligned_malloc(64, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_malloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_malloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));

  // Allocate device memory
  VTYPE* d_synapse;
  VTYPE* d_neuron_i;
  VTYPE* d_neuron_n;

  cudaMalloc((void**)&d_synapse, SYNAPSE_SIZE * sizeof(VTYPE));
  cudaMalloc((void**)&d_neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  cudaMalloc((void**)&d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE));

  cout << "initializing arrays\n";
  fill_convolution_shared_simple(*synapse, *neuron_i);  // Fill the arrays on the host


  cudaEvent_t synapse_todev_start, synapse_todev_stop, neur_todev_start, neur_todev_stop, kernel_start, kernel_stop, neur_tohost_start, neur_tohost_stop;
  cudaEventCreate(&synapse_todev_start);
  cudaEventCreate(&synapse_todev_stop);
  cudaEventCreate(&neur_todev_start);
  cudaEventCreate(&neur_todev_stop);
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  cudaEventCreate(&neur_tohost_start);
  cudaEventCreate(&neur_tohost_stop);

  // Copy the initialized data from host to device
  cudaEventRecord(synapse_todev_start);
  cudaMemcpy(d_synapse, synapse, SYNAPSE_SIZE * sizeof(VTYPE), cudaMemcpyHostToDevice);
  cudaEventRecord(synapse_todev_stop);
  cudaEventSynchronize(synapse_todev_stop);
  float synapse_todev_ms = 0;
  cudaEventElapsedTime(&synapse_todev_ms, synapse_todev_start, synapse_todev_stop);
  printf("synapse copy time: %f ms\n", synapse_todev_ms);

  cudaEventRecord(neur_todev_start);
  cudaMemcpy(d_neuron_i, neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
  cudaEventRecord(neur_todev_stop);
  cudaEventSynchronize(neur_todev_stop);
  float neur_todev_ms = 0;
  cudaEventElapsedTime(&neur_todev_ms, neur_todev_start, neur_todev_stop);
  printf("neuron_i copy time: %f ms\n", neur_todev_ms);

  cout << "starting computation\n";

    // Simple Version (on host)
  // begin_roi();
  // convolution_layer(*synapse, *neuron_i, *neuron_n2);
  // end_roi();
  // cout << "simple version complete!\n";

  // //Blocked Version (on host)
  // begin_roi();
  // convolution_layer_blocked(*synapse, *neuron_i, *neuron_n2);
  // end_roi();

  cout << "CPU version complete!\n";



  // compute launch parameters
  // dim3 block(BLOCK_X, BLOCK_Y, 16);
  // dim3 grid(56, 56, 4);

  dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 grid(Nx/Tx, Ny/Ty, Nn/BLOCK_Z);

  // launch
  //convolution_layer_kernel<<<grid, block>>>(d_synapse, d_neuron_i, d_neuron_n);

  cudaEventRecord(kernel_start);
  convKernel<<<grid, block>>>(d_synapse, d_neuron_i, d_neuron_n);
  cudaEventRecord(kernel_stop);
  cudaEventSynchronize(kernel_stop);
  float kernel_ms = 0;
  cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
  printf("kernel time: %f ms\n", kernel_ms);
  cudaError_t err = cudaGetLastError();
  printf("Kernel complete!\n");

  if (err != cudaSuccess) {
      printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
  }

  // Copy result back to host
  cudaEventRecord(neur_tohost_start);
  cudaMemcpy(*neuron_n, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
  cudaEventRecord(neur_tohost_stop);
  cudaEventSynchronize(neur_tohost_stop);
  float neur_tohost_ms = 0;
  cudaEventElapsedTime(&neur_tohost_ms, neur_tohost_start, neur_tohost_stop);
  //printf("neuron_n copy time: %f ms\n", neur_tohost_ms);
  // printf("neuron_n[0] = %f\n", (*neuron_n)[0]);


  // Compare results (host vs device)
  //compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n2, NYSCL * NXSCL * Nn);
  printf("Success!");
  //compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n2, 224);

  // Free device memory
  cudaFree(d_synapse);
  cudaFree(d_neuron_i);
  cudaFree(d_neuron_n);
}



