#include <iostream>
#include "dnn.hpp"

using namespace std;


// #ifndef Tii
//   // Tiling Sizes
//   #define Tnn 32  
//   #define Tii 32
//   //#define Tn 5
//   //#define Ti 25
//   #define Tn 16
//   #define Ti 16
// #endif

//Arrays:
VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Ni] __attribute__((aligned(64)));
VTYPE neuron_n[Nn] __attribute__((aligned(64))),    neuron_n2[Nn] __attribute__((aligned(64)));

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
    VTYPE (&neuron_n)[Nn],   VTYPE (&neuron_n2)[Nn]) {
  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      synapse[n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for(int i = 0; i < Ni; ++i) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for(int n = 0; n < Nn; ++n) {
    neuron_n[n] = 0; //i;
    neuron_n2[n] = 0; //i;
  }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
  int total_calc=0;
  for (int n = 0; n < Nn; n++) {
    VTYPE temp=0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

void classifier_layer_blocked(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
                              VTYPE (&neuron_n)[Nn]) {
  int total_calc=0;
  VTYPE sum[Nn]={0};
  for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
    for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
      for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
        for (int ii = iii; ii < iii + Tii; ii += Ti) {
          // — Original code —
          for (int n = nn; n < nn + Tn; n++) {
            VTYPE sum_sc=0;
            for (int i = ii; i < ii + Ti; i++) {
              sum_sc += (synapse[n][i] * neuron_i[i]);
            }
            sum[n]+=sum_sc;
          }
        }
      }
    }
    for (int nn = nnn; nn < nnn + Tnn; nn++) {
      neuron_n[nn] = transfer(sum[nn]);
    }
  }
}

__device__ VTYPE transfer_gpu(VTYPE i) {
  return (i > 0) ? i : i / 4;
}

__global__ void classifier_layer_kernel(VTYPE *synapse, VTYPE *neuron_i, VTYPE *neuron_n) {


  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  __shared__ VTYPE neuron_i_s[Ti];
  VTYPE sum = 0;

  for (int ii = 0; ii < Ni; ii+=Ti) {
    for (int i = ii + tx; i < ii + Ti; i+=blockDim.x) {
      if (i < Ni) {
        neuron_i_s[i - ii] = neuron_i[i];
      }
    }

    __syncthreads();

    for (int j = 0; j < Ti && (ii + j) < Ni; j++) {
      sum += synapse[gx * Ni + ii + j] * neuron_i_s[j];
    }

    __syncthreads();
  } 
  if (gx < Nn)
    neuron_n[gx] = transfer_gpu(sum);
}





int main(int argc, char** argv) {
  cout << "initializing arrays\n";

  fill_classifier(synapse, neuron_i, neuron_n, neuron_n2);

  // Allocate GPU memory
  VTYPE *synapse_d, *neuron_i_d, *neuron_n_d;
  cudaMalloc(&synapse_d, sizeof(VTYPE) * Nn * Ni);
  cudaMalloc(&neuron_i_d, sizeof(VTYPE) * Ni);
  cudaMalloc(&neuron_n_d, sizeof(VTYPE) * Nn);

  // Copy inputs to GPU
  cudaMemcpy(synapse_d, synapse, sizeof(VTYPE) * Nn * Ni, cudaMemcpyHostToDevice);
  cudaMemcpy(neuron_i_d, neuron_i, sizeof(VTYPE) * Ni, cudaMemcpyHostToDevice);

  cout << "starting GPU computation\n";
  begin_roi();
  {
    int blockSize = 256;
    int gridSize = (Nn + blockSize - 1) / blockSize;
    classifier_layer_kernel<<<gridSize, blockSize>>>(synapse_d, neuron_i_d, neuron_n_d);
    cudaDeviceSynchronize();
  }
  end_roi();
  cout << "GPU version complete!\n";

  // Copy result back to host
  cudaMemcpy(neuron_n, neuron_n_d, sizeof(VTYPE) * Nn, cudaMemcpyDeviceToHost);

  cout << "starting blocked CPU computation\n";
  begin_roi();
  classifier_layer_blocked(synapse, neuron_i, neuron_n2);
  end_roi();
  cout << "blocked computation complete!\n";

  compare(neuron_n, neuron_n2, Nn);

  // Cleanup
  cudaFree(synapse_d);
  cudaFree(neuron_i_d);
  cudaFree(neuron_n_d);
}



