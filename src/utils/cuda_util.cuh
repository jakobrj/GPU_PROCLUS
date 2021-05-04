#include "cuda_runtime_api.h"

int getSPcores()
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n");
      break;
      }

    return cores;
}

int cuda_cores = getSPcores();

int get_block_size(int number_of_blocks){
	return 0;
}