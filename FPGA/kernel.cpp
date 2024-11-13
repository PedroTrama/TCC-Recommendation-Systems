#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "kernel.h"

#define FACTORS 8
#define EPOCHS 10
#define BATCH_SIZE 128
#define ALPHA 0.01
#define BETA 0.02
#define CLUSTERS 10
#define MAX_ITER 100

// Matrix factorization kernel for training
 extern "C" {
void krnl_matrix_factorization(
	float* P, float* Q, float* R,
	int users, int movies, int factors)
{
	#pragma HLS INTERFACE m_axi port=P bundle=gmem0
	#pragma HLS INTERFACE m_axi port=Q bundle=gmem1
	#pragma HLS INTERFACE m_axi port=R bundle=gmem2
	#pragma HLS INTERFACE s_axilite port=users bundle=control
	#pragma HLS INTERFACE s_axilite port=movies bundle=control
	#pragma HLS INTERFACE s_axilite port=factors bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	for (int epoch = 0; epoch < EPOCHS; epoch++) {
    	#pragma HLS DATAFLOW

    	#pragma HLS loop_tripcount min=500 max=2000
    	for (int u = 0; u < users; u++) {

        	#pragma HLS loop_tripcount min=1000 max=5000
        	for (int m = 0; m < movies; m++) {
            	float error = R[u * movies + m];

            	float P_local[FACTORS];
            	float Q_local[FACTORS];

            	#pragma HLS PIPELINE II=1
            	#pragma HLS loop_tripcount min=4 max=10
            	for (int f = 0; f < factors; f++) {
                	P_local[f] = P[u * factors + f];
                	Q_local[f] = Q[m * factors + f];
                	error -= P_local[f] * Q_local[f];
            	}

            	for (int f = 0; f < factors; f++) {
                	P[u * factors + f] += ALPHA * (2 * error * Q_local[f] - BETA * P_local[f]);
                	Q[m * factors + f] += ALPHA * (2 * error * P_local[f] - BETA * Q_local[f]);
            	}
        	}
    	}
	}
}
}