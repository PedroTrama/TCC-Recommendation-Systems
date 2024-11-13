#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include "kernel.h"

// Constants
#define USERS 1000
#define MOVIES 4489
#define FACTORS 8
#define CLUSTERS 10
#define MAX_ITER 100

struct Movie {
	int id;
	float similarity;
};

// Read CSV files
void read_csv_data(const char* filename, int* ids, float* values, int* len) {
	FILE* file = fopen(filename, "r");
	if (!file) {
    	printf("Error opening %s\n", filename);
    	exit(1);
	}

	int i = 0;
	while (fscanf(file, "%d,%d,%f", &ids[i], &ids[i + 1], &values[i / 2]) != EOF) {
    	i += 2;
	}
	*len = i / 2;
	fclose(file);
}

// Cosine similarity between movie and centroid
float cosine_similarity(float* movie, float* centroid, int factors) {
	float dot_product = 0.0, norm_movie = 0.0, norm_centroid = 0.0;
	for (int i = 0; i < factors; i++) {
    	dot_product += movie[i] * centroid[i];
    	norm_movie += movie[i] * movie[i];
    	norm_centroid += centroid[i] * centroid[i];
	}
	return dot_product / (sqrt(norm_movie) * sqrt(norm_centroid) + 1e-8);
}

// KMeans clustering with top 10 most similar movies
void kmeans(float* Q, int* labels, int movies, int factors) {
	float** centroids = (float**)malloc(CLUSTERS * sizeof(float*));
	for (int i = 0; i < CLUSTERS; i++) {
    	centroids[i] = (float*)malloc(factors * sizeof(float));
	}
	int* counts = (int*)malloc(CLUSTERS * sizeof(int));

	for (int c = 0; c < CLUSTERS; c++) {
    	int movie_index = c * (movies / CLUSTERS);
    	for (int f = 0; f < factors; f++) {
        	centroids[c][f] = Q[movie_index * factors + f];
    	}
	}

	for (int iter = 0; iter < MAX_ITER; iter++) {
    	memset(counts, 0, CLUSTERS * sizeof(int));

    	for (int m = 0; m < movies; m++) {
        	float min_dist = FLT_MAX;
        	int cluster = 0;

        	for (int c = 0; c < CLUSTERS; c++) {
            	float dist = 0.0;
            	for (int f = 0; f < factors; f++) {
                	float diff = Q[m * factors + f] - centroids[c][f];
                	dist += diff * diff;
            	}

            	if (dist < min_dist) {
                	min_dist = dist;
                	cluster = c;
            	}
        	}

        	labels[m] = cluster;
        	counts[cluster]++;
    	}

    	for (int c = 0; c < CLUSTERS; c++) {
        	memset(centroids[c], 0, factors * sizeof(float));
    	}

    	for (int m = 0; m < movies; m++) {
        	int cluster = labels[m];
        	for (int f = 0; f < factors; f++) {
            	centroids[cluster][f] += Q[m * factors + f];
        	}
    	}

    	for (int c = 0; c < CLUSTERS; c++) {
        	if (counts[c] == 0) continue;
        	for (int f = 0; f < factors; f++) {
            	centroids[c][f] /= counts[c];
        	}
    	}
	}

	for (int c = 0; c < CLUSTERS; c++) {
    	printf("Cluster #%d:\n", c);

    	std::vector<Movie> top_movies;

    	for (int m = 0; m < movies; m++) {
        	if (labels[m] == c) {
            	float similarity = cosine_similarity(&Q[m * factors], centroids[c], factors);
            	top_movies.push_back({m, similarity});
        	}
    	}

    	std::sort(top_movies.begin(), top_movies.end(), [](const Movie& a, const Movie& b) {
        	return a.similarity > b.similarity;
    	});

    	for (int i = 0; i < 10 && i < top_movies.size(); i++) {
        	printf("\tMovie %d (Similarity: %.4f)\n", top_movies[i].id, top_movies[i].similarity);
    	}
	}

	for (int i = 0; i < CLUSTERS; i++) {
    	free(centroids[i]);
	}
	free(centroids);
	free(counts);
}

// Main
int main() {
	float* Q = (float*)malloc(MOVIES * FACTORS * sizeof(float));
	if (Q == NULL) {
    	printf("Memory allocation for Q failed!\n");
    	return 1;
	}

	int* labels = (int*)malloc(MOVIES * sizeof(int));
	if (labels == NULL) {
    	printf("Memory allocation for labels failed!\n");
    	free(Q);
    	return 1;
	}

	printf("Initializing movie embeddings...\n");
	for (int i = 0; i < MOVIES; i++) {
    	for (int f = 0; f < FACTORS; f++) {
        	Q[i * FACTORS + f] = (float)(rand() % 100) / 100.0f;
    	}
	}

	printf("Running KMeans clustering...\n");
	kmeans(Q, labels, MOVIES, FACTORS);

	free(Q);
	free(labels);

	printf("KMeans clustering completed successfully!\n");
	return 0;
}
