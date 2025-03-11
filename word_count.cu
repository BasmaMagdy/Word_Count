#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// CUDA kernel for tokenizing text
__global__ void count_words(char *chunk, int chunk_size, int *word_count, int *delimiter_positions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < chunk_size) {
        char c = chunk[idx];
        // Mark delimiters (spaces, newlines, etc.)
        if (c == ' ' || c == '\n' || c == '\t' || c == ',' || c == '.' || c == ';' || c == ':') {
            delimiter_positions[idx] = 1;
        } else {
            delimiter_positions[idx] = 0;
        }
    }

    // Count words using delimiter transitions
    __syncthreads();
    if (idx < chunk_size - 1 && delimiter_positions[idx] == 1 && delimiter_positions[idx + 1] == 0) {
        atomicAdd(word_count, 1);
    }
}

// Function to read a large text file into a buffer
char *read_file(const char *filename, int *size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc(*size * sizeof(char));
    fread(buffer, sizeof(char), *size, file);
    fclose(file);
    return buffer;
}

// Sequential word count implementation
int sequential_word_count(char *data, int size) {
    int word_count = 0;
    int in_word = 0;

    for (int i = 0; i < size; i++) {
        if (data[i] == ' ' || data[i] == '\n' || data[i] == '\t' || data[i] == ',' || data[i] == '.' || data[i] == ';') {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            word_count++;
        }
    }

    return word_count;
}

// MPI-only word count
int mpi_word_count(int rank, int size, char *file_data, int total_size) {
    int chunk_size = total_size / size;
    char *local_chunk = (char *)malloc(chunk_size * sizeof(char));

    MPI_Scatter(file_data, chunk_size, MPI_CHAR, local_chunk, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    int local_word_count = sequential_word_count(local_chunk, chunk_size);
    int global_word_count = 0;
    MPI_Reduce(&local_word_count, &global_word_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(local_chunk);
    return global_word_count;
}

// OpenMP word count implementation
int openmp_word_count(char *data, int size) {
    int word_count = 0;
    #pragma omp parallel for reduction(+:word_count)
    for (int i = 0; i < size; i++) {
        if ((i == 0 || data[i - 1] == ' ' || data[i - 1] == '\n' || data[i - 1] == '\t' || data[i - 1] == ',' || data[i - 1] == '.') &&
            !(data[i] == ' ' || data[i] == '\n' || data[i] == '\t' || data[i] == ',' || data[i] == '.')) {
            word_count++;
        }
    }
    return word_count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_size = 0;
    char *file_data = NULL;

    if (rank == 0) {
        file_data = read_file("book.txt", &total_size);
    }

    MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    double start_time, end_time;

    // Sequential Implementation (root process only)
    if (rank == 0) {
        start_time = MPI_Wtime();
        int seq_word_count = sequential_word_count(file_data, total_size);
        end_time = MPI_Wtime();
        printf("Sequential word count:       %d, Time: %.6f seconds\n", seq_word_count, end_time - start_time);
    }

    // MPI-only Implementation
    start_time = MPI_Wtime();
    int mpi_result = mpi_word_count(rank, size, file_data, total_size);
    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("MPI word count:              %d, Time: %.6f seconds\n", mpi_result, end_time - start_time);
    }

    // OpenMP Implementation (root process only)
    if (rank == 0) {
        start_time = MPI_Wtime();
        int omp_result = openmp_word_count(file_data, total_size);
        end_time = MPI_Wtime();
        printf("OpenMP word count:           %d, Time: %.6f seconds\n", omp_result, end_time - start_time);
    }

    // Fully Parallel Implementation with CUDA and MPI
    int chunk_size = total_size / size;
    char *local_chunk = (char *)malloc(chunk_size * sizeof(char));
    MPI_Scatter(file_data, chunk_size, MPI_CHAR, local_chunk, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    char *d_chunk;
    int *d_word_count, *d_delimiters;
    int word_count = 0;
    cudaMalloc((void **)&d_chunk, chunk_size * sizeof(char));
    cudaMalloc((void **)&d_word_count, sizeof(int));
    cudaMalloc((void **)&d_delimiters, chunk_size * sizeof(int));

    cudaMemcpy(d_chunk, local_chunk, chunk_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_count, &word_count, sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (chunk_size + threads_per_block - 1) / threads_per_block;

    start_time = MPI_Wtime();
    count_words<<<blocks_per_grid, threads_per_block>>>(d_chunk, chunk_size, d_word_count, d_delimiters);
    cudaDeviceSynchronize();
    cudaMemcpy(&word_count, d_word_count, sizeof(int), cudaMemcpyDeviceToHost);

    int global_word_count = 0;
    MPI_Reduce(&word_count, &global_word_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Fully parallel word count:   %d, Time: %.6f seconds\n", global_word_count, end_time - start_time);
    }

    cudaFree(d_chunk);
    cudaFree(d_word_count);
    cudaFree(d_delimiters);
    free(local_chunk);
    if (rank == 0) free(file_data);

    MPI_Finalize();
    return 0;
}
