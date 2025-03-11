Languages used in the project --> C, MPI, OPENMP, CUDA

Compiling and running of the code 
![project_screen](https://github.com/user-attachments/assets/b6a5d353-6153-4df0-b830-c3315c31cf2f)

Explanation of the project
This project is a word count project after reading the content of text file. The
implementation of a word count solution utilizing MPI, OpenMP, and CUDA. It combines
these parallel programming paradigms to achieve efficient word counting on various
hardware architectures. The project aims to demonstrate the effectiveness of hybrid
approaches in optimizing word count performance.
Applying comparison of sequential, partial parallel, and
fully parallel implementations:
1. Sequential: Sequential Word Count Function (sequential_word_count) this
function I have implemented a loop to count words in a given text buffer
sequentially.
2. Partial parallel: This part has been occurred while using mpi and openmp,
where:
• MPI uses MPI_Scatter to divide the text into chunks and each one sent to
process, each process performs sequential word count on its chunk using
sequential_word_count function and MPI_Reduce that combines the local
word counts into a global result at the root process.
MPI implementation is parallel at the level of multiple processes, but each
process performs its computation sequentially.
• OpenMP is used for multithreading execution while using #pragma line for
parallel. Threads independently process sections of the text and combine
results.
OpenMP implementation provides parallelism within a single node.
3. Fully parallel: This part has been occurred while using MPI and CUDA together,
where:
• MPI distributes the text chunks among processes using MPI_Scatter and
gather results using MPI_Reduce.
• CUDA, here the GPU kernel (count_word) processes each chunk to count
words by marking delimiter position and identifies word boundaries in
parallel and utilizing atomic operations (atomicAdd) to increment the
word count on the GPU. So threads within a GPU block work in parallel on
the chunk.• Synchronization, each process independently computes its word count
using CUDA, and the global word count is obtained by combining the
results from all processes using MPI_Reduce.
