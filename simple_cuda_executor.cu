#include <cassert>
#include <cstdio>
#include <iostream>

template<class Function, class T, class SharedFactory>
__global__ void kernel(Function f, T* ptr_to_result, SharedFactory shared_factory)
{
  using shared_type = typename std::result_of<SharedFactory()>::type;

  extern __shared__ shared_type ptr_to_shared[];
  
  // wait for all agents to reach this point before calling the shared factory
  __syncthreads();

  // the first thread invokes the shared factory
  if(threadIdx.x == 0)
  {
    // copy construct the shared parameter from the factory
    ::new (ptr_to_shared) shared_type(shared_factory());
  }

  // wait for all agents to reach this point before invoking f
  __syncthreads();

  // all threads invoke f
  f(threadIdx.x, *ptr_to_result, *ptr_to_shared);

  // wait for all agents to finish with f's invocation before destroying the shared parameter
  __syncthreads();

  // the first thread destroys the shared parameter
  if(threadIdx.x == 0)
  {
    ptr_to_shared->~shared_type();
  }
}

// a cuda_thread_block_executor's execution function creates a group of execution agents
// which execute on a single CUDA thread block
struct cuda_thread_block_executor 
{
  // using execution_category = concurrent_execution_tag;

  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory()>::type
  bulk_sync_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
  {
    // allocate storage for the result
    using result_type = typename std::result_of<ResultFactory()>::type;
    result_type* ptr_to_result = nullptr;
    cudaMallocManaged(&ptr_to_result, sizeof(result_type), cudaMemAttachGlobal);

    // invoke result factory and store it
    *ptr_to_result = result_factory();

    using shared_type = typename std::result_of<SharedFactory()>::type;

    // launch a kernel to create a group of execution agents
    kernel<<<1, n, sizeof(shared_type)>>>(f, ptr_to_result, shared_factory);

    // synchronize to wait for the execution agents to finish
    cudaDeviceSynchronize();

    // copy the result back to the host
    result_type result = *ptr_to_result;

    // deallocate the temporary result's storage
    cudaFree(ptr_to_result);

    return std::move(result);
  }
};

int main()
{
  cuda_thread_block_executor my_executor;

  auto result = my_executor.bulk_sync_execute([] __host__ __device__ (int idx, int& result, int& shared)
  {
    printf("Hello world, from agent %d\n", idx);

    // have the 8th thread add the shared parameter to the result
    if(idx == 8)
    {
      result += shared;
    }
  },
  16,
  [] __host__ __device__ () { return 7; }, // result factory
  [] __host__ __device__ () { return 13; } // shared factory
  );

  assert(result == 20);

  std::cout << "OK" << std::endl;

  return 0;
}

