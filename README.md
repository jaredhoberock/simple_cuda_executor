# simple_cuda_executor

This program demonstrates how a simple GPU executor uses a CUDA `__shared__`
variable as storage for a shared parameter. The example doesn't actually do
anything very interesting, but it should illustrate how the special semantics
of the factories are used to communicate where the corresponding parameter may
be located.

To build:

    $ nvcc -std=c++11 --expt-extended-lambda -arch=sm_30 simple_cuda_executor.cu -o simple_cuda_executor

Expected output:

    $ ./simple_cuda_executor
    Hello world, from agent 0
    Hello world, from agent 1
    Hello world, from agent 2
    Hello world, from agent 3
    Hello world, from agent 4
    Hello world, from agent 5
    Hello world, from agent 6
    Hello world, from agent 7
    Hello world, from agent 8
    Hello world, from agent 9
    Hello world, from agent 10
    Hello world, from agent 11
    Hello world, from agent 12
    Hello world, from agent 13
    Hello world, from agent 14
    Hello world, from agent 15
    OK
