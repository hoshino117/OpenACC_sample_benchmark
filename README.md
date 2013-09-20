OpenACC_sample_benchmark
========================

This is OpenACC sample benchmarks. 
Matrix Multiplication and 7-point Stencil computation is now available.

Comipile
> cd OpenACC_sample_benchmark
> ./setup.sh
> make

BMT Run
> cd matmul(or stencil)/pgi(or caps,cray,cuda)
> ./matmul(or diffusion)_pgi(or caps,cray,cuda)

If you want to change BMT parameters, please modify Makefile.conifg.
