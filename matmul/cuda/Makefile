include ../../Makefile.config

ALL : matmul_cuda

matmul_cuda : matmul_cuda.cuf
	$(CUDA_FORT) $(CUDA_OPTIONS) -Mpreprocess $(M_CUDA_DEFINE) $< -o $@

clean:
	rm -f *.o matmul_cuda *.mod



