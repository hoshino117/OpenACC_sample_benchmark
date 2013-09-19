include ../../Makefile.config

ALL : diffusion_cuda

diffusion_cuda : diffusion_cuda.cuf
	$(CUDA_FORT) $(CUDA_OPTIONS) -Mpreprocess $(S_CUDA_DEFINE) $< -o $@

clean:
	rm -f *.o diffusion_cuda *.mod



