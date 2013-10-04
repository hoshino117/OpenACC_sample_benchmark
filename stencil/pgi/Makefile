include ../../Makefile.config

ALL : diffusion_pgi

ifeq ($(useFPP),true)
diffusion_pgi : diffusion_pgi.f90
	$(PGI_FORT) $(PGI_OPTIONS) $< -o $@
else
diffusion_pgi : diffusion_pgi.F90
	$(PGI_FORT) $(PGI_OPTIONS) $(S_PGI_DEFINE) $< -o $@
endif

clean:
	rm -f diffusion_pgi *.o *~ *.mod *.gpu *.cubin *.ptx

