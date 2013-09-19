include ../../Makefile.config

ALL : diffusion_cray

ifeq ($(useFPP),true)
diffusion_cray : diffusion_cray.f90
	$(CRAY_FORT) $(CRAY_OPTIONS) $< -o $@
else
diffusion_cray : diffusion_cray.F90
	$(CRAY_FORT) $(CRAY_OPTIONS) $(M_CRAY_DEFINE) $< -o $@
endif

clean:
	rm -f diffusion_cray *.o *~ *.mod *.gpu *.cubin *.ptx

