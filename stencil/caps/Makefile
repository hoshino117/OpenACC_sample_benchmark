include ../../Makefile.config

ALL : diffusion_caps

ifeq ($(useFPP),true)
diffusion_caps : diffusion_caps.f90
	$(CAPS_FORT) $(CAPS_OPTIONS) $(CAPS_HOST) $(CAPS_HOST_OPTIONS) $< -o $@
else
diffusion_caps : diffusion_caps.F90
	$(CAPS_FORT) $(CAPS_OPTIONS) $(CAPS_HOST) $(CAPS_HOST_OPTIONS) $(M_CAPS_DEFINE) $< -o $@
endif

clean:
	rm -f diffusion_caps *.o *~ *.mod *.gpu *.cubin *.ptx *.hmf* *.hmd

