include ../../Makefile.config

ALL : matmul_caps

ifeq ($(useFPP),true)
matmul_caps : matmul_caps.f90
	$(CAPS_FORT) $(CAPS_OPTIONS) $(CAPS_HOST) $(CAPS_HOST_OPTIONS) $< -o $@
else
matmul_caps : matmul_caps.F90
	$(CAPS_FORT) $(CAPS_OPTIONS) $(CAPS_HOST) $(CAPS_HOST_OPTIONS) $(M_CAPS_DEFINE) $< -o $@
endif

clean:
	rm -f matmul_caps *.o *~ *.mod *.gpu *.cubin *.ptx *.hmf* *.hmd

