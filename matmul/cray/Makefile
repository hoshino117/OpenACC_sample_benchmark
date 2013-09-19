include ../../Makefile.config

ALL : matmul_cray

ifeq ($(useFPP),true)
matmul_cray : matmul_cray.f90
	$(CRAY_FORT) $(CRAY_OPTIONS) $< -o $@
else
matmul_cray : matmul_cray.F90
	$(CRAY_FORT) $(CRAY_OPTIONS) $(M_CRAY_DEFINE) $< -o $@
endif

clean:
	rm -f matmul_cray *.o *~ *.mod *.gpu *.cubin *.ptx

