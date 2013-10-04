include ../../Makefile.config

ALL : matmul_pgi

ifeq ($(useFPP),true)
matmul_pgi : matmul_pgi.f90
	$(PGI_FORT) $(PGI_OPTIONS) $< -o $@
else
matmul_pgi : matmul_pgi.F90
	$(PGI_FORT) $(PGI_OPTIONS) $(M_PGI_DEFINE) $< -o $@
endif

clean:
	rm -f matmul_pgi *.o *~ *.mod *.gpu *.cubin *.ptx

