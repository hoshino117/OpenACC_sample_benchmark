include ../Makefile.config

ALL : ${TARGET}

ifeq ($(useFPP),true)

PGI : 	matmul_pgi.f90
	mv matmul_pgi.f90 pgi/
	cd pgi; make

CAPS : 	matmul_caps.f90
	mv matmul_caps.f90 caps/
	cd caps; make

CRAY : 	matmul_cray.f90
	mv matmul_cray.f90 cray/
	cd cray; make

CUDA : 	matmul_cuda.cuf
	mv matmul_cuda.cuf cuda/
	cd cuda; make

%_pgi.f90 : %.fpp
	$(FPP) $(M_PGI_DEFINE) $< $@

%_caps.f90 : %.fpp
	$(FPP) $(M_CAPS_DEFINE) $< $@

%_cray.f90 : %.fpp
	$(FPP) $(M_CRAY_DEFINE) $< $@

%_cuda.cuf : %.fpp
	$(FPP) $(M_CUDA_DEFINE) $< $@

else

PGI : 	matmul_pgi.F90
	mv matmul_pgi.F90 pgi/
	cd pgi; make

CAPS : 	matmul_caps.F90
	mv matmul_caps.F90 caps/
	cd caps; make

CRAY : 	matmul_cray.F90
	mv matmul_cray.F90 cray/
	cd cray; make

CUDA : 	matmul_cuda.cuf
	mv matmul_cuda.cuf cuda/
	cd cuda; make

%_pgi.F90 : %.fpp
	cp $< $@

%_caps.F90 : %.fpp
	cp $< $@

%_cray.F90 : %.fpp
	cp $< $@

%_cuda.cuf : %.fpp
	cp $< $@
endif

clean:
	cd pgi ; make clean
	cd caps ; make clean
	cd cray ; make clean
	cd cuda ; make clean
