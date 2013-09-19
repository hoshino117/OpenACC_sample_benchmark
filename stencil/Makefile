include ../Makefile.config

ALL : ${TARGET}

ifeq ($(useFPP),true)

PGI : 	diffusion_pgi.f90
	mv diffusion_pgi.f90 pgi/
	cd pgi; make

CAPS : 	diffusion_caps.f90
	mv diffusion_caps.f90 caps/
	cd caps; make

CRAY : 	diffusion_cray.f90
	mv diffusion_cray.f90 cray/
	cd cray; make

CUDA : 	diffusion_cuda.cuf
	mv diffusion_cuda.cuf cuda/
	cd cuda; make

%_pgi.f90 : %.fpp
	$(FPP) $(S_PGI_DEFINE) $< $@

%_caps.f90 : %.fpp
	$(FPP) $(S_CAPS_DEFINE) $< $@

%_cray.f90 : %.fpp
	$(FPP) $(S_CRAY_DEFINE) $< $@

%_cuda.cuf : %.fpp
	$(FPP) $(S_CUDA_DEFINE) $< $@

else

PGI : 	diffusion_pgi.F90
	mv diffusion_pgi.F90 pgi/
	cd pgi; make

CAPS : 	diffusion_caps.F90
	mv diffusion_caps.F90 caps/
	cd caps; make

CRAY : 	diffusion_cray.F90
	mv diffusion_cray.F90 cray/
	cd cray; make

CUDA : 	diffusion_cuda.cuf
	mv diffusion_cuda.cuf cuda/
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
