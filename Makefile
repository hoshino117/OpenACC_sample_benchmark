
include Makefile.config

ALL : MATMUL STENCIL

MATMUL :
	cd matmul; make

STENCIL :
	cd stencil; make

clean:
	cd matmul ; make clean
	cd stencil ; make clean
