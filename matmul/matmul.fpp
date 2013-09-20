!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! if you type following comand, 
!!!! > fpp -DCUDA(PGI,CAPS,CRAY) diffusion.fpp diffusion_cuda(pgi,caps,cray).cuf(f90)
!!!! you can get CUDA(OpenACC for PGI,CAPS,CRAY) code.
!!!! if you don't have fpp, 
!!!! > cp diffusion.fpp diffusion.cuf(diffusion.f90)
!!!! > pgfortran -Mpreprocess -DPGI -MCUDA ${OPTIONS} diffusion.cuf -o diffusion_cuda
!!!! > pgfortran -Mpreprocess -DPGI -acc   ${OPTIONS} diffusion.f90 -o diffusion_pgi
!!!! > hmpp ${HMPP_OPTIONS} ifort(gfortran) -cpp      diffusion.f90 -o diffusion_caps
!!!! > ftn       -e F         -DCRAY       ${OPTIONS} diffusion.f90 -o diffusion_cray
!!!! also you can get CUDA(OpenACC) binary file. 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module matmul_mod


#ifndef DATASIZE
#define DATASIZE 2048
#endif

#ifdef PGI

#ifndef THREAD_X
#define THREAD_X 64
#endif
#ifndef THREAD_Y
#define THREAD_Y 4
#endif

#elif CAPS

#ifndef THREAD_X
#define THREAD_X 512
#endif
#define THREAD_Y 1

#elif CRAY

#ifndef THREAD_X
#define THREAD_X 512
#endif
#define THREAD_Y 1

#elif CUDA

  use cudafor
#ifndef THREAD_X
#define THREAD_X 64
#endif
#ifndef THREAD_Y
#define THREAD_Y 4
#endif

#endif

  implicit none

contains

  real(8) function accuracy(cgpu, ccpu, max)
    integer ,intent(in) :: max
    real(8),dimension(max,max) :: cgpu, ccpu
    real(8) :: err
    integer :: i,j
    err = 0.0
    do j = 1,max
       do i = 1,max
          err = err + (cgpu(i,j) - ccpu(i,j)) * (cgpu(i,j) - ccpu(i,j))
       end do
    end do
    accuracy = real(sqrt(err/(max*max)))
  end function accuracy

  subroutine main(max)
    integer, intent(in) :: max
    real(8), dimension(max,max) :: a, b, c
#ifdef CUDA
    real(8), device, dimension(max,max) :: deva, devb, devc
    integer :: ierror
#endif
    real(8), dimension(max,max) :: ccpu
    integer :: t1, t2, t_rate, t_max, diff
    real(8) :: flops,time,err
    integer :: i,j,k

#ifdef PGI
    integer :: devicesync
    devicesync = 0
#endif

    call random_number(a)
    call random_number(b)
    c(:,:) = 0
    ccpu(:,:) = 0

#ifdef DEBUG
    ccpu = matmul(a, b)
#endif

    print "(A)", "optimization type, matrix size[^2], time[sec], flops[GFlops], thread_x size, thread_y size"

#ifdef BASE

#ifdef CUDA 
    deva(:,:) = a(:,:)
    devb(:,:) = b(:,:)
    devc(:,:) = c(:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call matmul_base(deva, devb, devc, max)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    c(:,:) = devc(:,:)
#else

!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call matmul_base(a, b, c, max)

#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif
    call system_clock(t2, t_rate, t_max)
!$acc end data
#endif !CUDA

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)
    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
#ifdef CUDA
    print "(A,I5,A,F10.6,A,F10.6,A,A)", "baseline,", max, ",",time,",",flops,",16",",16"
#else
    print "(A,I5,A,F10.6,A,F10.6,A,A)", "baseline,", max, ",",time,",",flops,",chosen by compiler",",chosen by compiler"
#endif

#ifdef DEBUG
    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err
#endif

#endif !BASE

#ifdef THREAD
    c(:,:) = 0

#ifdef CUDA 
    deva(:,:) = a(:,:)
    devb(:,:) = b(:,:)
    devc(:,:) = c(:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call matmul_thread(deva, devb, devc, max)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    c(:,:) = devc(:,:)
#else
!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call matmul_thread(a, b, c, max)

#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif
    call system_clock(t2, t_rate, t_max)
!$acc end data
#endif !CUDA
    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)
    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
    print "(A,I5,A,F10.6,A,F10.6,A,I3,A,I3)", "opt thread block,", max, ",",time,",",flops,",", THREAD_X, ",", THREAD_Y

#ifdef DEBUG
    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err
#endif
#endif


#ifdef CACHE
    c(:,:) = 0

#ifdef CUDA 
    deva(:,:) = a(:,:)
    devb(:,:) = b(:,:)
    devc(:,:) = c(:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call matmul_cache(deva, devb, devc, max)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    c(:,:) = devc(:,:)
#else
!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call matmul_cache(a, b, c, max)

#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif
    call system_clock(t2, t_rate, t_max)
!$acc end data
#endif !CUDA

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)
    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
    print "(A,I5,A,F10.6,A,F10.6,A,I3,A,I3)", "cache,", max, ",",time,",",flops,",", THREAD_X, ",", THREAD_Y


#ifdef DEBUG
    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err
#endif
#endif

#ifdef UNROLL
    c(:,:) = 0

#ifdef CUDA 
    deva(:,:) = a(:,:)
    devb(:,:) = b(:,:)
    devc(:,:) = c(:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call matmul_unroll(deva, devb, devc, max)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    c(:,:) = devc(:,:)
#else
!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call matmul_unroll(a, b, c, max)

#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif
    call system_clock(t2, t_rate, t_max)
!$acc end data

#endif !CUDA
    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)
    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
    print "(A,I5,A,F10.6,A,F10.6,A,I3,A,I3)", "unroll,", max, ",",time,",",flops,",", THREAD_X, ",", THREAD_Y


#ifdef DEBUG
    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err
#endif
#endif

#ifdef CUDA
#ifdef SHARED
    c(:,:) = 0

    deva(:,:) = a(:,:)
    devb(:,:) = b(:,:)
    devc(:,:) = c(:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call matmul_shared(deva, devb, devc, max)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    c(:,:) = devc(:,:)
    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)

    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
    print "(A,I5,A,F10.6,A,F10.6,A,I3,A,I3)", "shared,", max, ",",time,",",flops,",", 16, ",", 16


#ifdef DEBUG
    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err
#endif
#endif
#endif


  end subroutine main

#ifdef CUDA

  attributes(global) subroutine matmul_base_kernel(A, B, C, max)
    integer,value, intent(in) :: max
    real(8), dimension(max,max), intent(in) :: A
    real(8), dimension(max,max), intent(in) :: B
    real(8), dimension(max,max), intent(inout) :: C
    integer :: i, j, k 
    real(8) :: Cij

    i = (blockidx%x-1) * blockdim%x + threadidx%x
    j = (blockidx%y-1) * blockdim%y + threadidx%y

    Cij = 0.0
    do k = 1, max
       Cij = Cij + A(i,k) * B(k,j)
    enddo
    C(i,j) = Cij

  end subroutine matmul_base_kernel

  attributes(global) subroutine matmul_cache_kernel( A, B, C, max)
    integer,value, intent(in) :: max
    real(8), dimension(max,max), intent(in) :: A
    real(8), dimension(max,max), intent(in) :: B
    real(8), dimension(max,max), intent(inout) :: C
    integer :: i, j, k
    real(8) :: C00,C01,C02,C03
    real(8) :: C10,C11,C12,C13
    real(8) :: C20,C21,C22,C23
    real(8) :: C30,C31,C32,C33

    i = ((blockidx%x-1) * blockdim%x + threadidx%x)*4 -3
    j = ((blockidx%y-1) * blockdim%y + threadidx%y)*4 -3

    C00 = 0.0; C01 = 0.0; C02 = 0.0; C03 = 0.0;
    C10 = 0.0; C11 = 0.0; C12 = 0.0; C13 = 0.0;
    C20 = 0.0; C21 = 0.0; C22 = 0.0; C23 = 0.0;
    C30 = 0.0; C31 = 0.0; C32 = 0.0; C33 = 0.0;
    do k = 1, max
       C00 = C00 + A(i  ,k) * B(k,j  )
       C10 = C10 + A(i+1,k) * B(k,j  )
       C20 = C20 + A(i+2,k) * B(k,j  )
       C30 = C30 + A(i+3,k) * B(k,j  )
       C01 = C01 + A(i  ,k) * B(k,j+1)
       C11 = C11 + A(i+1,k) * B(k,j+1)
       C21 = C21 + A(i+2,k) * B(k,j+1)
       C31 = C31 + A(i+3,k) * B(k,j+1)
       C02 = C02 + A(i  ,k) * B(k,j+2)
       C12 = C12 + A(i+1,k) * B(k,j+2)
       C22 = C22 + A(i+2,k) * B(k,j+2)
       C32 = C32 + A(i+3,k) * B(k,j+2)
       C03 = C03 + A(i  ,k) * B(k,j+3)
       C13 = C13 + A(i+1,k) * B(k,j+3)
       C23 = C23 + A(i+2,k) * B(k,j+3)
       C33 = C33 + A(i+3,k) * B(k,j+3)
    enddo
    C(i  ,j  ) = C00
    C(i+1,j  ) = C10
    C(i+2,j  ) = C20
    C(i+3,j  ) = C30
    C(i  ,j+1) = C01
    C(i+1,j+1) = C11
    C(i+2,j+1) = C21
    C(i+3,j+1) = C31
    C(i  ,j+2) = C02
    C(i+1,j+2) = C12
    C(i+2,j+2) = C22
    C(i+3,j+2) = C32
    C(i  ,j+3) = C03
    C(i+1,j+3) = C13
    C(i+2,j+3) = C23
    C(i+3,j+3) = C33
  end subroutine matmul_cache_kernel

  attributes(global) subroutine matmul_unroll_kernel( A, B, C, max)
    integer,value, intent(in) :: max
    real(8), dimension(max,max), intent(in) :: A
    real(8), dimension(max,max), intent(in) :: B
    real(8), dimension(max,max), intent(inout) :: C
    integer :: i, j, k
    real(8) :: C00,C01,C02,C03
    real(8) :: C10,C11,C12,C13
    real(8) :: C20,C21,C22,C23
    real(8) :: C30,C31,C32,C33

    i = ((blockidx%x-1) * blockdim%x + threadidx%x)*4 -3
    j = ((blockidx%y-1) * blockdim%y + threadidx%y)*4 -3

    C00 = 0.0; C01 = 0.0; C02 = 0.0; C03 = 0.0;
    C10 = 0.0; C11 = 0.0; C12 = 0.0; C13 = 0.0;
    C20 = 0.0; C21 = 0.0; C22 = 0.0; C23 = 0.0;
    C30 = 0.0; C31 = 0.0; C32 = 0.0; C33 = 0.0;
    do k = 1, max, 16
       C00 = C00 + A(i  ,k) * B(k,j  )
       C10 = C10 + A(i+1,k) * B(k,j  )
       C20 = C20 + A(i+2,k) * B(k,j  )
       C30 = C30 + A(i+3,k) * B(k,j  )
       C01 = C01 + A(i  ,k) * B(k,j+1)
       C11 = C11 + A(i+1,k) * B(k,j+1)
       C21 = C21 + A(i+2,k) * B(k,j+1)
       C31 = C31 + A(i+3,k) * B(k,j+1)
       C02 = C02 + A(i  ,k) * B(k,j+2)
       C12 = C12 + A(i+1,k) * B(k,j+2)
       C22 = C22 + A(i+2,k) * B(k,j+2)
       C32 = C32 + A(i+3,k) * B(k,j+2)
       C03 = C03 + A(i  ,k) * B(k,j+3)
       C13 = C13 + A(i+1,k) * B(k,j+3)
       C23 = C23 + A(i+2,k) * B(k,j+3)
       C33 = C33 + A(i+3,k) * B(k,j+3)
       C00 = C00 + A(i  ,k+1) * B(k+1,j  )
       C10 = C10 + A(i+1,k+1) * B(k+1,j  )
       C20 = C20 + A(i+2,k+1) * B(k+1,j  )
       C30 = C30 + A(i+3,k+1) * B(k+1,j  )
       C01 = C01 + A(i  ,k+1) * B(k+1,j+1)
       C11 = C11 + A(i+1,k+1) * B(k+1,j+1)
       C21 = C21 + A(i+2,k+1) * B(k+1,j+1)
       C31 = C31 + A(i+3,k+1) * B(k+1,j+1)
       C02 = C02 + A(i  ,k+1) * B(k+1,j+2)
       C12 = C12 + A(i+1,k+1) * B(k+1,j+2)
       C22 = C22 + A(i+2,k+1) * B(k+1,j+2)
       C32 = C32 + A(i+3,k+1) * B(k+1,j+2)
       C03 = C03 + A(i  ,k+1) * B(k+1,j+3)
       C13 = C13 + A(i+1,k+1) * B(k+1,j+3)
       C23 = C23 + A(i+2,k+1) * B(k+1,j+3)
       C33 = C33 + A(i+3,k+1) * B(k+1,j+3)
       C00 = C00 + A(i  ,k+2) * B(k+2,j  )
       C10 = C10 + A(i+1,k+2) * B(k+2,j  )
       C20 = C20 + A(i+2,k+2) * B(k+2,j  )
       C30 = C30 + A(i+3,k+2) * B(k+2,j  )
       C01 = C01 + A(i  ,k+2) * B(k+2,j+1)
       C11 = C11 + A(i+1,k+2) * B(k+2,j+1)
       C21 = C21 + A(i+2,k+2) * B(k+2,j+1)
       C31 = C31 + A(i+3,k+2) * B(k+2,j+1)
       C02 = C02 + A(i  ,k+2) * B(k+2,j+2)
       C12 = C12 + A(i+1,k+2) * B(k+2,j+2)
       C22 = C22 + A(i+2,k+2) * B(k+2,j+2)
       C32 = C32 + A(i+3,k+2) * B(k+2,j+2)
       C03 = C03 + A(i  ,k+2) * B(k+2,j+3)
       C13 = C13 + A(i+1,k+2) * B(k+2,j+3)
       C23 = C23 + A(i+2,k+2) * B(k+2,j+3)
       C33 = C33 + A(i+3,k+2) * B(k+2,j+3)
       C00 = C00 + A(i  ,k+3) * B(k+3,j  )
       C10 = C10 + A(i+1,k+3) * B(k+3,j  )
       C20 = C20 + A(i+2,k+3) * B(k+3,j  )
       C30 = C30 + A(i+3,k+3) * B(k+3,j  )
       C01 = C01 + A(i  ,k+3) * B(k+3,j+1)
       C11 = C11 + A(i+1,k+3) * B(k+3,j+1)
       C21 = C21 + A(i+2,k+3) * B(k+3,j+1)
       C31 = C31 + A(i+3,k+3) * B(k+3,j+1)
       C02 = C02 + A(i  ,k+3) * B(k+3,j+2)
       C12 = C12 + A(i+1,k+3) * B(k+3,j+2)
       C22 = C22 + A(i+2,k+3) * B(k+3,j+2)
       C32 = C32 + A(i+3,k+3) * B(k+3,j+2)
       C03 = C03 + A(i  ,k+3) * B(k+3,j+3)
       C13 = C13 + A(i+1,k+3) * B(k+3,j+3)
       C23 = C23 + A(i+2,k+3) * B(k+3,j+3)
       C33 = C33 + A(i+3,k+3) * B(k+3,j+3)
       C00 = C00 + A(i  ,k+4) * B(k+4,j  )
       C10 = C10 + A(i+1,k+4) * B(k+4,j  )
       C20 = C20 + A(i+2,k+4) * B(k+4,j  )
       C30 = C30 + A(i+3,k+4) * B(k+4,j  )
       C01 = C01 + A(i  ,k+4) * B(k+4,j+1)
       C11 = C11 + A(i+1,k+4) * B(k+4,j+1)
       C21 = C21 + A(i+2,k+4) * B(k+4,j+1)
       C31 = C31 + A(i+3,k+4) * B(k+4,j+1)
       C02 = C02 + A(i  ,k+4) * B(k+4,j+2)
       C12 = C12 + A(i+1,k+4) * B(k+4,j+2)
       C22 = C22 + A(i+2,k+4) * B(k+4,j+2)
       C32 = C32 + A(i+3,k+4) * B(k+4,j+2)
       C03 = C03 + A(i  ,k+4) * B(k+4,j+3)
       C13 = C13 + A(i+1,k+4) * B(k+4,j+3)
       C23 = C23 + A(i+2,k+4) * B(k+4,j+3)
       C33 = C33 + A(i+3,k+4) * B(k+4,j+3)
       C00 = C00 + A(i  ,k+5) * B(k+5,j  )
       C10 = C10 + A(i+1,k+5) * B(k+5,j  )
       C20 = C20 + A(i+2,k+5) * B(k+5,j  )
       C30 = C30 + A(i+3,k+5) * B(k+5,j  )
       C01 = C01 + A(i  ,k+5) * B(k+5,j+1)
       C11 = C11 + A(i+1,k+5) * B(k+5,j+1)
       C21 = C21 + A(i+2,k+5) * B(k+5,j+1)
       C31 = C31 + A(i+3,k+5) * B(k+5,j+1)
       C02 = C02 + A(i  ,k+5) * B(k+5,j+2)
       C12 = C12 + A(i+1,k+5) * B(k+5,j+2)
       C22 = C22 + A(i+2,k+5) * B(k+5,j+2)
       C32 = C32 + A(i+3,k+5) * B(k+5,j+2)
       C03 = C03 + A(i  ,k+5) * B(k+5,j+3)
       C13 = C13 + A(i+1,k+5) * B(k+5,j+3)
       C23 = C23 + A(i+2,k+5) * B(k+5,j+3)
       C33 = C33 + A(i+3,k+5) * B(k+5,j+3)
       C00 = C00 + A(i  ,k+6) * B(k+6,j  )
       C10 = C10 + A(i+1,k+6) * B(k+6,j  )
       C20 = C20 + A(i+2,k+6) * B(k+6,j  )
       C30 = C30 + A(i+3,k+6) * B(k+6,j  )
       C01 = C01 + A(i  ,k+6) * B(k+6,j+1)
       C11 = C11 + A(i+1,k+6) * B(k+6,j+1)
       C21 = C21 + A(i+2,k+6) * B(k+6,j+1)
       C31 = C31 + A(i+3,k+6) * B(k+6,j+1)
       C02 = C02 + A(i  ,k+6) * B(k+6,j+2)
       C12 = C12 + A(i+1,k+6) * B(k+6,j+2)
       C22 = C22 + A(i+2,k+6) * B(k+6,j+2)
       C32 = C32 + A(i+3,k+6) * B(k+6,j+2)
       C03 = C03 + A(i  ,k+6) * B(k+6,j+3)
       C13 = C13 + A(i+1,k+6) * B(k+6,j+3)
       C23 = C23 + A(i+2,k+6) * B(k+6,j+3)
       C33 = C33 + A(i+3,k+6) * B(k+6,j+3)
       C00 = C00 + A(i  ,k+7) * B(k+7,j  )
       C10 = C10 + A(i+1,k+7) * B(k+7,j  )
       C20 = C20 + A(i+2,k+7) * B(k+7,j  )
       C30 = C30 + A(i+3,k+7) * B(k+7,j  )
       C01 = C01 + A(i  ,k+7) * B(k+7,j+1)
       C11 = C11 + A(i+1,k+7) * B(k+7,j+1)
       C21 = C21 + A(i+2,k+7) * B(k+7,j+1)
       C31 = C31 + A(i+3,k+7) * B(k+7,j+1)
       C02 = C02 + A(i  ,k+7) * B(k+7,j+2)
       C12 = C12 + A(i+1,k+7) * B(k+7,j+2)
       C22 = C22 + A(i+2,k+7) * B(k+7,j+2)
       C32 = C32 + A(i+3,k+7) * B(k+7,j+2)
       C03 = C03 + A(i  ,k+7) * B(k+7,j+3)
       C13 = C13 + A(i+1,k+7) * B(k+7,j+3)
       C23 = C23 + A(i+2,k+7) * B(k+7,j+3)
       C33 = C33 + A(i+3,k+7) * B(k+7,j+3)
       C00 = C00 + A(i  ,k+8) * B(k+8,j  )
       C10 = C10 + A(i+1,k+8) * B(k+8,j  )
       C20 = C20 + A(i+2,k+8) * B(k+8,j  )
       C30 = C30 + A(i+3,k+8) * B(k+8,j  )
       C01 = C01 + A(i  ,k+8) * B(k+8,j+1)
       C11 = C11 + A(i+1,k+8) * B(k+8,j+1)
       C21 = C21 + A(i+2,k+8) * B(k+8,j+1)
       C31 = C31 + A(i+3,k+8) * B(k+8,j+1)
       C02 = C02 + A(i  ,k+8) * B(k+8,j+2)
       C12 = C12 + A(i+1,k+8) * B(k+8,j+2)
       C22 = C22 + A(i+2,k+8) * B(k+8,j+2)
       C32 = C32 + A(i+3,k+8) * B(k+8,j+2)
       C03 = C03 + A(i  ,k+8) * B(k+8,j+3)
       C13 = C13 + A(i+1,k+8) * B(k+8,j+3)
       C23 = C23 + A(i+2,k+8) * B(k+8,j+3)
       C33 = C33 + A(i+3,k+8) * B(k+8,j+3)
       C00 = C00 + A(i  ,k+9) * B(k+9,j  )
       C10 = C10 + A(i+1,k+9) * B(k+9,j  )
       C20 = C20 + A(i+2,k+9) * B(k+9,j  )
       C30 = C30 + A(i+3,k+9) * B(k+9,j  )
       C01 = C01 + A(i  ,k+9) * B(k+9,j+1)
       C11 = C11 + A(i+1,k+9) * B(k+9,j+1)
       C21 = C21 + A(i+2,k+9) * B(k+9,j+1)
       C31 = C31 + A(i+3,k+9) * B(k+9,j+1)
       C02 = C02 + A(i  ,k+9) * B(k+9,j+2)
       C12 = C12 + A(i+1,k+9) * B(k+9,j+2)
       C22 = C22 + A(i+2,k+9) * B(k+9,j+2)
       C32 = C32 + A(i+3,k+9) * B(k+9,j+2)
       C03 = C03 + A(i  ,k+9) * B(k+9,j+3)
       C13 = C13 + A(i+1,k+9) * B(k+9,j+3)
       C23 = C23 + A(i+2,k+9) * B(k+9,j+3)
       C33 = C33 + A(i+3,k+9) * B(k+9,j+3)
       C00 = C00 + A(i  ,k+10) * B(k+10,j  )
       C10 = C10 + A(i+1,k+10) * B(k+10,j  )
       C20 = C20 + A(i+2,k+10) * B(k+10,j  )
       C30 = C30 + A(i+3,k+10) * B(k+10,j  )
       C01 = C01 + A(i  ,k+10) * B(k+10,j+1)
       C11 = C11 + A(i+1,k+10) * B(k+10,j+1)
       C21 = C21 + A(i+2,k+10) * B(k+10,j+1)
       C31 = C31 + A(i+3,k+10) * B(k+10,j+1)
       C02 = C02 + A(i  ,k+10) * B(k+10,j+2)
       C12 = C12 + A(i+1,k+10) * B(k+10,j+2)
       C22 = C22 + A(i+2,k+10) * B(k+10,j+2)
       C32 = C32 + A(i+3,k+10) * B(k+10,j+2)
       C03 = C03 + A(i  ,k+10) * B(k+10,j+3)
       C13 = C13 + A(i+1,k+10) * B(k+10,j+3)
       C23 = C23 + A(i+2,k+10) * B(k+10,j+3)
       C33 = C33 + A(i+3,k+10) * B(k+10,j+3)
       C00 = C00 + A(i  ,k+11) * B(k+11,j  )
       C10 = C10 + A(i+1,k+11) * B(k+11,j  )
       C20 = C20 + A(i+2,k+11) * B(k+11,j  )
       C30 = C30 + A(i+3,k+11) * B(k+11,j  )
       C01 = C01 + A(i  ,k+11) * B(k+11,j+1)
       C11 = C11 + A(i+1,k+11) * B(k+11,j+1)
       C21 = C21 + A(i+2,k+11) * B(k+11,j+1)
       C31 = C31 + A(i+3,k+11) * B(k+11,j+1)
       C02 = C02 + A(i  ,k+11) * B(k+11,j+2)
       C12 = C12 + A(i+1,k+11) * B(k+11,j+2)
       C22 = C22 + A(i+2,k+11) * B(k+11,j+2)
       C32 = C32 + A(i+3,k+11) * B(k+11,j+2)
       C03 = C03 + A(i  ,k+11) * B(k+11,j+3)
       C13 = C13 + A(i+1,k+11) * B(k+11,j+3)
       C23 = C23 + A(i+2,k+11) * B(k+11,j+3)
       C33 = C33 + A(i+3,k+11) * B(k+11,j+3)
       C00 = C00 + A(i  ,k+12) * B(k+12,j  )
       C10 = C10 + A(i+1,k+12) * B(k+12,j  )
       C20 = C20 + A(i+2,k+12) * B(k+12,j  )
       C30 = C30 + A(i+3,k+12) * B(k+12,j  )
       C01 = C01 + A(i  ,k+12) * B(k+12,j+1)
       C11 = C11 + A(i+1,k+12) * B(k+12,j+1)
       C21 = C21 + A(i+2,k+12) * B(k+12,j+1)
       C31 = C31 + A(i+3,k+12) * B(k+12,j+1)
       C02 = C02 + A(i  ,k+12) * B(k+12,j+2)
       C12 = C12 + A(i+1,k+12) * B(k+12,j+2)
       C22 = C22 + A(i+2,k+12) * B(k+12,j+2)
       C32 = C32 + A(i+3,k+12) * B(k+12,j+2)
       C03 = C03 + A(i  ,k+12) * B(k+12,j+3)
       C13 = C13 + A(i+1,k+12) * B(k+12,j+3)
       C23 = C23 + A(i+2,k+12) * B(k+12,j+3)
       C33 = C33 + A(i+3,k+12) * B(k+12,j+3)
       C00 = C00 + A(i  ,k+13) * B(k+13,j  )
       C10 = C10 + A(i+1,k+13) * B(k+13,j  )
       C20 = C20 + A(i+2,k+13) * B(k+13,j  )
       C30 = C30 + A(i+3,k+13) * B(k+13,j  )
       C01 = C01 + A(i  ,k+13) * B(k+13,j+1)
       C11 = C11 + A(i+1,k+13) * B(k+13,j+1)
       C21 = C21 + A(i+2,k+13) * B(k+13,j+1)
       C31 = C31 + A(i+3,k+13) * B(k+13,j+1)
       C02 = C02 + A(i  ,k+13) * B(k+13,j+2)
       C12 = C12 + A(i+1,k+13) * B(k+13,j+2)
       C22 = C22 + A(i+2,k+13) * B(k+13,j+2)
       C32 = C32 + A(i+3,k+13) * B(k+13,j+2)
       C03 = C03 + A(i  ,k+13) * B(k+13,j+3)
       C13 = C13 + A(i+1,k+13) * B(k+13,j+3)
       C23 = C23 + A(i+2,k+13) * B(k+13,j+3)
       C33 = C33 + A(i+3,k+13) * B(k+13,j+3)
       C00 = C00 + A(i  ,k+14) * B(k+14,j  )
       C10 = C10 + A(i+1,k+14) * B(k+14,j  )
       C20 = C20 + A(i+2,k+14) * B(k+14,j  )
       C30 = C30 + A(i+3,k+14) * B(k+14,j  )
       C01 = C01 + A(i  ,k+14) * B(k+14,j+1)
       C11 = C11 + A(i+1,k+14) * B(k+14,j+1)
       C21 = C21 + A(i+2,k+14) * B(k+14,j+1)
       C31 = C31 + A(i+3,k+14) * B(k+14,j+1)
       C02 = C02 + A(i  ,k+14) * B(k+14,j+2)
       C12 = C12 + A(i+1,k+14) * B(k+14,j+2)
       C22 = C22 + A(i+2,k+14) * B(k+14,j+2)
       C32 = C32 + A(i+3,k+14) * B(k+14,j+2)
       C03 = C03 + A(i  ,k+14) * B(k+14,j+3)
       C13 = C13 + A(i+1,k+14) * B(k+14,j+3)
       C23 = C23 + A(i+2,k+14) * B(k+14,j+3)
       C33 = C33 + A(i+3,k+14) * B(k+14,j+3)
       C00 = C00 + A(i  ,k+15) * B(k+15,j  )
       C10 = C10 + A(i+1,k+15) * B(k+15,j  )
       C20 = C20 + A(i+2,k+15) * B(k+15,j  )
       C30 = C30 + A(i+3,k+15) * B(k+15,j  )
       C01 = C01 + A(i  ,k+15) * B(k+15,j+1)
       C11 = C11 + A(i+1,k+15) * B(k+15,j+1)
       C21 = C21 + A(i+2,k+15) * B(k+15,j+1)
       C31 = C31 + A(i+3,k+15) * B(k+15,j+1)
       C02 = C02 + A(i  ,k+15) * B(k+15,j+2)
       C12 = C12 + A(i+1,k+15) * B(k+15,j+2)
       C22 = C22 + A(i+2,k+15) * B(k+15,j+2)
       C32 = C32 + A(i+3,k+15) * B(k+15,j+2)
       C03 = C03 + A(i  ,k+15) * B(k+15,j+3)
       C13 = C13 + A(i+1,k+15) * B(k+15,j+3)
       C23 = C23 + A(i+2,k+15) * B(k+15,j+3)
       C33 = C33 + A(i+3,k+15) * B(k+15,j+3)
    enddo
    C(i  ,j  ) = C00
    C(i+1,j  ) = C10
    C(i+2,j  ) = C20
    C(i+3,j  ) = C30
    C(i  ,j+1) = C01
    C(i+1,j+1) = C11
    C(i+2,j+1) = C21
    C(i+3,j+1) = C31
    C(i  ,j+2) = C02
    C(i+1,j+2) = C12
    C(i+2,j+2) = C22
    C(i+3,j+2) = C32
    C(i  ,j+3) = C03
    C(i+1,j+3) = C13
    C(i+2,j+3) = C23
    C(i+3,j+3) = C33
  end subroutine matmul_unroll_kernel


  attributes(global) subroutine matmul_shared_kernel( A, B, C, max )
    integer,value, intent(in) :: max
    real(8), dimension(max,max), intent(in) :: A
    real(8), dimension(max,max), intent(in) :: B
    real(8), dimension(max,max), intent(inout) :: C
    integer :: i, j, kb, tx, ty, k
    real(8), shared :: Asub(17,16), Bsub(17,64)
    real(8) :: Cij1, Cij2, Cij3, Cij4

    tx = threadidx%x
    ty = threadidx%y
    i = (blockidx%x-1) * 16 + threadidx%x
    j = (blockidx%y-1) * 16 * 4 + threadidx%y

    Cij1 = 0.0
    Cij2 = 0.0
    Cij3 = 0.0
    Cij4 = 0.0
    do kb = 0, max-1, 16
       Asub(tx,ty   ) = A(i,kb+ty)
       Bsub(tx,ty   ) = B(kb+tx,j   )
       Bsub(tx,ty+16) = B(kb+tx,j+16)
       Bsub(tx,ty+32) = B(kb+tx,j+32)
       Bsub(tx,ty+48) = B(kb+tx,j+48)
       call syncthreads()
#if 1
       do k = 1, 16
          Cij1 = Cij1 + Asub(tx,k) * Bsub(k,ty   )
          Cij2 = Cij2 + Asub(tx,k) * Bsub(k,ty+16)
          Cij3 = Cij3 + Asub(tx,k) * Bsub(k,ty+32)
          Cij4 = Cij4 + Asub(tx,k) * Bsub(k,ty+48)
       end do 
#else
       Cij1 = Cij1 + Asub(tx,1) * Bsub(1,ty   )
       Cij2 = Cij2 + Asub(tx,1) * Bsub(1,ty+16)
       Cij3 = Cij3 + Asub(tx,1) * Bsub(1,ty+32)
       Cij4 = Cij4 + Asub(tx,1) * Bsub(1,ty+48)
       Cij1 = Cij1 + Asub(tx,2) * Bsub(2,ty   )
       Cij2 = Cij2 + Asub(tx,2) * Bsub(2,ty+16)
       Cij3 = Cij3 + Asub(tx,2) * Bsub(2,ty+32)
       Cij4 = Cij4 + Asub(tx,2) * Bsub(2,ty+48)
       Cij1 = Cij1 + Asub(tx,3) * Bsub(3,ty   )
       Cij2 = Cij2 + Asub(tx,3) * Bsub(3,ty+16)
       Cij3 = Cij3 + Asub(tx,3) * Bsub(3,ty+32)
       Cij4 = Cij4 + Asub(tx,3) * Bsub(3,ty+48)
       Cij1 = Cij1 + Asub(tx,4) * Bsub(4,ty   )
       Cij2 = Cij2 + Asub(tx,4) * Bsub(4,ty+16)
       Cij3 = Cij3 + Asub(tx,4) * Bsub(4,ty+32)
       Cij4 = Cij4 + Asub(tx,4) * Bsub(4,ty+48)
       Cij1 = Cij1 + Asub(tx,5) * Bsub(5,ty   )
       Cij2 = Cij2 + Asub(tx,5) * Bsub(5,ty+16)
       Cij3 = Cij3 + Asub(tx,5) * Bsub(5,ty+32)
       Cij4 = Cij4 + Asub(tx,5) * Bsub(5,ty+48)
       Cij1 = Cij1 + Asub(tx,6) * Bsub(6,ty   )
       Cij2 = Cij2 + Asub(tx,6) * Bsub(6,ty+16)
       Cij3 = Cij3 + Asub(tx,6) * Bsub(6,ty+32)
       Cij4 = Cij4 + Asub(tx,6) * Bsub(6,ty+48)
       Cij1 = Cij1 + Asub(tx,7) * Bsub(7,ty   )
       Cij2 = Cij2 + Asub(tx,7) * Bsub(7,ty+16)
       Cij3 = Cij3 + Asub(tx,7) * Bsub(7,ty+32)
       Cij4 = Cij4 + Asub(tx,7) * Bsub(7,ty+48)
       Cij1 = Cij1 + Asub(tx,8) * Bsub(8,ty   )
       Cij2 = Cij2 + Asub(tx,8) * Bsub(8,ty+16)
       Cij3 = Cij3 + Asub(tx,8) * Bsub(8,ty+32)
       Cij4 = Cij4 + Asub(tx,8) * Bsub(8,ty+48)
       Cij1 = Cij1 + Asub(tx,9) * Bsub(9,ty   )
       Cij2 = Cij2 + Asub(tx,9) * Bsub(9,ty+16)
       Cij3 = Cij3 + Asub(tx,9) * Bsub(9,ty+32)
       Cij4 = Cij4 + Asub(tx,9) * Bsub(9,ty+48)
       Cij1 = Cij1 + Asub(tx,10) * Bsub(10,ty   )
       Cij2 = Cij2 + Asub(tx,10) * Bsub(10,ty+16)
       Cij3 = Cij3 + Asub(tx,10) * Bsub(10,ty+32)
       Cij4 = Cij4 + Asub(tx,10) * Bsub(10,ty+48)
       Cij1 = Cij1 + Asub(tx,11) * Bsub(11,ty   )
       Cij2 = Cij2 + Asub(tx,11) * Bsub(11,ty+16)
       Cij3 = Cij3 + Asub(tx,11) * Bsub(11,ty+32)
       Cij4 = Cij4 + Asub(tx,11) * Bsub(11,ty+48)
       Cij1 = Cij1 + Asub(tx,12) * Bsub(12,ty   )
       Cij2 = Cij2 + Asub(tx,12) * Bsub(12,ty+16)
       Cij3 = Cij3 + Asub(tx,12) * Bsub(12,ty+32)
       Cij4 = Cij4 + Asub(tx,12) * Bsub(12,ty+48)
       Cij1 = Cij1 + Asub(tx,13) * Bsub(13,ty   )
       Cij2 = Cij2 + Asub(tx,13) * Bsub(13,ty+16)
       Cij3 = Cij3 + Asub(tx,13) * Bsub(13,ty+32)
       Cij4 = Cij4 + Asub(tx,13) * Bsub(13,ty+48)
       Cij1 = Cij1 + Asub(tx,14) * Bsub(14,ty   )
       Cij2 = Cij2 + Asub(tx,14) * Bsub(14,ty+16)
       Cij3 = Cij3 + Asub(tx,14) * Bsub(14,ty+32)
       Cij4 = Cij4 + Asub(tx,14) * Bsub(14,ty+48)
       Cij1 = Cij1 + Asub(tx,15) * Bsub(15,ty   )
       Cij2 = Cij2 + Asub(tx,15) * Bsub(15,ty+16)
       Cij3 = Cij3 + Asub(tx,15) * Bsub(15,ty+32)
       Cij4 = Cij4 + Asub(tx,15) * Bsub(15,ty+48)
       Cij1 = Cij1 + Asub(tx,16) * Bsub(16,ty   )
       Cij2 = Cij2 + Asub(tx,16) * Bsub(16,ty+16)
       Cij3 = Cij3 + Asub(tx,16) * Bsub(16,ty+32)
       Cij4 = Cij4 + Asub(tx,16) * Bsub(16,ty+48)
#endif
       call syncthreads()
    enddo
    C(i,j   ) = Cij1
    C(i,j+16) = Cij2
    C(i,j+32) = Cij3
    C(i,j+48) = Cij4
  end subroutine matmul_shared_kernel

#endif !CUDA


#ifdef BASE
  subroutine matmul_base(a, b, c, max)
    integer, intent(in) :: max
#ifdef CUDA
    real(8), device, dimension(max,max), intent(in) :: a
    real(8), device, dimension(max,max), intent(in) :: b
    real(8), device, dimension(max,max), intent(inout) :: c
    type(dim3) :: dimGrid, dimBlock
#else
    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c
#endif
    integer :: i, j, k
    real(8) :: cc

#ifdef CUDA
    dimGrid  = dim3((max-1)/16 + 1, (max-1)/16+ 1, 1)
    dimBlock = dim3(16, 16, 1)
    call matmul_base_kernel<<<dimGrid, dimBlock>>>(a, b, c, max)
#else

!$acc kernels present(a, b, c)
!$acc loop 
    do j = 1, max
!$acc loop 
       do i = 1, max
          cc = 0.0
!$acc loop seq
          do k = 1, max
             cc = cc + a(i,k) * b(k,j)
          end do
          c(i,j) = cc
       end do
    end do
!$acc end kernels
#endif !CUDA
  end subroutine matmul_base
#endif !BASE


#ifdef THREAD
  subroutine matmul_thread(a, b, c, max)
    integer, intent(in) :: max
#ifdef CUDA
    real(8), device, dimension(max,max), intent(in) :: a
    real(8), device, dimension(max,max), intent(in) :: b
    real(8), device, dimension(max,max), intent(inout) :: c
    type(dim3) :: dimGrid, dimBlock
#else
    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c
#endif
    integer :: i, j, k
    real(8) :: cc

#ifdef CUDA
    dimGrid  = dim3((max-1)/THREAD_X + 1, (max-1)/THREAD_Y+ 1, 1)
    dimBlock = dim3(THREAD_X, THREAD_Y, 1)
    call matmul_base_kernel<<<dimGrid, dimBlock>>>(a, b, c, max)
#else

!$acc kernels present(a, b, c)
#ifdef PGI
!$acc loop gang vector(THREAD_Y)
#elif CAPS
!$acc loop gang(max)
#elif CRAY
!$acc loop gang
#endif
    do j = 1, max
#ifdef PGI
!$acc loop gang vector(THREAD_X)
#elif CAPS
!$acc loop worker(THREAD_X)
#elif CRAY
!$acc loop vector(THREAD_X)
#endif
       do i = 1, max
          cc = 0.0
!$acc loop seq
          do k = 1, max
             cc = cc + a(i,k) * b(k,j)
          end do
          c(i,j) = cc
       end do
    end do
!$acc end kernels
#endif !CUDA
  end subroutine matmul_thread
#endif !THREAD


#ifdef CACHE
#if 0
  subroutine matmul_cache(a, b, c, max)
    integer, intent(in) :: max
#ifdef CUDA
    real(8), device, dimension(max,max), intent(in) :: a
    real(8), device, dimension(max,max), intent(in) :: b
    real(8), device, dimension(max,max), intent(inout) :: c
    type(dim3) :: dimGrid, dimBlock
#else
    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c
#endif
    integer :: i, j, k, o
    real(8) :: cc


!$acc kernels present(a,b,c)
!$acc loop gang vector(16)
    do i = 1, max 
!$acc loop gang vector(16) 
       do j = 1, max
          cc = 0
!$acc loop seq
          do o = 1, max, 16
!!$acc cache (a(i:i+15,o:o+15))
!!$acc cache (a(i:i+15,o:o+15))
!!$acc cache (b(o:o+15,j:j+15))
             do k = o, o+15
!$acc cache (b(o:o+15,j:j+15))
                cc = cc + a(i,k) * b(k,j)
             end do
          end do
          c(i,j) = cc
       end do
    end do
!$acc end kernels

  end subroutine matmul_cache

  subroutine matmul_cache(a, b, c, max)
    integer, intent(in) :: max
#ifdef CUDA
    real(8), device, dimension(max,max), intent(in) :: a
    real(8), device, dimension(max,max), intent(in) :: b
    real(8), device, dimension(max,max), intent(inout) :: c
    type(dim3) :: dimGrid, dimBlock
#else
    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c
#endif
    integer :: i, j, k
    real(8) :: cc


!$acc kernels present(a,b,c)
!$acc loop gang vector(16)
    do i = 1, max 
!$acc loop gang vector(16) 
       do j = 1, max
          cc = 0
          do k = 1, max
!$acc cache (b(k:k+15,j:j+15))
             cc = cc + a(i,k) * b(k,j)
          end do
          c(i,j) = cc
       end do
    end do
!$acc end kernels

  end subroutine matmul_cache
#endif


  subroutine matmul_cache(a, b, c, max)
    integer, intent(in) :: max
#ifdef CUDA
    real(8), device, dimension(max,max), intent(in) :: a
    real(8), device, dimension(max,max), intent(in) :: b
    real(8), device, dimension(max,max), intent(inout) :: c
    type(dim3) :: dimGrid, dimBlock
#else
    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c
#endif
    integer :: i, j, k
    real(8) :: c00,c10,c20,c30,c01,c11,c21,c31,c02,c12,c22,c32,c03,c13,c23,c33

#ifdef CUDA
    dimGrid  = dim3((max-1)/4/THREAD_X + 1, (max-1)/4/THREAD_Y+ 1, 1)
    dimBlock = dim3(THREAD_X, THREAD_Y, 1)
    call matmul_cache_kernel<<<dimGrid, dimBlock>>>(a, b, c, max)
#else
!$acc kernels present(a,b,c)
#ifdef PGI
!$acc loop gang vector(THREAD_Y)
#elif CAPS
!$acc loop gang(max/4)
#elif CRAY
!$acc loop gang
#endif

    do j = 1, max, 4
#ifdef PGI
!$acc loop gang vector(THREAD_X)
#elif CAPS
!$acc loop worker(THREAD_X)
#elif CRAY
!$acc loop vector(THREAD_X)
#endif
       do i = 1, max, 4
          c00 = 0
          c10 = 0
          c20 = 0
          c30 = 0
          c01 = 0
          c11 = 0
          c21 = 0
          c31 = 0
          c02 = 0
          c12 = 0
          c22 = 0
          c32 = 0
          c03 = 0
          c13 = 0
          c23 = 0
          c33 = 0
          do k = 1, max
             c00 = c00 + a(i  ,k) * b(k,j  )
             c10 = c10 + a(i+1,k) * b(k,j  )
             c20 = c20 + a(i+2,k) * b(k,j  )
             c30 = c30 + a(i+3,k) * b(k,j  )
             c01 = c01 + a(i  ,k) * b(k,j+1)
             c11 = c11 + a(i+1,k) * b(k,j+1)
             c21 = c21 + a(i+2,k) * b(k,j+1)
             c31 = c31 + a(i+3,k) * b(k,j+1)
             c02 = c02 + a(i  ,k) * b(k,j+2)
             c12 = c12 + a(i+1,k) * b(k,j+2)
             c22 = c22 + a(i+2,k) * b(k,j+2)
             c32 = c32 + a(i+3,k) * b(k,j+2)
             c03 = c03 + a(i  ,k) * b(k,j+3)
             c13 = c13 + a(i+1,k) * b(k,j+3)
             c23 = c23 + a(i+2,k) * b(k,j+3)
             c33 = c33 + a(i+3,k) * b(k,j+3)
          end do
          c(i  ,j  ) = c00
          c(i+1,j  ) = c10
          c(i+2,j  ) = c20
          c(i+3,j  ) = c30
          c(i  ,j+1) = c01
          c(i+1,j+1) = c11
          c(i+2,j+1) = c21
          c(i+3,j+1) = c31
          c(i  ,j+2) = c02
          c(i+1,j+2) = c12
          c(i+2,j+2) = c22
          c(i+3,j+2) = c32
          c(i  ,j+3) = c03
          c(i+1,j+3) = c13
          c(i+2,j+3) = c23
          c(i+3,j+3) = c33
       end do
    end do
!$acc end kernels
#endif !CUDA

  end subroutine matmul_cache
#endif

#ifdef UNROLL
  subroutine matmul_unroll(a, b, c, max)
    integer, intent(in) :: max
#ifdef CUDA
    real(8), device, dimension(max,max), intent(in) :: a
    real(8), device, dimension(max,max), intent(in) :: b
    real(8), device, dimension(max,max), intent(inout) :: c
    type(dim3) :: dimGrid, dimBlock
#else
    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c
#endif
    integer :: i, j, k
    real(8) :: c00,c10,c20,c30,c01,c11,c21,c31,c02,c12,c22,c32,c03,c13,c23,c33

#ifdef CUDA
    dimGrid  = dim3((max-1)/4/THREAD_X + 1, (max-1)/4/THREAD_Y+ 1, 1)
    dimBlock = dim3(THREAD_X, THREAD_Y, 1)
    call matmul_unroll_kernel<<<dimGrid, dimBlock>>>(a, b, c, max)
#else
!$acc kernels present(a,b,c)
#ifdef PGI
!$acc loop gang vector(THREAD_Y)
#elif CAPS
!$acc loop gang(max/4)
#elif CRAY
!$acc loop gang
#endif
    do j = 1, max, 4
#ifdef PGI
!$acc loop gang vector(THREAD_X)
#elif CAPS
!$acc loop worker(THREAD_X)
#elif CRAY
!$acc loop vector(THREAD_X)
#endif
       do i = 1, max, 4
          c00 = 0
          c10 = 0
          c20 = 0
          c30 = 0
          c01 = 0
          c11 = 0
          c21 = 0
          c31 = 0
          c02 = 0
          c12 = 0
          c22 = 0
          c32 = 0
          c03 = 0
          c13 = 0
          c23 = 0
          c33 = 0
          do k = 1, max, 16
             c00 = c00 + a(i  ,k  ) * b(k  ,j  )
             c10 = c10 + a(i+1,k  ) * b(k  ,j  )
             c20 = c20 + a(i+2,k  ) * b(k  ,j  )
             c30 = c30 + a(i+3,k  ) * b(k  ,j  )
             c01 = c01 + a(i  ,k  ) * b(k  ,j+1)
             c11 = c11 + a(i+1,k  ) * b(k  ,j+1)
             c21 = c21 + a(i+2,k  ) * b(k  ,j+1)
             c31 = c31 + a(i+3,k  ) * b(k  ,j+1)
             c02 = c02 + a(i  ,k  ) * b(k  ,j+2)
             c12 = c12 + a(i+1,k  ) * b(k  ,j+2)
             c22 = c22 + a(i+2,k  ) * b(k  ,j+2)
             c32 = c32 + a(i+3,k  ) * b(k  ,j+2)
             c03 = c03 + a(i  ,k  ) * b(k  ,j+3)
             c13 = c13 + a(i+1,k  ) * b(k  ,j+3)
             c23 = c23 + a(i+2,k  ) * b(k  ,j+3)
             c33 = c33 + a(i+3,k  ) * b(k  ,j+3)
             c00 = c00 + a(i  ,k+1) * b(k+1,j  )
             c10 = c10 + a(i+1,k+1) * b(k+1,j  )
             c20 = c20 + a(i+2,k+1) * b(k+1,j  )
             c30 = c30 + a(i+3,k+1) * b(k+1,j  )
             c01 = c01 + a(i  ,k+1) * b(k+1,j+1)
             c11 = c11 + a(i+1,k+1) * b(k+1,j+1)
             c21 = c21 + a(i+2,k+1) * b(k+1,j+1)
             c31 = c31 + a(i+3,k+1) * b(k+1,j+1)
             c02 = c02 + a(i  ,k+1) * b(k+1,j+2)
             c12 = c12 + a(i+1,k+1) * b(k+1,j+2)
             c22 = c22 + a(i+2,k+1) * b(k+1,j+2)
             c32 = c32 + a(i+3,k+1) * b(k+1,j+2)
             c03 = c03 + a(i  ,k+1) * b(k+1,j+3)
             c13 = c13 + a(i+1,k+1) * b(k+1,j+3)
             c23 = c23 + a(i+2,k+1) * b(k+1,j+3)
             c33 = c33 + a(i+3,k+1) * b(k+1,j+3)
             c00 = c00 + a(i  ,k+2) * b(k+2,j  )
             c10 = c10 + a(i+1,k+2) * b(k+2,j  )
             c20 = c20 + a(i+2,k+2) * b(k+2,j  )
             c30 = c30 + a(i+3,k+2) * b(k+2,j  )
             c01 = c01 + a(i  ,k+2) * b(k+2,j+1)
             c11 = c11 + a(i+1,k+2) * b(k+2,j+1)
             c21 = c21 + a(i+2,k+2) * b(k+2,j+1)
             c31 = c31 + a(i+3,k+2) * b(k+2,j+1)
             c02 = c02 + a(i  ,k+2) * b(k+2,j+2)
             c12 = c12 + a(i+1,k+2) * b(k+2,j+2)
             c22 = c22 + a(i+2,k+2) * b(k+2,j+2)
             c32 = c32 + a(i+3,k+2) * b(k+2,j+2)
             c03 = c03 + a(i  ,k+2) * b(k+2,j+3)
             c13 = c13 + a(i+1,k+2) * b(k+2,j+3)
             c23 = c23 + a(i+2,k+2) * b(k+2,j+3)
             c33 = c33 + a(i+3,k+2) * b(k+2,j+3)
             c00 = c00 + a(i  ,k+3) * b(k+3,j  )
             c10 = c10 + a(i+1,k+3) * b(k+3,j  )
             c20 = c20 + a(i+2,k+3) * b(k+3,j  )
             c30 = c30 + a(i+3,k+3) * b(k+3,j  )
             c01 = c01 + a(i  ,k+3) * b(k+3,j+1)
             c11 = c11 + a(i+1,k+3) * b(k+3,j+1)
             c21 = c21 + a(i+2,k+3) * b(k+3,j+1)
             c31 = c31 + a(i+3,k+3) * b(k+3,j+1)
             c02 = c02 + a(i  ,k+3) * b(k+3,j+2)
             c12 = c12 + a(i+1,k+3) * b(k+3,j+2)
             c22 = c22 + a(i+2,k+3) * b(k+3,j+2)
             c32 = c32 + a(i+3,k+3) * b(k+3,j+2)
             c03 = c03 + a(i  ,k+3) * b(k+3,j+3)
             c13 = c13 + a(i+1,k+3) * b(k+3,j+3)
             c23 = c23 + a(i+2,k+3) * b(k+3,j+3)
             c33 = c33 + a(i+3,k+3) * b(k+3,j+3)
             c00 = c00 + a(i  ,k+4) * b(k+4,j  )
             c10 = c10 + a(i+1,k+4) * b(k+4,j  )
             c20 = c20 + a(i+2,k+4) * b(k+4,j  )
             c30 = c30 + a(i+3,k+4) * b(k+4,j  )
             c01 = c01 + a(i  ,k+4) * b(k+4,j+1)
             c11 = c11 + a(i+1,k+4) * b(k+4,j+1)
             c21 = c21 + a(i+2,k+4) * b(k+4,j+1)
             c31 = c31 + a(i+3,k+4) * b(k+4,j+1)
             c02 = c02 + a(i  ,k+4) * b(k+4,j+2)
             c12 = c12 + a(i+1,k+4) * b(k+4,j+2)
             c22 = c22 + a(i+2,k+4) * b(k+4,j+2)
             c32 = c32 + a(i+3,k+4) * b(k+4,j+2)
             c03 = c03 + a(i  ,k+4) * b(k+4,j+3)
             c13 = c13 + a(i+1,k+4) * b(k+4,j+3)
             c23 = c23 + a(i+2,k+4) * b(k+4,j+3)
             c33 = c33 + a(i+3,k+4) * b(k+4,j+3)
             c00 = c00 + a(i  ,k+5) * b(k+5,j  )
             c10 = c10 + a(i+1,k+5) * b(k+5,j  )
             c20 = c20 + a(i+2,k+5) * b(k+5,j  )
             c30 = c30 + a(i+3,k+5) * b(k+5,j  )
             c01 = c01 + a(i  ,k+5) * b(k+5,j+1)
             c11 = c11 + a(i+1,k+5) * b(k+5,j+1)
             c21 = c21 + a(i+2,k+5) * b(k+5,j+1)
             c31 = c31 + a(i+3,k+5) * b(k+5,j+1)
             c02 = c02 + a(i  ,k+5) * b(k+5,j+2)
             c12 = c12 + a(i+1,k+5) * b(k+5,j+2)
             c22 = c22 + a(i+2,k+5) * b(k+5,j+2)
             c32 = c32 + a(i+3,k+5) * b(k+5,j+2)
             c03 = c03 + a(i  ,k+5) * b(k+5,j+3)
             c13 = c13 + a(i+1,k+5) * b(k+5,j+3)
             c23 = c23 + a(i+2,k+5) * b(k+5,j+3)
             c33 = c33 + a(i+3,k+5) * b(k+5,j+3)
             c00 = c00 + a(i  ,k+6) * b(k+6,j  )
             c10 = c10 + a(i+1,k+6) * b(k+6,j  )
             c20 = c20 + a(i+2,k+6) * b(k+6,j  )
             c30 = c30 + a(i+3,k+6) * b(k+6,j  )
             c01 = c01 + a(i  ,k+6) * b(k+6,j+1)
             c11 = c11 + a(i+1,k+6) * b(k+6,j+1)
             c21 = c21 + a(i+2,k+6) * b(k+6,j+1)
             c31 = c31 + a(i+3,k+6) * b(k+6,j+1)
             c02 = c02 + a(i  ,k+6) * b(k+6,j+2)
             c12 = c12 + a(i+1,k+6) * b(k+6,j+2)
             c22 = c22 + a(i+2,k+6) * b(k+6,j+2)
             c32 = c32 + a(i+3,k+6) * b(k+6,j+2)
             c03 = c03 + a(i  ,k+6) * b(k+6,j+3)
             c13 = c13 + a(i+1,k+6) * b(k+6,j+3)
             c23 = c23 + a(i+2,k+6) * b(k+6,j+3)
             c33 = c33 + a(i+3,k+6) * b(k+6,j+3)
             c00 = c00 + a(i  ,k+7) * b(k+7,j  )
             c10 = c10 + a(i+1,k+7) * b(k+7,j  )
             c20 = c20 + a(i+2,k+7) * b(k+7,j  )
             c30 = c30 + a(i+3,k+7) * b(k+7,j  )
             c01 = c01 + a(i  ,k+7) * b(k+7,j+1)
             c11 = c11 + a(i+1,k+7) * b(k+7,j+1)
             c21 = c21 + a(i+2,k+7) * b(k+7,j+1)
             c31 = c31 + a(i+3,k+7) * b(k+7,j+1)
             c02 = c02 + a(i  ,k+7) * b(k+7,j+2)
             c12 = c12 + a(i+1,k+7) * b(k+7,j+2)
             c22 = c22 + a(i+2,k+7) * b(k+7,j+2)
             c32 = c32 + a(i+3,k+7) * b(k+7,j+2)
             c03 = c03 + a(i  ,k+7) * b(k+7,j+3)
             c13 = c13 + a(i+1,k+7) * b(k+7,j+3)
             c23 = c23 + a(i+2,k+7) * b(k+7,j+3)
             c33 = c33 + a(i+3,k+7) * b(k+7,j+3)
             c00 = c00 + a(i  ,k+8) * b(k+8,j  )
             c10 = c10 + a(i+1,k+8) * b(k+8,j  )
             c20 = c20 + a(i+2,k+8) * b(k+8,j  )
             c30 = c30 + a(i+3,k+8) * b(k+8,j  )
             c01 = c01 + a(i  ,k+8) * b(k+8,j+1)
             c11 = c11 + a(i+1,k+8) * b(k+8,j+1)
             c21 = c21 + a(i+2,k+8) * b(k+8,j+1)
             c31 = c31 + a(i+3,k+8) * b(k+8,j+1)
             c02 = c02 + a(i  ,k+8) * b(k+8,j+2)
             c12 = c12 + a(i+1,k+8) * b(k+8,j+2)
             c22 = c22 + a(i+2,k+8) * b(k+8,j+2)
             c32 = c32 + a(i+3,k+8) * b(k+8,j+2)
             c03 = c03 + a(i  ,k+8) * b(k+8,j+3)
             c13 = c13 + a(i+1,k+8) * b(k+8,j+3)
             c23 = c23 + a(i+2,k+8) * b(k+8,j+3)
             c33 = c33 + a(i+3,k+8) * b(k+8,j+3)
             c00 = c00 + a(i  ,k+9) * b(k+9,j  )
             c10 = c10 + a(i+1,k+9) * b(k+9,j  )
             c20 = c20 + a(i+2,k+9) * b(k+9,j  )
             c30 = c30 + a(i+3,k+9) * b(k+9,j  )
             c01 = c01 + a(i  ,k+9) * b(k+9,j+1)
             c11 = c11 + a(i+1,k+9) * b(k+9,j+1)
             c21 = c21 + a(i+2,k+9) * b(k+9,j+1)
             c31 = c31 + a(i+3,k+9) * b(k+9,j+1)
             c02 = c02 + a(i  ,k+9) * b(k+9,j+2)
             c12 = c12 + a(i+1,k+9) * b(k+9,j+2)
             c22 = c22 + a(i+2,k+9) * b(k+9,j+2)
             c32 = c32 + a(i+3,k+9) * b(k+9,j+2)
             c03 = c03 + a(i  ,k+9) * b(k+9,j+3)
             c13 = c13 + a(i+1,k+9) * b(k+9,j+3)
             c23 = c23 + a(i+2,k+9) * b(k+9,j+3)
             c33 = c33 + a(i+3,k+9) * b(k+9,j+3)
             c00 = c00 + a(i  ,k+10) * b(k+10,j  )
             c10 = c10 + a(i+1,k+10) * b(k+10,j  )
             c20 = c20 + a(i+2,k+10) * b(k+10,j  )
             c30 = c30 + a(i+3,k+10) * b(k+10,j  )
             c01 = c01 + a(i  ,k+10) * b(k+10,j+1)
             c11 = c11 + a(i+1,k+10) * b(k+10,j+1)
             c21 = c21 + a(i+2,k+10) * b(k+10,j+1)
             c31 = c31 + a(i+3,k+10) * b(k+10,j+1)
             c02 = c02 + a(i  ,k+10) * b(k+10,j+2)
             c12 = c12 + a(i+1,k+10) * b(k+10,j+2)
             c22 = c22 + a(i+2,k+10) * b(k+10,j+2)
             c32 = c32 + a(i+3,k+10) * b(k+10,j+2)
             c03 = c03 + a(i  ,k+10) * b(k+10,j+3)
             c13 = c13 + a(i+1,k+10) * b(k+10,j+3)
             c23 = c23 + a(i+2,k+10) * b(k+10,j+3)
             c33 = c33 + a(i+3,k+10) * b(k+10,j+3)
             c00 = c00 + a(i  ,k+11) * b(k+11,j  )
             c10 = c10 + a(i+1,k+11) * b(k+11,j  )
             c20 = c20 + a(i+2,k+11) * b(k+11,j  )
             c30 = c30 + a(i+3,k+11) * b(k+11,j  )
             c01 = c01 + a(i  ,k+11) * b(k+11,j+1)
             c11 = c11 + a(i+1,k+11) * b(k+11,j+1)
             c21 = c21 + a(i+2,k+11) * b(k+11,j+1)
             c31 = c31 + a(i+3,k+11) * b(k+11,j+1)
             c02 = c02 + a(i  ,k+11) * b(k+11,j+2)
             c12 = c12 + a(i+1,k+11) * b(k+11,j+2)
             c22 = c22 + a(i+2,k+11) * b(k+11,j+2)
             c32 = c32 + a(i+3,k+11) * b(k+11,j+2)
             c03 = c03 + a(i  ,k+11) * b(k+11,j+3)
             c13 = c13 + a(i+1,k+11) * b(k+11,j+3)
             c23 = c23 + a(i+2,k+11) * b(k+11,j+3)
             c33 = c33 + a(i+3,k+11) * b(k+11,j+3)
             c00 = c00 + a(i  ,k+12) * b(k+12,j  )
             c10 = c10 + a(i+1,k+12) * b(k+12,j  )
             c20 = c20 + a(i+2,k+12) * b(k+12,j  )
             c30 = c30 + a(i+3,k+12) * b(k+12,j  )
             c01 = c01 + a(i  ,k+12) * b(k+12,j+1)
             c11 = c11 + a(i+1,k+12) * b(k+12,j+1)
             c21 = c21 + a(i+2,k+12) * b(k+12,j+1)
             c31 = c31 + a(i+3,k+12) * b(k+12,j+1)
             c02 = c02 + a(i  ,k+12) * b(k+12,j+2)
             c12 = c12 + a(i+1,k+12) * b(k+12,j+2)
             c22 = c22 + a(i+2,k+12) * b(k+12,j+2)
             c32 = c32 + a(i+3,k+12) * b(k+12,j+2)
             c03 = c03 + a(i  ,k+12) * b(k+12,j+3)
             c13 = c13 + a(i+1,k+12) * b(k+12,j+3)
             c23 = c23 + a(i+2,k+12) * b(k+12,j+3)
             c33 = c33 + a(i+3,k+12) * b(k+12,j+3)
             c00 = c00 + a(i  ,k+13) * b(k+13,j  )
             c10 = c10 + a(i+1,k+13) * b(k+13,j  )
             c20 = c20 + a(i+2,k+13) * b(k+13,j  )
             c30 = c30 + a(i+3,k+13) * b(k+13,j  )
             c01 = c01 + a(i  ,k+13) * b(k+13,j+1)
             c11 = c11 + a(i+1,k+13) * b(k+13,j+1)
             c21 = c21 + a(i+2,k+13) * b(k+13,j+1)
             c31 = c31 + a(i+3,k+13) * b(k+13,j+1)
             c02 = c02 + a(i  ,k+13) * b(k+13,j+2)
             c12 = c12 + a(i+1,k+13) * b(k+13,j+2)
             c22 = c22 + a(i+2,k+13) * b(k+13,j+2)
             c32 = c32 + a(i+3,k+13) * b(k+13,j+2)
             c03 = c03 + a(i  ,k+13) * b(k+13,j+3)
             c13 = c13 + a(i+1,k+13) * b(k+13,j+3)
             c23 = c23 + a(i+2,k+13) * b(k+13,j+3)
             c33 = c33 + a(i+3,k+13) * b(k+13,j+3)
             c00 = c00 + a(i  ,k+14) * b(k+14,j  )
             c10 = c10 + a(i+1,k+14) * b(k+14,j  )
             c20 = c20 + a(i+2,k+14) * b(k+14,j  )
             c30 = c30 + a(i+3,k+14) * b(k+14,j  )
             c01 = c01 + a(i  ,k+14) * b(k+14,j+1)
             c11 = c11 + a(i+1,k+14) * b(k+14,j+1)
             c21 = c21 + a(i+2,k+14) * b(k+14,j+1)
             c31 = c31 + a(i+3,k+14) * b(k+14,j+1)
             c02 = c02 + a(i  ,k+14) * b(k+14,j+2)
             c12 = c12 + a(i+1,k+14) * b(k+14,j+2)
             c22 = c22 + a(i+2,k+14) * b(k+14,j+2)
             c32 = c32 + a(i+3,k+14) * b(k+14,j+2)
             c03 = c03 + a(i  ,k+14) * b(k+14,j+3)
             c13 = c13 + a(i+1,k+14) * b(k+14,j+3)
             c23 = c23 + a(i+2,k+14) * b(k+14,j+3)
             c33 = c33 + a(i+3,k+14) * b(k+14,j+3)
             c00 = c00 + a(i  ,k+15) * b(k+15,j  )
             c10 = c10 + a(i+1,k+15) * b(k+15,j  )
             c20 = c20 + a(i+2,k+15) * b(k+15,j  )
             c30 = c30 + a(i+3,k+15) * b(k+15,j  )
             c01 = c01 + a(i  ,k+15) * b(k+15,j+1)
             c11 = c11 + a(i+1,k+15) * b(k+15,j+1)
             c21 = c21 + a(i+2,k+15) * b(k+15,j+1)
             c31 = c31 + a(i+3,k+15) * b(k+15,j+1)
             c02 = c02 + a(i  ,k+15) * b(k+15,j+2)
             c12 = c12 + a(i+1,k+15) * b(k+15,j+2)
             c22 = c22 + a(i+2,k+15) * b(k+15,j+2)
             c32 = c32 + a(i+3,k+15) * b(k+15,j+2)
             c03 = c03 + a(i  ,k+15) * b(k+15,j+3)
             c13 = c13 + a(i+1,k+15) * b(k+15,j+3)
             c23 = c23 + a(i+2,k+15) * b(k+15,j+3)
             c33 = c33 + a(i+3,k+15) * b(k+15,j+3)
          end do
          c(i  ,j  ) = c00
          c(i+1,j  ) = c10
          c(i+2,j  ) = c20
          c(i+3,j  ) = c30
          c(i  ,j+1) = c01
          c(i+1,j+1) = c11
          c(i+2,j+1) = c21
          c(i+3,j+1) = c31
          c(i  ,j+2) = c02
          c(i+1,j+2) = c12
          c(i+2,j+2) = c22
          c(i+3,j+2) = c32
          c(i  ,j+3) = c03
          c(i+1,j+3) = c13
          c(i+2,j+3) = c23
          c(i+3,j+3) = c33
       end do
    end do
!$acc end kernels
#endif !CUDA
  end subroutine matmul_unroll
#endif


#ifdef CUDA
#ifdef SHARED
  subroutine matmul_shared(a, b, c, max)
    integer, intent(in) :: max
    real(8), device, dimension(max,max), intent(in) :: a
    real(8), device, dimension(max,max), intent(in) :: b
    real(8), device, dimension(max,max), intent(inout) :: c
    type(dim3) :: dimGrid, dimBlock
    integer :: i, j, k
    real(8) :: cc

    dimGrid  = dim3((max-1)/16 + 1, (max-1)/4/16+ 1, 1)
    dimBlock = dim3(16, 16, 1)
    call matmul_shared_kernel<<<dimGrid, dimBlock>>>(a, b, c, max)

  end subroutine matmul_shared
#endif !SHARED
#endif !CUDA


end module matmul_mod


program matmul

  use matmul_mod
  implicit none

  integer max
  max = DATASIZE
  call main(max)

end program matmul


