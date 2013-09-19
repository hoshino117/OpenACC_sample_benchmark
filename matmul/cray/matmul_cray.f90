# 1 "matmul.fpp"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! if you type following comand,
!!!! > fpp -DCUDA(PGI,CAPS,1) diffusion.fpp diffusion_cuda(pgi,caps,cray).cuf(f90)
!!!! you can get CUDA(OpenACC for PGI,CAPS,1) code.
!!!! if you don't have fpp,
!!!! > cp diffusion.fpp diffusion.cuf(diffusion.f90)
!!!! > pgfortran -Mpreprocess -DPGI -MCUDA ${OPTIONS} diffusion.cuf -o diffusion_cuda
!!!! > pgfortran -Mpreprocess -DPGI -acc   ${OPTIONS} diffusion.f90 -o diffusion_pgi
!!!! > hmpp ${HMPP_OPTIONS} ifort(gfortran) -cpp      diffusion.f90 -o diffusion_caps
!!!! > ftn       -e F         -DCRAY       ${OPTIONS} diffusion.f90 -o diffusion_cray
!!!! also you can get CUDA(OpenACC) binary file.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module matmul_mod


# 19


# 37


# 41



# 54


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
# 80

    real(8), dimension(max,max) :: ccpu
    integer :: t1, t2, t_rate, t_max, diff
    real(8) :: flops,time,err
    integer :: i,j,k

# 89


    call random_number(a)
    call random_number(b)
    c(:,:) = 0
    ccpu(:,:) = 0


    ccpu = matmul(a, b)


    print "(A)", "optimization type, matrix size[^2], time[sec], flops[GFlops], thread_x size, thread_y size"



# 114


!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call matmul_base(a, b, c, max)

# 126

!$acc wait

    call system_clock(t2, t_rate, t_max)
!$acc end data
!CUDA

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)
    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
# 142

    print "(A,I5,A,F10.6,A,F10.6,A,A)", "baseline,", max, ",",time,",",flops,",chosen by compiler",",chosen by compiler"



    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err


!1


    c(:,:) = 0

# 167

!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call matmul_thread(a, b, c, max)

# 178

!$acc wait

    call system_clock(t2, t_rate, t_max)
!$acc end data
!CUDA
    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)
    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
    print "(A,I5,A,F10.6,A,F10.6,A,I3,A,I3)", "opt thread block,", max, ",",time,",",flops,",", 512, ",", 1


    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err





    c(:,:) = 0

# 214

!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call matmul_cache(a, b, c, max)

# 225

!$acc wait

    call system_clock(t2, t_rate, t_max)
!$acc end data
!CUDA

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)
    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
    print "(A,I5,A,F10.6,A,F10.6,A,I3,A,I3)", "cache,", max, ",",time,",",flops,",", 512, ",", 1



    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err




    c(:,:) = 0

# 262

!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call matmul_unroll(a, b, c, max)

# 273

!$acc wait

    call system_clock(t2, t_rate, t_max)
!$acc end data

!CUDA
    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    time = diff/dble(t_rate)
    flops = ((max / (1024.0 * 1024.0 * 1024.0)) * max) * (max-1) * 2 / (time) 
    print "(A,I5,A,F10.6,A,F10.6,A,I3,A,I3)", "unroll,", max, ",",time,",",flops,",", 512, ",", 1



    err = 0
    err = accuracy(c, ccpu, max)
    print "(A, E10.3)", "accuracy :", err



# 327



  end subroutine main

# 799
!CUDA



  subroutine matmul_base(a, b, c, max)
    integer, intent(in) :: max
# 810

    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c

    integer :: i, j, k
    real(8) :: cc

# 822


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
!CUDA
  end subroutine matmul_base
!1



  subroutine matmul_thread(a, b, c, max)
    integer, intent(in) :: max
# 851

    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c

    integer :: i, j, k
    real(8) :: cc

# 863


!$acc kernels present(a, b, c)
# 870

!$acc loop gang

    do j = 1, max
# 878

!$acc loop vector(512)

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
!CUDA
  end subroutine matmul_thread
!1



# 969



  subroutine matmul_cache(a, b, c, max)
    integer, intent(in) :: max
# 979

    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c

    integer :: i, j, k
    real(8) :: c00,c10,c20,c30,c01,c11,c21,c31,c02,c12,c22,c32,c03,c13,c23,c33

# 991

!$acc kernels present(a,b,c)
# 997

!$acc loop gang


    do j = 1, max, 4
# 1006

!$acc loop vector(512)

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
!CUDA

  end subroutine matmul_cache



  subroutine matmul_unroll(a, b, c, max)
    integer, intent(in) :: max
# 1076

    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c

    integer :: i, j, k
    real(8) :: c00,c10,c20,c30,c01,c11,c21,c31,c02,c12,c22,c32,c03,c13,c23,c33

# 1088

!$acc kernels present(a,b,c)
# 1094

!$acc loop gang

    do j = 1, max, 4
# 1102

!$acc loop vector(512)

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
!CUDA
  end subroutine matmul_unroll



# 1421
!CUDA


end module matmul_mod


program matmul

  use matmul_mod
  implicit none

  integer max
  max = 2048
  call main(max)

end program matmul


