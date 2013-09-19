# 1 "matmul.fpp"
program matmul_mine             














# 28




  implicit none

  integer max
  max = 2048
  call main(max)

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
    real(8), dimension(max,max) :: ccpu
    integer :: t1, t2, t_rate, t_max, diff
    real(8) :: mmmax,err
    integer :: i,j,k


    integer :: devicesync
    devicesync = 0


    call random_number(a)
    call random_number(b)
    c(:,:) = 0
    ccpu(:,:) = 0

# 75



!$acc data copy(devicesync)
!$acc end data
# 82


# 115



    c(:,:) = 0

!$acc data &
!$acc copy  (c(1:max,1:max)) &
!$acc copyin(a(1:max,1:max)) &
!$acc copyin(b(1:max,1:max))
    call system_clock(t1)
    call mymatmul_thread(a, b, c, max)


!$acc data copy(devicesync)
!$acc end data
# 132

    call system_clock(t2, t_rate, t_max)
!$acc end data

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    mmmax = ((max * 1.0e-09) * max) * (max-1) * 2 / (diff / dble(t_rate)) 
    print "(A,I5,A,I5,A,F10.6,A,F10.6,A,I3,A,I3)", "optimize thread block size,size,", max, ", * ,", max,",time,",diff/dble(t_rate),",flops,",mmmax,",thread block,",64,",",4


# 149




# 186


# 221



  end subroutine main


# 251




  subroutine mymatmul_thread(a, b, c, max)
    integer, intent(in) :: max
    real(8), dimension(max,max), intent(in) :: a
    real(8), dimension(max,max), intent(in) :: b
    real(8), dimension(max,max), intent(inout) :: c
    integer :: i, j, k
    real(8) :: cc

!$acc kernels present(a, b, c)

!$acc loop gang vector(4)
# 268


    do j = 1, max

!$acc loop gang vector(64)
# 277

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
  end subroutine mymatmul_thread



# 432


# 753



end program matmul_mine
