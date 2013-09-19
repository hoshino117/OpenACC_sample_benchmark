# 1 "diffusion.fpp"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! if you type following comand,
!!!! > fpp -DCUDA(PGI,1,CRAY) diffusion.fpp diffusion_cuda(pgi,caps,cray).cuf(f90)
!!!! you can get CUDA(OpenACC for PGI,1,CRAY) code.
!!!! if you don't have fpp,
!!!! > cp diffusion.fpp diffusion.cuf(diffusion.f90)
!!!! > pgfortran -Mpreprocess -DPGI -MCUDA ${OPTIONS} diffusion.cuf -o diffusion_cuda
!!!! > pgfortran -Mpreprocess -DPGI -acc   ${OPTIONS} diffusion.f90 -o diffusion_pgi
!!!! > hmpp ${HMPP_OPTIONS} ifort(gfortran) -cpp      diffusion.f90 -o diffusion_caps
!!!! > ftn       -e F         -DCRAY       ${OPTIONS} diffusion.f90 -o diffusion_cray
!!!! also you can get CUDA(OpenACC) binary file.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


module stencil_mod

# 19


# 30


# 34



# 54


contains 

  subroutine init(buff,nx,ny,nz,kx,ky,kz,dx,dy,dz,kappa,time)
    integer, intent(in) :: nx, ny, nz
    real, dimension(1:nx,1:ny,1:nz,1:2) :: buff
    real, intent(in) :: kx, ky, kz
    real, intent(in) :: dx, dy, dz
    real, intent(in) :: kappa, time
    real :: ax, ay, az
    integer :: jz, jy, jx, j
    real :: x, y, z, f0

    ax = exp(-kappa*time*(kx*kx))
    ay = exp(-kappa*time*(ky*ky))
    az = exp(-kappa*time*(kz*kz))
    do j = 1, 2
       do jz = 1, nz
          do jy = 1, ny
             do jx = 1, nx
                x = dx * real(jx - 0.5)
                y = dy * real(jy - 0.5)
                z = dz * real(jz - 0.5)
                f0 = real(0.125) * (1.0 - ax*cos(kx*x)) * (1.0 - ay*cos(ky*y)) * (1.0 - az*cos(kz*z))
                buff(jx,jy,jz,j) = f0
             end do
          end do
       end do
    end do

  end subroutine init


  real function accuracy(b1, b2, sw, nx, ny, nz)
    integer, intent(in) :: nx, ny, nz, sw
    real,dimension(nx,ny,nz,2) :: b1,b2
    real :: err
    integer :: i,j,k
    err = 0.0
    do k = 1,nz
       do j = 1,ny
          do i = 1,nx
             err = err + (b1(i,j,k,sw) - b2(i,j,k,sw)) * (b1(i,j,k,sw) - b2(i,j,k,sw))
          end do
       end do
    end do
    accuracy = real(sqrt(err/(nx * ny * nz)))
  end function accuracy


  subroutine main(nx,ny,nz)
    integer,intent(in) :: nx, ny, nz
    real :: time
    integer :: count
    real :: l, dx, dy, dz, kx, ky, kz, kappa, dt
    real :: ce, cw, cn, cs, ct, cb, cc
    real,dimension(1:nx,1:ny,1:nz,1:2) :: f,answer
# 115

    real :: err
    real(8) :: gflops,thput,elapsed_time
    integer :: t1, t2, t_rate, t_max, diff
    integer :: sw 
# 123

    time = 0.0
    count = 0
    l = 1.0
    kappa = 0.1
    dx = l / nx
    dy = l / ny
    dz = l / nz
    kx = 2.0 * 3.1415926535897932384626
    ky = kx
    kz = kx

    dt = 0.1*dx*dx / kappa
    ce = kappa*dt/(dx*dx)
    cw = ce
    cn = kappa*dt/(dy*dy)
    cs = cn
    ct = kappa*dt/(dz*dz)
    cb = ct
    cc = 1.0 - (ce + cw + cn + cs + ct + cb)

    print "(A)", "optimization type, data size[^3], elapsed time[s], flops[GFlops], throughput[GB/s], accuracy, thread_x size, thread_y size"
  

    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

# 158


!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)
    call diffusion_baseline(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
# 166

!$acc wait
!PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data
!CUDA

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    elapsed_time = diff/dble(t_rate)

    call init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)
    sw = mod(count, 2) + 1
    err = accuracy(f,answer,sw, nx, ny, nz)
    gflops = (nx*ny*nz)*13.0*count/elapsed_time / (1024.0**3)
    thput  = (nx*ny*nz)*4*2.0*count/elapsed_time / (1024.0**3)

# 188

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A)", "baseline,", 256, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ", chosen by compiler, chosen by compiler"


!1



    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

# 207

!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)

    call diffusion_thread(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
# 215

!$acc wait
!PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data

!CUDA
    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    elapsed_time = diff/dble(t_rate)

    call init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)
    sw = mod(count, 2) + 1
    err = accuracy(f,answer,sw, nx, ny, nz)
    gflops = (nx*ny*nz)*13.0*count/elapsed_time / (1024.0**3)
    thput  = (nx*ny*nz)*4*2.0*count/elapsed_time / (1024.0**3)

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A, I3, A, I3)", "opt thread block,", 256, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ",", 256, "," ,1

!1




    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)

    call diffusion_interchange(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
# 252

!$acc wait
!PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    elapsed_time = diff/dble(t_rate)

    call init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)
    sw = mod(count, 2) + 1
    err = accuracy(f,answer,sw, nx, ny, nz)
    gflops = (nx*ny*nz)*13.0*count/elapsed_time / (1024.0**3)
    thput  = (nx*ny*nz)*4*2.0*count/elapsed_time / (1024.0**3)

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A, I3, A, I3)", "loop interchange,", 256, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ",", 256, "," ,1

!1
!CUDA



    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

# 289

!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)

    call diffusion_peeling(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
# 297

!$acc wait
!PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data
!CUDA

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    elapsed_time = diff/dble(t_rate)

    call init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)
    sw = mod(count, 2) + 1
    err = accuracy(f,answer,sw, nx, ny, nz)
    gflops = (nx*ny*nz)*13.0*count/elapsed_time / (1024.0**3)
    thput  = (nx*ny*nz)*4*2.0*count/elapsed_time / (1024.0**3)

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A, I3, A, I3)", "branch hoisting,", 256, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ",", 256, "," ,1

!1



    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

# 334

!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)

    call diffusion_register_blocking(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
# 342

!$acc wait
!PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data
!CUDA

    if ( t2 < t1 ) then
       diff = t_max - t1 + t2
    else
       diff = t2 - t1
    endif
    elapsed_time = diff/dble(t_rate)

    call init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)
    sw = mod(count, 2) + 1
    err = accuracy(f,answer,sw, nx, ny, nz)
    gflops = (nx*ny*nz)*13.0*count/elapsed_time / (1024.0**3)
    thput  = (nx*ny*nz)*4*2.0*count/elapsed_time / (1024.0**3)

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A, I3, A, I3)", "register blocking,", 256, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ",", 256, "," ,1

!1


  end subroutine main



! single
  subroutine diffusion_cpu(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, dimension(1:nx,1:ny,1:nz,1:2) :: f
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
       do z = 1,nz
          do y = 1,ny
             do x = 1,nx
                w = -1; e = 1; n = -1; s = 1; b = -1; t = 1;
                if(x == 1)  w = 0
                if(x == nx) e = 0
                if(y == 1)  n = 0
                if(y == ny) s = 0
                if(z == 1)  b = 0
                if(z == nz) t = 0
                f(x,y,z,2-sw) = cc * f(x,y,z,1+sw) + cw * f(x+w,y,z,1+sw) &
                     + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) + cn * f(x,y+n,z,1+sw) &
                     + cb * f(x,y,z+b,1+sw) + ct * f(x,y,z+t,1+sw)
             end do
          end do
       end do
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_cpu

# 512
!CUDA


  subroutine diffusion_baseline(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
# 523

    real, dimension(1:nx,1:ny,1:nz,1:2) :: f

    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
# 537

!$acc kernels present(f)
# 541

!$acc loop
# 545

       do z = 1,nz
# 549

!$acc loop
# 553

          do y = 1,ny
# 557

!$acc loop
# 561

             do x = 1,nx
                w = -1; e = 1; n = -1; s = 1; b = -1; t = 1;
                if(x == 1)  w = 0
                if(x == nx) e = 0
                if(y == 1)  n = 0
                if(y == ny) s = 0
                if(z == 1)  b = 0
                if(z == nz) t = 0
                f(x,y,z,2-sw) = cc * f(x,y,z,1+sw) + cw * f(x+w,y,z,1+sw) &
                     + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) + cn * f(x,y+n,z,1+sw) &
                     + cb * f(x,y,z+b,1+sw) + ct * f(x,y,z+t,1+sw)
             end do
          end do
       end do
!$acc end kernels
!CUDA
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_baseline
!1



  subroutine diffusion_thread(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
# 595

    real, dimension(1:nx,1:ny,1:nz,1:2) :: f

    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
# 609

!$acc kernels present(f)
# 613

!$acc loop
# 617

       do z = 1,nz
# 621

!$acc loop gang(ny)
# 625

          do y = 1,ny
# 629

!$acc loop worker(256)
# 633

             do x = 1,nx
                w = -1; e = 1; n = -1; s = 1; b = -1; t = 1;
                if(x == 1)  w = 0
                if(x == nx) e = 0
                if(y == 1)  n = 0
                if(y == ny) s = 0
                if(z == 1)  b = 0
                if(z == nz) t = 0
                f(x,y,z,2-sw) = cc * f(x,y,z,1+sw) + cw * f(x+w,y,z,1+sw) &
                     + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) + cn * f(x,y+n,z,1+sw) &
                     + cb * f(x,y,z+b,1+sw) + ct * f(x,y,z+t,1+sw)
             end do
          end do
       end do
!$acc end kernels
!CUDA
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_thread
!1



  subroutine diffusion_interchange(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, dimension(1:nx,1:ny,1:nz,1:2) :: f
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
!$acc kernels present(f)
# 675

!$acc loop gang(ny)
# 679

       do y = 1,ny
# 683

!$acc loop worker(256)
# 687

          do x = 1,nx
!$acc loop seq
             do z = 1,nz
                w = -1; e = 1; n = -1; s = 1; b = -1; t = 1;
                if(x == 1)  w = 0
                if(x == nx) e = 0
                if(y == 1)  n = 0
                if(y == ny) s = 0
                if(z == 1)  b = 0
                if(z == nz) t = 0
                f(x,y,z,2-sw) = cc * f(x,y,z,1+sw) + cw * f(x+w,y,z,sw+1) &
                     + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) + cn * f(x,y+n,z,1+sw) &
                     + cb * f(x,y,z+b,1+sw) + ct * f(x,y,z+t,1+sw)
             end do
          end do
       end do
!$acc end kernels
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_interchange
!1



  subroutine diffusion_peeling(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
# 722

    real, dimension(1:nx,1:ny,1:nz,1:2) :: f

    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
# 736

!$acc kernels present(f)
# 740

!$acc loop gang(ny)
# 744

       do y = 1,ny
# 748

!$acc loop worker(256)
# 752

          do x = 1,nx
             z = 1
             w = -1; e = 1; n = -1; s = 1;
             if(x == 1)  w = 0
             if(x == nx) e = 0
             if(y == 1)  n = 0
             if(y == ny) s = 0
             f(x,y,z,2-sw) = cc * f(x,y,z,1+sw) + cw * f(x+w,y,z,1+sw) &
                  + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) + cn * f(x,y+n,z,1+sw) &
                  + cb * f(x,y,z,1+sw) + ct * f(x,y,z+1,1+sw)
             do z = 2,nz-1
                f(x,y,z,2-sw) = cc * f(x,y,z,1+sw) + cw * f(x+w,y,z,1+sw) &
                     + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) + cn * f(x,y+n,z,1+sw) + &
                     cb * f(x,y,z-1,1+sw) + ct * f(x,y,z+1,1+sw)
             end do
             z = nz
             f(x,y,z,2-sw) = cc * f(x,y,z,1+sw) + cw * f(x+w,y,z,1+sw) &
                  + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) + cn * f(x,y+n,z,1+sw) &
                  + cb * f(x,y,z-1,1+sw) + ct * f(x,y,z,1+sw)
          end do
       end do
!$acc end kernels
!CUDA
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_peeling
!1



  subroutine diffusion_register_blocking(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
# 793

    real, dimension(1:nx,1:ny,1:nz,1:2) :: f

    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    real :: f_t, f_c, f_b
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
# 808

!$acc kernels present(f)
# 812

!$acc loop gang(ny)
# 816

       do y = 1,ny
# 820

!$acc loop worker(256)
# 824

          do x = 1,nx
             z = 1
             w = -1; e = 1; n = -1; s = 1;
             
             if(x == 1)  w = 0
             if(x == nx) e = 0
             if(y == 1)  n = 0
             if(y == ny) s = 0
             t = 1

             f_t = f(x,y,z+t,1+sw) 
             f_c = f(x,y,z,1+sw)
             f_b = f_c
             f(x,y,z,2-sw) = cc * f_c + cw * f(x+w,y,z,1+sw) &
                  + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) &
                  + cn * f(x,y+n,z,1+sw) + cb * f_b + ct * f_t
             do z = 2,nz-1
                f_b = f_c
                f_c = f_t
                f_t = f(x,y,z+t,1+sw) 
                f(x,y,z,2-sw) = cc * f_c + cw * f(x+w,y,z,1+sw) &
                     + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) &
                     + cn * f(x,y+n,z,1+sw) + cb * f_b + ct * f_t
             end do
             z = nz
             f_b = f_c
             f_c = f_t
             f_t = f_t
             f(x,y,z,2-sw) = cc * f_c + cw * f(x+w,y,z,1+sw) &
                  + ce * f(x+e,y,z,1+sw) + cs * f(x,y+s,z,1+sw) &
                  + cn * f(x,y+n,z,1+sw) + cb * f_b + ct * f_t
          end do
       end do
!$acc end kernels
!CUDA
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_register_blocking
!1


end module stencil_mod


program stencil

  use stencil_mod
  implicit none

  integer :: NX, NY, NZ
  NX = 256
  NY = 256
  NZ = 256

  call main(NX, NY, NZ)

end program stencil

