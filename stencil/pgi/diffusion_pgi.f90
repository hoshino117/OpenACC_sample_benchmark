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


module stencil_mod

#ifndef DATASIZE
#define DATASIZE 128
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
#define THREAD_X 128
#endif
#define THREAD_Y 1

#elif CRAY

#ifndef THREAD_X
#define THREAD_X 128
#endif
#define THREAD_Y 1

#elif CUDA

  use cudafor
#ifndef THREAD_X
#define THREAD_X 64
#endif
#ifndef THREAD_Y
#define THREAD_Y 2
#endif

#endif

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
#ifdef CUDA
    real,device,dimension(1:nx,1:ny,1:nz,1:2) :: devf
    integer :: ierror
#endif
    real :: err
    real(8) :: gflops,thput,elapsed_time
    integer :: t1, t2, t_rate, t_max, diff
    integer :: sw 
#ifdef PGI
    integer devicesync
    devicesync = 0
#endif
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
  
#ifdef BASE
    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

#ifdef CUDA
    devf(:,:,:,:) = f(:,:,:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call diffusion_baseline(devf, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    f(:,:,:,:) = devf(:,:,:,:)
#else

!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)
    call diffusion_baseline(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif !PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data
#endif !CUDA

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

#ifdef CUDA
    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A)", "baseline,", DATASIZE, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ", 64, 4"
#else
    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A)", "baseline,", DATASIZE, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ", chosen by compiler, chosen by compiler"
#endif

#endif !BASE 


#ifdef THREAD
    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

#ifdef CUDA
    devf(:,:,:,:) = f(:,:,:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call diffusion_thread(devf, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    f(:,:,:,:) = devf(:,:,:,:)
#else
!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)

    call diffusion_thread(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif !PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data

#endif !CUDA
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

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A, I3, A, I3)", "opt thread block,", DATASIZE, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ",", THREAD_X, "," ,THREAD_Y

#endif !THREAD


#ifndef CUDA
#ifdef INTERCHANGE
    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)

    call diffusion_interchange(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif !PGI
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

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A, I3, A, I3)", "loop interchange,", DATASIZE, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ",", THREAD_X, "," ,THREAD_Y

#endif !INTERCHANGE
#endif !CUDA


#ifdef PEELING
    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

#ifdef CUDA
    devf(:,:,:,:) = f(:,:,:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call diffusion_peeling(devf, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    f(:,:,:,:) = devf(:,:,:,:)
#else
!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)

    call diffusion_peeling(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif !PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data
#endif !CUDA

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

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A, I3, A, I3)", "branch hoisting,", DATASIZE, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ",", THREAD_X, "," ,THREAD_Y

#endif !PEELING


#ifdef REGISTER
    time = 0.0
    call init(f, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time)

#ifdef CUDA
    devf(:,:,:,:) = f(:,:,:,:)
    ierror = cudaDeviceSynchronize()
    call system_clock(t1)
    call diffusion_register_blocking(devf, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
    ierror = cudaDeviceSynchronize()
    call system_clock(t2, t_rate, t_max)
    f(:,:,:,:) = devf(:,:,:,:)
#else
!$acc data copy(f(1:nx,1:ny,1:nz,1:2))
    call system_clock(t1)

    call diffusion_register_blocking(f, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, time, count)
#ifdef PGI
!$acc data copy(devicesync)
!$acc end data
#else
!$acc wait
#endif !PGI
    call system_clock(t2, t_rate, t_max)
!$acc end data
#endif !CUDA

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

    print "(A, I4, A, F10.5, A, F10.5, A, F10.5, A, E12.5, A, I3, A, I3)", "register blocking,", DATASIZE, ",", elapsed_time, ",", gflops, ",", thput, ",", err, ",", THREAD_X, "," ,THREAD_Y

#endif !REGISTER


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

#ifdef CUDA
  attributes(global) subroutine diffusion_baseline_kernel(f,nx,ny,nz,sw,ce,cw,cn,cs,ct,cb,cc)

    integer,value :: nx,ny,nz,sw
    real,dimension(nx,ny,nz,2) :: f
    real,value :: ce,cw,cn,cs,ct,cb,cc
    integer :: x, y, z, c, w, e, n, s, b, t
    integer :: tx, ty

    x = blockdim%x * (blockidx%x-1) + threadidx%x
    y = blockdim%y * (blockidx%y-1) + threadidx%y

    do z = 1, nz
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

  end subroutine diffusion_baseline_kernel


  attributes(global) subroutine diffusion_peeling_kernel(f,nx,ny,nz,sw,ce,cw,cn,cs,ct,cb,cc)

    integer,value :: nx,ny,nz,sw
    real,dimension(nx,ny,nz,2) :: f
    real,value :: ce,cw,cn,cs,ct,cb,cc
    integer :: x, y, z, c, w, e, n, s, b, t
    integer :: tx, ty

    x = blockdim%x * (blockidx%x-1) + threadidx%x
    y = blockdim%y * (blockidx%y-1) + threadidx%y

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
    
  end subroutine diffusion_peeling_kernel

  attributes(global) subroutine diffusion_register_blocking_kernel(f,nx,ny,nz,sw,ce,cw,cn,cs,ct,cb,cc)

    integer,value :: nx,ny,nz,sw
    real,dimension(nx,ny,nz,2) :: f
    real,value :: ce,cw,cn,cs,ct,cb,cc
    integer :: x, y, z, c, w, e, n, s, b, t
    integer :: tx, ty
    real :: f_t, f_c, f_b

    x = blockdim%x * (blockidx%x-1) + threadidx%x
    y = blockdim%y * (blockidx%y-1) + threadidx%y

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
    
  end subroutine diffusion_register_blocking_kernel

#endif !CUDA

#ifdef BASE
  subroutine diffusion_baseline(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
#ifdef CUDA
    real,device, dimension(1:nx,1:ny,1:nz,1:2) :: f
    type(dim3) :: dimGrid, dimBlock
#else
    real, dimension(1:nx,1:ny,1:nz,1:2) :: f
#endif
    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
#ifdef CUDA
       dimGrid  = dim3((nx-1)/64 + 1, (ny-1)/4+ 1, 1)
       dimBlock = dim3(64, 4, 1)
       call diffusion_baseline_kernel<<<dimGrid, dimBlock>>>(f,nx,ny,nz,sw,ce,cw,cn,cs,ct,cb,cc)
#else
!$acc kernels present(f)
#ifdef PGI
!$acc loop independent
#elif CAPS
!$acc loop 
#elif CRAY
!$acc loop independent
#endif
       do z = 1,nz
#ifdef PGI
!$acc loop independent
#elif CAPS
!$acc loop 
#elif CRAY
!$acc loop independent
#endif
          do y = 1,ny
#ifdef PGI
!$acc loop independent
#elif CAPS
!$acc loop 
#elif CRAY
!$acc loop independent
#endif
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
#endif !CUDA
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_baseline
#endif !BASE


#ifdef THREAD
  subroutine diffusion_thread(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
#ifdef CUDA
    real,device, dimension(1:nx,1:ny,1:nz,1:2) :: f
    type(dim3) :: dimGrid, dimBlock
#else
    real, dimension(1:nx,1:ny,1:nz,1:2) :: f
#endif
    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
#ifdef CUDA
       dimGrid  = dim3((nx-1)/THREAD_X + 1, (ny-1)/THREAD_Y+ 1, 1)
       dimBlock = dim3(THREAD_X, THREAD_Y, 1)
       call diffusion_baseline_kernel<<<dimGrid, dimBlock>>>(f,nx,ny,nz,sw,ce,cw,cn,cs,ct,cb,cc)
#else
!$acc kernels present(f)
#ifdef PGI
!$acc loop seq
#elif CAPS
!$acc loop 
#elif CRAY
!$acc loop independent gang
#endif
       do z = 1,nz
#ifdef PGI
!$acc loop independent gang vector(THREAD_Y)
#elif CAPS
!$acc loop gang(ny)
#elif CRAY
!$acc loop independent
#endif
          do y = 1,ny
#ifdef PGI
!$acc loop independent gang vector(THREAD_X)
#elif CAPS
!$acc loop worker(THREAD_X)
#elif CRAY
!$acc loop independent vector(THREAD_X) 
#endif
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
#endif !CUDA
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_thread
#endif !THREAD

#ifndef CUDA
#ifdef INTERCHANGE
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
#ifdef PGI
!$acc loop independent gang vector(THREAD_Y)
#elif CAPS
!$acc loop gang(ny)
#elif CRAY
!$acc loop independent gang
#endif
       do y = 1,ny
#ifdef PGI
!$acc loop independent gang vector(THREAD_X)
#elif CAPS
!$acc loop worker(THREAD_X)
#elif CRAY
!$acc loop independent vector(THREAD_X)
#endif
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
#endif !INTERCHANGE
#endif

#ifdef PEELING
  subroutine diffusion_peeling(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
#ifdef CUDA
    real,device, dimension(1:nx,1:ny,1:nz,1:2) :: f
    type(dim3) :: dimGrid, dimBlock
#else
    real, dimension(1:nx,1:ny,1:nz,1:2) :: f
#endif
    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
#ifdef CUDA
       dimGrid  = dim3((nx-1)/THREAD_X + 1, (ny-1)/THREAD_Y+ 1, 1)
       dimBlock = dim3(THREAD_X, THREAD_Y, 1)
       call diffusion_peeling_kernel<<<dimGrid, dimBlock>>>(f,nx,ny,nz,sw,ce,cw,cn,cs,ct,cb,cc)
#else
!$acc kernels present(f)
#ifdef PGI
!$acc loop independent gang vector(THREAD_Y)
#elif CAPS
!$acc loop gang(ny)
#elif CRAY
!$acc loop independent gang
#endif
       do y = 1,ny
#ifdef PGI
!$acc loop independent gang vector(THREAD_X)
#elif CAPS
!$acc loop worker(THREAD_X)
#elif CRAY
!$acc loop independent vector(THREAD_X)
#endif
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
#endif !CUDA
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_peeling
#endif !PEELING


#ifdef REGISTER
  subroutine diffusion_register_blocking(f,nx,ny,nz,ce,cw,cn,cs,ct,cb,cc,dt,time,count)
    integer, intent(in) :: nx,ny,nz
    real, intent(in) :: ce,cw,cn,cs,ct,cb,cc,dt
    real, intent(out) :: time
    integer, intent(out) :: count
#ifdef CUDA
    real,device, dimension(1:nx,1:ny,1:nz,1:2) :: f
    type(dim3) :: dimGrid, dimBlock
#else
    real, dimension(1:nx,1:ny,1:nz,1:2) :: f
#endif
    integer :: sw
    integer :: x, y, z, c, w, e, n, s, b, t
    real :: f_t, f_c, f_b
    time = 0.0
    count = 0

    do while(time + 0.5*dt < 0.1)
       sw = mod(count,2) 
#ifdef CUDA
       dimGrid  = dim3((nx-1)/THREAD_X + 1, (ny-1)/THREAD_Y+ 1, 1)
       dimBlock = dim3(THREAD_X, THREAD_Y, 1)
       call diffusion_register_blocking_kernel<<<dimGrid, dimBlock>>>(f,nx,ny,nz,sw,ce,cw,cn,cs,ct,cb,cc)
#else
!$acc kernels present(f)
#ifdef PGI
!$acc loop independent gang vector(THREAD_Y)
#elif CAPS
!$acc loop gang(ny)
#elif CRAY
!$acc loop independent gang
#endif
       do y = 1,ny
#ifdef PGI
!$acc loop independent gang vector(THREAD_X)
#elif CAPS
!$acc loop worker(THREAD_X)
#elif CRAY
!$acc loop independent vector(THREAD_X)
#endif
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
#endif !CUDA
       time = time + dt
       count = count + 1
    end do

  end subroutine diffusion_register_blocking
#endif !REGISTER


end module stencil_mod


program stencil

  use stencil_mod
  implicit none

  integer :: NX, NY, NZ
  NX = DATASIZE
  NY = DATASIZE
  NZ = DATASIZE

  call main(NX, NY, NZ)

end program stencil

