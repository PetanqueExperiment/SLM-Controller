# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 19:41:13 2017

@author: Sylvain

CUDA installation : Cuda v10.2
Visual Studio: v2019
NVIDIA driver: https://www.nvidia.com/download/index.aspx?lang=en-us
    
I did not succeed in having pycuda working...
Instead I installed pyopencl
I followed the procedure here: https://wiki.tiker.net/PyOpenCL/Installation/Windows
"""

'''
Version:
    
1- Made in France with pycuda

2- Made in Japan with pyopencl

3- Scalability is improved drastically
    In previous versions, I was sending N_trap grating-like holograms
    to the GPU. This was taking a lot of time. I now calculate these simple gratings on the GPU using the formula

    grating(i,j) = 2pi/lambda * (M/f)  (x_slm * X_trap + y_slm * Y_trap)
                            = (i * k_x + j * k_y)      (*)
    
    The gpu_module receives the grating normal vector [k_x,k_y] for each trap. They are sent to the GPU to calculate the grating 
    equation (*)
    
    --- Physical insights ---
    
    x_slm = i * pixel_size, y_slm = j * pixel_size
    k_x = 2pi/lambda * (M/f) * pixel_size * X_trap
    k_y = 2pi/lambda * (M/f) * pixel_size * Y_trap
         
    f/M is the magnified focal length of the microscope objective, taking into account the relay telescopes
    of magnification M
    
    theta_x = X_trap * (M/f) is the angle of the wavefront making a trap at position X_trap
    
    theta_x * pixel_size is the wavefront advance from one pixel to the next
    
    2pi/lambda * theta_x * pixel_size is the wavefront advance in unit of 2pi from one pixel to the next
    

    --- GPU calculation ---
    
    **Index of 2d Python array and 1d GPU array**
        The index (i,j) corresponds to the horizontal and vertical pixel position of the SLM. 
        The python SLM array encodes the pixel values as phase_slm[j,i] (first dimension is vertical, second is horizontal)
        The GPU processes flattened arrays:
            
            phase_slm[j,i] = phase_slm_gpu[k]  
            
            with k = j * N_i + i  (N_i is the number of pixels along the horizontal)
            
        The index i and j are retrieved using: 
            
            j = k/N_i
            i = k%N_i        
            
    **Working with complex number on the GPU**
    I use the pyopencl-complex library to add and multiply complex numbers

4- Add pupil for each trap

7 - Modif by Martin, add z degree of freedom on trap position

8 - Tom: removed the global pupil to avoid bugs with the global grating (ghost tweezer) and changed init of phase_slm (ones -> zeros)

-> Petanque Fork on 2026/02/03 (v2_0 and onwards)

'''

import time
import numpy as np

import pyopencl as cl
import pyopencl.array
import pyopencl.reduction
# Check the driver of the Intel GPU. --> Device Manager --> Display adapters --> Intel HD Graphics 530
# I could not use the GPU when the driver updated (automatically) to 9/5/2020. Rolling back to 6/5/2020 solved the issue. 
# I keep an installation file fordis the drivers in SLM_python/drivers_that_work_for_GPU...

USE_GPU_INDEX = 0
PLATFORMS = cl.get_platforms()
DEVICE = [PLATFORMS[USE_GPU_INDEX].get_devices()[0]]
print('Using GPU:', DEVICE)

calculate_hologram_string = """        
    #include <pyopencl-complex.h>
        
    __kernel void calculate_hologram(
        __global const float *k_x,
        __global const float *k_y,
        __global const float *k_z,
        __global const float *trap_phase,
        __global cfloat_t *slm_phase,
        __global const float *trap_weight,
        __global const float *pupil_longaxis,
        __global const float *pupil_shortaxis,
        __global const float *pupil_angle,
        __global const float *pupil_norms,
        int n_trap,
        int X_max,
        int Y_max

        )
    {
     int k = get_global_id(0);
     int j = k/X_max;
     int i = k%X_max;

             
     for (int n = 0 ; n < n_trap ; n++) {
              
             
             float grating = k_x[n] * i + k_y[n] * j + k_z[n]* ( (i-X_max/2) * (i-X_max/2) + (j-Y_max/2) * (j-Y_max/2) );                                                                  

             float x_rot = cos(pupil_angle[n]) * (i-X_max/2) + sin(pupil_angle[n]) * (j-Y_max/2);
             float y_rot = sin(pupil_angle[n]) * (i-X_max/2) - cos(pupil_angle[n]) * (j-Y_max/2);
             float ellipse_equation = pow(x_rot/pupil_longaxis[n],2) + pow(y_rot/pupil_shortaxis[n],2);
             
             slm_phase[get_global_id(0)] = cfloat_add(
                      slm_phase[get_global_id(0)],
                          cfloat_mulr(
                              cfloat_exp(cfloat_new(0,grating+trap_phase[n])),
                              trap_weight[n] * exp(-ellipse_equation)/pupil_norms[n]
                                )
                          );
             
              }
          
    }
    """
    
def trap_phase_to_slm(k_list, theta, weight = None, pupil = None, shape = (600,792), device = DEVICE):
    '''
    Perform in the GPU : 
        
        phase_slm[pixel] = sum( exp(delta_trap[pixel][trap] + trap_phase[trap]) )
        
    where the sum is performed for trap in range(N_traps)

    This is an element-wise operation. To calculate the phase_slm on a given pixel,
    we need the value of delta_trap on this pixel and we sum for all traps.
    '''    
    
    Y_max, X_max = shape
    X_max = np.int32(X_max)
    Y_max = np.int32(Y_max)
    
    N_traps = len(theta)
    N_traps = np.int32(N_traps)
    N_pixel = np.int32(Y_max * X_max)

    # Ensure the right type for the numbers
    k_x = k_list[:,0].astype(np.float32)
    k_y = k_list[:,1].astype(np.float32)
    k_z = k_list[:,2].astype(np.float32)

    theta = theta.astype(np.float32)
    phase_slm = np.zeros((N_pixel), dtype=np.complex64)
    
    if weight is None:
        weight = np.ones(N_traps).astype(np.float32)
    else:
        weight = weight.astype(np.float32)
    
    if pupil is None:
        pupil_longaxis = np.ones(N_traps).astype(np.float32) * X_max
        pupil_shortaxis = np.ones(N_traps).astype(np.float32) * X_max
        pupil_angle = np.zeros(N_traps).astype(np.float32)
    else:
        pupil_longaxis = pupil['pupil_longaxis'].astype(np.float32)
        pupil_shortaxis = pupil['pupil_shortaxis'].astype(np.float32)
        pupil_angle = pupil['pupil_angle'].astype(np.float32)
    
    #Pupil normalization
    beam_radius_pix = 300 #3.75 mm beam radius -->  300 pixels
    r = np.arange(-beam_radius_pix,beam_radius_pix).reshape((1,-1))
    L = pupil_longaxis.reshape((-1,1))
    S = pupil_shortaxis.reshape((-1,1))
    long_norms = np.sum( np.exp(-r**2/L**2)*np.exp(-r**2/beam_radius_pix**2), axis = 1) #assume uniform beam for now
    short_norms = np.sum( np.exp(-r**2/S**2)*np.exp(-r**2/beam_radius_pix**2), axis = 1) #assume uniform beam for now
    pupil_norms = long_norms * short_norms 
    pupil_norms /= np.mean(pupil_norms) 
    pupil_norms =  pupil_norms.astype(np.float32)
    
    # Create context for GPU access
    ctx = cl.Context(device)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    
    # Allocate memory in the gpu for the arrays
    k_x_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_x) 
    k_y_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_y)
    k_z_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_z) 
    
    theta_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta)
    phase_slm_gpu = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf = phase_slm)
    weight_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight)
    pupil_longaxis_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pupil_longaxis)
    pupil_shortaxis_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pupil_shortaxis)
    pupil_angle_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pupil_angle)
    pupil_norms_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pupil_norms)
    
    # Perform the calculation on gpu
    prg = cl.Program(ctx, calculate_hologram_string).build()
    prg.calculate_hologram(
            queue,
            phase_slm.shape,
            None,
            k_x_gpu,
            k_y_gpu,
            k_z_gpu,
            theta_gpu,
            phase_slm_gpu,
            weight_gpu,
            pupil_longaxis_gpu,
            pupil_shortaxis_gpu,
            pupil_angle_gpu,
            pupil_norms_gpu,
            N_traps, 
            X_max,
            Y_max,
            )

    # Get the array back from gpu to cpu   
    phase_slm_result = np.empty_like(phase_slm)    
    cl.enqueue_copy(queue, phase_slm_result, phase_slm_gpu)
    
    # Get the angle of the complex number at each pixel
    phi = np.angle(phase_slm_result)
    phi = np.reshape(phi, shape)
    return phi

def slm_to_trap_amp(phi, k_list, device = DEVICE):
    '''
    In contrast to the other function trap_phase_to_slm we use gpuarray module of pycuda
    to transfer the arrays to the gpu
    
    We also use the pycuda ReductionKernel and not the SourceModule to define our CUDA function
    pyCuda developpers implemented a solution proposed by M. Harris on his tutorial 
    "Optimizing Parallel Reduction in CUDA" (Google it)
    You can have a look at the source code of ReductionKernel 
    
    
    The krnl function is called to get the field amplitude at each trap location
        amplitude = sum_over_all_pixel ( exp(i (hologram - grating) ) )
    
    -- dtype_out : python type of the array we want at the end
    -- neutral : the reduction (integration here) starts at 0
    -- reduce_expr : we want an integration, so we sum elements : a+b
    -- map_expr : before integrating the matrix, we perform an element wise operation
    -- arguments : input arguments. 
            pycuda::complex<float> is a type defined in pycuda-complex.hpp by pycuda developers
                for complex number (the equivalent type in python is np.complex64, NOT complex128!)
    -- preamble : we include the definition of the complex number
    '''
    #Get the number of traps
    Y_max, X_max = phi.shape
    N_pixel = np.int32(Y_max*X_max)
   
    phi_local = np.copy(phi)
    
    phi_flatten = 1j*np.reshape(phi_local, (-1))

    #Create context for GPU access
    ctx = cl.Context(device)
    queue = cl.CommandQueue(ctx)
    
    phi_gpu = cl.array.Array(queue, N_pixel, dtype=np.complex64)
    phi_gpu.set(phi_flatten)
    
    map_expr = '''
        cfloat_exp(
                cfloat_sub(x[i],
                    cfloat_add(
                        cfloat_add(
                            cfloat_mulr(k_x,i%X_max),
                            cfloat_mulr(k_y,i/X_max)
                            ),
                        cfloat_mulr(k_z,(i%X_max - X_max/2)*(i%X_max-X_max/2)+ (i/X_max-Y_max/2)*(i/X_max-Y_max/2))
                        )
                    )
                )
        '''

    krnl = cl.reduction.ReductionKernel(
        ctx, 
        dtype_out = np.complex64, 
        neutral="cfloat_fromreal(0)",
        reduce_expr="cfloat_add(a,b)",
        map_expr=map_expr,
        arguments="cfloat_t *x, cfloat_t k_x, cfloat_t k_y, cfloat_t k_z, int X_max, int Y_max",
        preamble = "#include <pyopencl-complex.h>"
        )
    
    #Calculate each trap amplitude
    trap_amps = []
    for k in k_list:
        k_x, k_y, k_z = k
        
        my_sum = krnl(phi_gpu, 1j*k_x, 1j*k_y, 1j*k_z, X_max, Y_max).get()
        
        # my_sum = krnl(phi_gpu, 1j*k_x, 1j*k_y, X_max).get()
        
        trap_amps.append(my_sum)
    
    return np.asarray(trap_amps)       
