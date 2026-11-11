"""
CUDA installation : Cuda v10.2
Visual Studio: v2019
NVIDIA driver: https://www.nvidia.com/download/index.aspx?lang=en-us
    
I did not succeed in having pycuda working...
Instead I installed pyopencl
I followed the procedure here: https://wiki.tiker.net/PyOpenCL/Installation/Windows
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
"""

import numpy as np

import pyopencl as cl
# Check the driver of the Intel GPU. --> Device Manager --> Display adapters --> Intel HD Graphics 530
#I could not use the GPU when the driver updated (automatically) to 9/5/2020. Rolling back to 6/5/2020 solved the issue. 
#I keep an installation file fordis the drivers in SLM_python/drivers_that_work_for_GPU...

USE_GPU_INDEX = 0
PLATFORMS = cl.get_platforms()
DEVICE = [PLATFORMS[USE_GPU_INDEX].get_devices()[0]]
print('Using GPU:', DEVICE)

# Build the hologram kernel once per device. Re-building cl.Program on every
# trap_phase_to_slm call can trigger OUT_OF_HOST_MEMORY from the NVIDIA compiler
# and is unnecessarily slow during WGS iterations.
_HOLOGRAM_CL = None  # (ctx, queue, prg, device_key)


def _device_key(dev_list):
    return tuple(getattr(d, "int_ptr", id(d)) for d in dev_list)


def _get_hologram_cl_resources(device):
    global _HOLOGRAM_CL
    key = _device_key(device)
    if _HOLOGRAM_CL is None or _HOLOGRAM_CL[3] != key:
        ctx = cl.Context(device)
        queue = cl.CommandQueue(ctx)
        # No #include <pyopencl-complex.h>: that header makes NVidia's clBuildProgram
        # use enough host RAM to hit OUT_OF_HOST_MEMORY on some Windows setups.
        try:
            prg = cl.Program(ctx, calculate_hologram_string).build()
        except Exception:
            prg = cl.Program(ctx, calculate_hologram_string).build(
                options=["-cl-opt-disable"],
            )
        _HOLOGRAM_CL = (ctx, queue, prg, key)
    return _HOLOGRAM_CL[0], _HOLOGRAM_CL[1], _HOLOGRAM_CL[2]


# Complex field as float2 (real, imag); matches numpy.complex64 / cfloat_t memory layout.
calculate_hologram_string = """
    __kernel void calculate_hologram(
        __global const float *k_x,
        __global const float *k_y,
        __global const float *k_z,
        __global const float *trap_phase,
        __global float2 *slm_phase,
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

             float w = trap_weight[n] * exp(-ellipse_equation) / pupil_norms[n];
             float ph = grating + trap_phase[n];
             float2 contrib = (float2)(w * cos(ph), w * sin(ph));
             slm_phase[k] = slm_phase[k] + contrib;
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
    
    ctx, queue, prg = _get_hologram_cl_resources(device)
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
    
    # Perform the calculation on gpu (program built once — see _get_hologram_cl_resources)
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

def slm_to_trap_amp(phi, k_list, device=DEVICE):
    '''
    Field amplitude at each trap (sum over SLM pixels):

        amplitude[trap] = sum_pixel exp( i * (phi[pixel] - grating[trap](pixel)) )

    OpenCL ReductionKernel + pyopencl-complex.h was removed: it caused
    clBuildProgram OUT_OF_HOST_MEMORY on some NVIDIA Windows setups.
    '''
    Y_max, X_max = phi.shape
    phi_flat = np.ascontiguousarray(phi, dtype=np.float64).ravel()
    n = phi_flat.size
    lin = np.arange(n, dtype=np.float64)
    i_h = lin % X_max
    j_v = lin // X_max
    fresnel = (i_h - X_max / 2.0) ** 2 + (j_v - Y_max / 2.0) ** 2

    trap_amps = np.empty(len(k_list), dtype=np.complex64)
    for t, kvec in enumerate(k_list):
        k_x, k_y, k_z = float(kvec[0]), float(kvec[1]), float(kvec[2])
        grating = k_x * i_h + k_y * j_v + k_z * fresnel
        trap_amps[t] = np.sum(np.exp(1j * (phi_flat - grating)))
    return trap_amps
