"""Benchmarks / self-tests of the GPU hologram code (libslm/gpu.py). Not used by the GUI.

Moved unchanged from gpu_module_v7_3.py. Run from the SLM-Controller folder, e.g.
    python -c "from dev.gpu_benchmarks import test_GPU; test_GPU(L=3)"
Known issue: the tests build 2-column (x, y) trap arrays, while trap_phase_to_slm expects 3 columns (x, y, z).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from libslm.gpu import DEVICE, PLATFORMS, slm_to_trap_amp, trap_phase_to_slm  # noqa: E402


# =============================================================================
# Test functions
# =============================================================================
def _test_wavelength_m(wavelength_m=None):
    """Wavelength [m] for the test functions: the argument, else wavelength_nm of the profile the GUI would start with."""
    if wavelength_m is not None:
        return float(wavelength_m)
    from gui.settings import AppSettings, loadProfileParams
    name = AppSettings.load().startupProfile()
    params, _ = loadProfileParams(name)
    wavelength_nm = params["grating"]["wavelength_nm"]
    print(f"Test wavelength from profile {name!r}: {wavelength_nm} nm")
    return wavelength_nm / 1e9


def test_GPU_vs_CPU(L = 5, wavelength_m = None):
    import matplotlib.pyplot as plt
    from gui.parameters import param
    wavelength_m = _test_wavelength_m(wavelength_m)
    
    #==============================================================================
    # Parameters
    #==============================================================================
    Nx = param.ImgResX
    Ny = param.ImgResY
    SLM_shape = (Ny,Nx)
                
    micron_to_step_per_pixel = 2*np.pi/wavelength_m*(param.M/param.f)*param.pslm
    
    #Load traps and convert to micrometer
#    LOCAL_DIR = 'G:\\Shared drives\\3G_Common\\Software\\SLM_python'
#    trap_file = '15x15_5x5um_rot.dat'
#    trap_file = '10x10_5x5um_rot.dat'
#    trap_file = '5x5_10x10um.dat'

#    traps = np.loadtxt(LOCAL_DIR+'\\patterns\\'+trap_file, delimiter = ',')
#    traps[:,1:3] *= 1e-6
#    N_traps = len(traps)

    N_traps = L**2
    spacing = 5e-6
    traps = []
    for i in range(L):
        for j in range(L):
            trap = [spacing*i,spacing*j]
            traps.append(trap)
    traps = np.array(traps)
    
    k_list = traps*micron_to_step_per_pixel
    
    #Get initial random phase and unit weight 
    theta_traps = np.random.rand(N_traps)*2*np.pi
    weight_traps = np.ones(N_traps)
    
    # =============================================================================
    # #CPU calculation
    # =============================================================================
    t0 = time.time()
    
    #Initialize SLM pixel index arrays
    X,Y = np.meshgrid(
        np.arange(Nx),
        np.arange(Ny),
        )
    
    phi_CPU = np.zeros(SLM_shape).astype('complex64')
    for trap in range(N_traps):
        kx = k_list[trap,0]
        ky = k_list[trap,1]
        
        w = weight_traps[trap]
        theta = theta_traps[trap]
        
        grating = X * kx + Y * ky
        
        phi_CPU += w * np.exp(1j * (grating + theta))
    phi_CPU = np.angle(phi_CPU)
    t1 = time.time()
    print('CPU: %.1f ms'%((t1-t0)*1e3))
    plt.matshow(phi_CPU)
    
    
    # =============================================================================
    # #GPU calculation
    # =============================================================================
    t0 = time.time()
    phi_GPU = trap_phase_to_slm(k_list,theta_traps, shape = SLM_shape)
    t1 = time.time()
    T_sum = t1-t0
    print('GPU GS: %.1f ms'%((t1-t0)*1e3))
    plt.matshow(phi_GPU)
    
    
    # =============================================================================
    # #Get traps amplitude with CPU
    # =============================================================================
    t0 = time.time()
    traps_amplitude_CPU = np.zeros(N_traps).astype('complex64')
    for trap in range(N_traps):
        kx = k_list[trap,0]
        ky = k_list[trap,1]
        
        w = weight_traps[trap]
        theta = theta_traps[trap]
        
        grating = X * kx + Y * ky
                
        f = np.exp(1j*(phi_CPU - grating))
        traps_amplitude_CPU[trap] =  N_traps/(Nx*Ny) * np.sum(f)
    t1 = time.time()
    print('FT CPU: %.1f ms'%((t1-t0)*1e3))
#    
#    
    # =============================================================================
    # #Get traps amplitude with GPU
    # =============================================================================
    t0 = time.time()
    traps_amplitude_GPU = N_traps**0.5/(Nx*Ny)*slm_to_trap_amp(phi_GPU, k_list)
    t1 = time.time()
    T_reduce = t1-t0
    print('FT GPU: %.1f ms'%((t1-t0)*1e3))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(np.abs(traps_amplitude_GPU)**2,'o')
    ax.plot(np.abs(traps_amplitude_CPU)**2,'x')
    
    ax.set_ylim(0,2)
    
    ax.set_xlabel('Trap index')
    ax.set_ylabel('Trap intensity')
    plt.show()

#    return T_sum,T_reduce


def test_GPU(L = 5, device = DEVICE, wavelength_m = None):
    import matplotlib.pyplot as plt
    from gui.parameters import param
    wavelength_m = _test_wavelength_m(wavelength_m)
    
    #==============================================================================
    # Parameters
    #==============================================================================
    Nx = param.ImgResX
    Ny = param.ImgResY
    SLM_shape = (Ny,Nx)
                
    micron_to_step_per_pixel = 2*np.pi/wavelength_m*(param.M/param.f)*param.pslm
    
    N_traps = L**2
    spacing = 5e-6
    traps = []
    for i in range(L):
        for j in range(L):
            trap = [spacing*i,spacing*j]
            traps.append(trap)
    traps = np.array(traps)
    
    k_list = traps*micron_to_step_per_pixel
    
    #Get initial random phase and unit weight 
    theta_traps = np.random.rand(N_traps)*2*np.pi
    weight_traps = np.ones(N_traps)
    
    
    # =============================================================================
    # #GPU calculation
    # =============================================================================
    t0 = time.time()
    phi_GPU = trap_phase_to_slm(k_list,theta_traps, shape = SLM_shape, device = device)
    t1 = time.time()
    T_sum = t1-t0
  

    # =============================================================================
    # #Get traps amplitude with GPU
    # =============================================================================
    t0 = time.time()
    traps_amplitude_GPU = N_traps**0.5/(Nx*Ny)*slm_to_trap_amp(phi_GPU, k_list, device = device)
    t1 = time.time()
    T_reduce = t1-t0

    return T_sum,T_reduce



def test_GPU_pupil(L = 5, wavelength_m = None):
    import matplotlib.pyplot as plt
    from gui.parameters import param
    wavelength_m = _test_wavelength_m(wavelength_m)
    
    #==============================================================================
    # Parameters
    #==============================================================================
    Nx = param.ImgResX
    Ny = param.ImgResY
    SLM_shape = (Ny,Nx)
                
    micron_to_step_per_pixel = 2*np.pi/wavelength_m*(param.M/param.f)*param.pslm
    
    N_traps = L**2
    spacing = 5e-6
    traps = []
    for i in range(L):
        for j in range(L):
            trap = [spacing*i+10e-6,spacing*j+10e-6]
            traps.append(trap)
    traps = np.array(traps)
    # traps = np.array([[0,0],[10e-6,10e-6]])
    k_list = traps*micron_to_step_per_pixel
    
    #Get initial random phase and unit weight 
    theta_traps = np.random.rand(N_traps)*2*np.pi
    weight_traps = np.ones(N_traps)
    
    
    pupil = {
        'pupil_longaxis':np.array([150]*N_traps),
        'pupil_shortaxis':np.array([300]*N_traps),
        'pupil_angle':np.array([10*np.pi/180]*N_traps),
              }
    
    # pupil['pupil_longaxis'][0] = 600
    # pupil['pupil_shortaxis'][0] = 600
    
    # pupil = {
    #     'pupil_longaxis':np.array([500,150,10,10]),
    #     'pupil_shortaxis':np.array([500,200,100,100]),
    #     'pupil_angle':np.array([0,0,0,0]),#+10/180*3.14,
    #          }
    # =============================================================================
    # #GPU calculation
    # =============================================================================
    t0 = time.time()
    phi_GPU = trap_phase_to_slm(k_list,theta_traps,pupil = pupil,shape = SLM_shape)
    t1 = time.time()
    T_sum = t1-t0
  

    # =============================================================================
    # #Get traps amplitude with GPU
    # =============================================================================
    t0 = time.time()
    traps_amplitude_GPU = N_traps**0.5/(Nx*Ny)*slm_to_trap_amp(phi_GPU, k_list)
    t1 = time.time()
    T_reduce = t1-t0
    
    plt.imshow(phi_GPU)

    return T_sum,T_reduce
# test_GPU_pupil()

def test_CPU(L = 5, wavelength_m = None):
    
    import matplotlib.pyplot as plt
    from gui.parameters import param
    wavelength_m = _test_wavelength_m(wavelength_m)
    
    #==============================================================================
    # Parameters
    #==============================================================================
    Nx = param.ImgResX
    Ny = param.ImgResY
    SLM_shape = (Ny,Nx)
                
    micron_to_step_per_pixel = 2*np.pi/wavelength_m*(param.M/param.f)*param.pslm
    
  
    N_traps = L**2
    spacing = 5e-6
    traps = []
    for i in range(L):
        for j in range(L):
            trap = [spacing*i,spacing*j]
            traps.append(trap)
    traps = np.array(traps)
    
    k_list = traps*micron_to_step_per_pixel
    
    #Get initial random phase and unit weight 
    theta_traps = np.random.rand(N_traps)*2*np.pi
    weight_traps = np.ones(N_traps)
    
    # =============================================================================
    # #CPU calculation
    # =============================================================================
    t0 = time.time()
    
    #Initialize SLM pixel index arrays
    X,Y = np.meshgrid(
        np.arange(Nx),
        np.arange(Ny),
        )
    
    phi_CPU = np.zeros(SLM_shape).astype('complex64')
    for trap in range(N_traps):
        kx = k_list[trap,0]
        ky = k_list[trap,1]
        
        w = weight_traps[trap]
        theta = theta_traps[trap]
        
        grating = X * kx + Y * ky
        
        phi_CPU += w * np.exp(1j * (grating + theta))
    phi_CPU = np.angle(phi_CPU)
    t1 = time.time()
    T_sum = t1-t0

#    print('CPU: %.1f ms'%((t1-t0)*1e3))
   
    
    # =============================================================================
    # #Get traps amplitude with CPU
    # =============================================================================
    t0 = time.time()
    traps_amplitude_CPU = np.zeros(N_traps).astype('complex64')
    for trap in range(N_traps):
        kx = k_list[trap,0]
        ky = k_list[trap,1]
        
        w = weight_traps[trap]
        theta = theta_traps[trap]
        
        grating = X * kx + Y * ky
                
        f = np.exp(1j*(phi_CPU - grating))
        traps_amplitude_CPU[trap] =  N_traps/(Nx*Ny) * np.sum(f)
    t1 = time.time()
    T_reduce = t1-t0
    
    return T_sum,T_reduce

def scaling():
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
        
#   GPU calculation
    T_sum_list = []
    T_reduce_list = []
    L = np.array([1,3,5,10])
    for l in L:
        T_sum, T_reduce = test_CPU(l)
        T_sum_list.append(T_sum)
        T_reduce_list.append(T_reduce)
    
    
    ax.plot(L**2,T_sum_list,'C0o-', label = 'numpy')    
    ax.plot(L**2,T_reduce_list, 'C0o--')
            
            
#   GPU calculation
     
    DEVICE_LIST = [ 
    [PLATFORMS[0].get_devices()[0]],
    [PLATFORMS[0].get_devices()[1]],
    PLATFORMS[1].get_devices()
    ]

    LABEL_LIST = [
        'Intel(R) HD Graphics 530 on Intel(R) OpenCL',
        'Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz on Intel(R) OpenCL',
        'Quadro P600 on NVIDIA CUDA'
        ]

    for i, DEVICE in enumerate(DEVICE_LIST):
        
        
        T_sum_list = []
        T_reduce_list = []
        if i ==1 :
            L = np.array([1,3,5,10,30])
        else: 
            L = np.array([1,3,5,10,30,100])

        for l in L:
            T_sum, T_reduce = test_GPU(l, device = DEVICE)
            T_sum_list.append(T_sum)
            T_reduce_list.append(T_reduce)
        
        ax.plot(L**2,T_sum_list, 'C%do-'%(i+1),label = LABEL_LIST[i])    
        ax.plot(L**2,T_reduce_list, 'C%do--'%(i+1))

    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Number of traps')
    ax.set_ylabel('Time (s)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
          ncol=1, fancybox=True, shadow=True)
    plt.grid()
    
    plt.savefig('scaling.png')
    plt.show()
    
    
