"""
Script and functions to calculate the parameters from Ramsey fringes to do trap equalizaition.
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, Minimizer

from basic_analysis import extract_probabilities


DIR_TRAP_EQUALIZATION = 'C:\\SLM'


def fit_ramsey(user_function, x, y, f0 = 0.0, tau = 0.2, off = 0.5,amp = 0.3):
    '''
    Define the fit parameters (ex: contrast, offset, frequency ...)
    
    Parameters can take optional arguments:
        - value: initial guess for the fit routine
        - min, max: bounds for the parameter
    '''
    params = Parameters()
    params.add('contrast', value = amp, min = 0.15, max = 0.75, vary = True)
    params.add('offset', value = off, vary = False)
    params.add('f0', value =  f0, min = f0-1/tau/2, max = f0+1/tau/2)
    params.add('tau', value = tau, vary = False)

    mini = Minimizer(
        userfcn = user_function, #fitting function
        params = params,
        fcn_args = (x,), #positional arguments for the fitting function
        fcn_kws = {'data':y,}, #keywords arguments for the fitting function
        nan_policy='omit'
        )
        
    #FIT 
    result = mini.minimize(method='leastsq')
    
    return result
    
def func_ramsey(params, x, data = None):
    '''
    Define the fit function (model). 
    '''
    parvals = params.valuesdict()        
    contrast = parvals['contrast']
    offset = parvals['offset']
    f0 = parvals['f0']
    tau = parvals['tau']

    model = offset + contrast*np.cos(2*np.pi*tau*(x-f0))
    
    #return the model 
    if data is None :
        return model
    
    #return the residuals 
    return model-data 


def cal(filename, loop = None):
    
    folder_name = filename
    
    BY_ATOM = True
    
    # row = 20
    # col = 7
    
    
    RAMSEY_WAIT = 3.5 #millisecond
    
    f0_notrap = 0#8.75
    
    F0 = -3.232 #DLS value
        
    f0_fit_list = []
    f0_fit_list_err = []    
    # sigma_fit_list = []
    
    threshold = np.loadtxt(folder_name+'//threshold.dat')#150
    N_atoms = len(np.loadtxt(folder_name+'//rois_list.dat'))
    extract_probabilities(folder_name,N_atoms,threshold, round_decimal=4)
    
    param_list = np.load(folder_name+'//param_list.npy')
    recap_per_atom = np.load(folder_name+'//recap_per_atom.npy')
    # fluo_per_atom = np.load(folder_name+'//fluo_per_atom.npy')
    loaded_per_atom = np.load(folder_name+'//loaded_per_atom.npy')
     
    N_list = np.load(folder_name+'//N_per_atom.npy')
    
    
    #Average over atom
    N_tot = np.nansum(N_list,axis = 1)
    p_avg_recap = np.nansum(recap_per_atom*N_list,axis = 1)/N_tot
    # p_loaded= np.nanmean(loaded_per_atom,axis = 1)
    err = np.sqrt(p_avg_recap*(1-p_avg_recap)/N_tot)
    
    ramsey_param = param_list[:,1]
    ramsey_param = np.unique(ramsey_param)
    ramsey_param_size = np.size(ramsey_param)
    
    choose_exp_start = loop*ramsey_param_size
    choose_exp_end = int(int(loop + 1)*ramsey_param_size)
    
    print('choose exp', choose_exp_start)
    
    p_avg_recap = p_avg_recap[choose_exp_start:choose_exp_end]
    recap_per_atom = recap_per_atom[choose_exp_start:choose_exp_end, :]
    
    N_list = N_list[choose_exp_start:choose_exp_end, :]
    
    
    x = - ramsey_param - f0_notrap#-f_center
    y = p_avg_recap
    
    
    # x_list = np.linspace(x.min(),x.max(),100)
    # max_f = x[np.argmax(y)]
    amp= (np.max(y)-np.min(y))/2
    amp = 0.26
    off = np.mean(y)
    off = 0.45
    
    result_fit = fit_ramsey(func_ramsey,x,y, f0 = 0., tau = RAMSEY_WAIT, off = off, amp = amp)
    # x0 = result_fit.params['f0'].value
    # xerr = result_fit.params['f0'].stderr
    
    if BY_ATOM :
        # T = 0
        
        for atom in range(N_atoms):
            # try:
            # select_x = x ==x 
            
            x_s = x
            p = recap_per_atom[:,atom]
    
            # N = N_list[:,atom]
            # err = np.sqrt(p*(1-p)/N)
            
            amp= (np.nanmax(p)-np.nanmin(p))/2
            off = np.nanmean(p)
            
            amp = 0.26
            off = 0.45
    
            # try:
            result_fit = fit_ramsey(func_ramsey,x_s,p, f0 = F0, tau = RAMSEY_WAIT, off = off, amp = amp)
            f0_fit_list.append(result_fit.params['f0'].value)
            
            stderr = result_fit.params['f0'].stderr
            if stderr:
                f0_fit_list_err.append(stderr)
            else:
                f0_fit_list_err.append(np.nan)
    
    
    f0_fit_list =  np.array(f0_fit_list)
    rel_intensity = f0_fit_list/np.mean(f0_fit_list)
    rel_intensity_err = np.nanmean(f0_fit_list_err)/np.abs(np.mean(f0_fit_list))
    
    
    print(f'Std: {np.std(rel_intensity)*100:.2f} % ({rel_intensity_err*100:.2f})')
    print('Mean freq: %.3f'%np.mean(f0_fit_list))
    # np.save('C:\\SLM\\map.npy',rel_intensity)
    
    return rel_intensity