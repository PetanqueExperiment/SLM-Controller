import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, Minimizer
import re
import os
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def find_experiments(experiment_name,date_list):
        
    list_subfolders_with_paths = [f.path for date in date_list for f in os.scandir(date) if f.is_dir()]
    
    nameRegex = re.compile(experiment_name)
    folder_list = []
    for path in list_subfolders_with_paths:
        mo = nameRegex.search(path)
        if mo:        
            folder_list.append(path)
            
    return folder_list

def extract_probabilities(folder_name,N_atoms = 20,threshold = 130, round_decimal = None):
    
    t0 = time.time()
    
    #How many parameters are being looped on
    with open(folder_name+'\\info.dat','r') as info_file:
        info = [x.strip().split('\t') for x in info_file]
    info_file.close()
    nb_loops = int(info[3][1])
    
    #if binary file exist use it (much faster)
    file_list = os.listdir(folder_name)
    
    if os.path.exists(folder_name+'\\analog_input.bin'):
        with open(folder_name+'\\analog_input.bin','rb') as f:
            analog_input = np.fromfile(f, dtype = np.float64)
            np.save(folder_name+'//analog_input.npy',analog_input)
    
    if 'data.npy' in file_list:
        data = np.load(folder_name+'\\data.npy')
        data = data.reshape((-1,2*N_atoms))
        if nb_loops != 0:
            param_list = np.load(folder_name+'\\params.npy')
            param_list = param_list.reshape((-1,nb_loops))
        else:
            #if no loops, create a fake param_list with the parameter 0 repeated 
            #for the number of experimental run 
            N_run = len(data)
            param_list = np.zeros((N_run,1))
            
    #for analyzing data older than 26th April 2021
    elif 'data.bin' in file_list:
        
        with open(folder_name+'\\data.bin','rb') as f:
            data = np.fromfile(f, dtype = np.uint16)            
            #The data is currently a 1d array
            #Reshape into a 2d array with each rows corresponding to 1 experimental run
            #and containing the 2*N_atoms fluorescence of the first and second image
            data = data.reshape((-1,2*N_atoms))
        

        
        if nb_loops != 0:
            with open(folder_name+'\\params.bin','rb') as f:
                param_list = np.fromfile(f, dtype = np.float64)
                #The list of parameters is currently a 1d array
                #Reshape into a 2d array with each rows corresponding to 1 experimental run
                #and containing the nb_loops parameters that we are looping on 
                param_list = param_list.reshape((-1,nb_loops))
        else:
            #if no loops, create a fake param_list with the parameter 0 repeated 
            #for the number of experimental run 
            N_run = len(data)
            param_list = np.zeros((N_run,1))
       
    #else use text file (for analyzing older data)
    elif 'data.dat' in file_list:
        data = np.loadtxt(folder_name+'//data.dat',delimiter = '\t', usecols = np.arange(1,nb_loops+2*N_atoms+1,1))
        
        param_list = data[:,:nb_loops]
        #round the parameter
        
            
        #remove the parameters from the data
        data = data[:,nb_loops:]
        
    else:
        print('no data file : ( in %s'%folder_name)
    t1 = time.time()
#    print('Loading: %d ms'%((t1-t0)*1e3))
    if round_decimal != None: 
        param_list = np.round(param_list, decimals = round_decimal)
                
    fluo_list = []
    fluo_list_2 = []
    bck_list = []
    loaded_list = []
    recapture_list = []
    filling_list = []
    ghost_list = []
    N_list = []
    N_run_list = []
    std_list = []
    
    if param_list.shape[0] != data.shape[0]:
        print('STRANGE: more parameter than fluo...')
        param_list = param_list[:data.shape[0]]
        print('Ignore the other parameters')
        
    param_list_unique = np.unique(param_list, axis = 0)

    for i, param in enumerate(param_list_unique):
#        try:
            current_run = (param_list == param)
            current_run = np.product(current_run,axis = 1).astype(bool)
            N_run_list.append(np.sum(current_run))
    
            first = data[current_run,::2]
            second = data[current_run,1::2]

            loaded = first>threshold
            recap = second>threshold
                                
            fluo_1 = np.zeros_like(first).astype(float)
            fluo_1[loaded] = first[loaded]
            fluo_1[~loaded] = np.nan                
            
            bckg_1 = np.zeros_like(first).astype(float)
            bckg_1[loaded] = np.nan
            bckg_1[~loaded] = first[~loaded]
            
            fluo_2 = np.zeros_like(first).astype(float)
            fluo_2[recap] = second[recap]
            fluo_2[~recap] = np.nan                
            
            bckg_2 = np.zeros_like(first).astype(float)
            bckg_2[recap] = np.nan
            bckg_2[~recap] = second[~recap]
                 
            recap = recap.astype(float)
            recap[~loaded] = np.nan #if atom is not loaded, set
            
            
            fluo_per_atom = np.nanmean(fluo_1,axis = 0)
            fluo_per_atom_2 = np.nanmean(fluo_2,axis = 0)
            bck_per_atom = np.nanmean(bckg_1,axis = 0)
            N_per_atom = np.nansum(loaded, axis = 0)
            loaded_per_atom = np.nanmean(loaded, axis = 0)
#            filling_per_atom = np.nanmean(filling))
            recapture_per_atom = np.nanmean(recap, axis = 0)
#            ghost_per_atom.append(np.mean(ghost))
            std_per_shot = np.nanstd(np.nanmean(recap,axis = 1),axis = 0)

            fluo_list.append(fluo_per_atom)
            fluo_list_2.append(fluo_per_atom_2)
            bck_list.append(bck_per_atom)
            loaded_list.append(loaded_per_atom)
            recapture_list.append(recapture_per_atom)
            N_list.append(N_per_atom)
            std_list.append(std_per_shot)

    loaded_list = np.asarray(loaded_list)
    recapture_list = np.asarray(recapture_list)     
    N_list = np.asarray(N_list)     
    
    t2 = time.time()
    
    np.save(folder_name+'//param_list.npy',param_list_unique)
    np.save(folder_name+'//bck_per_atom.npy',bck_list)
    np.save(folder_name+'//fluo_per_atom.npy',fluo_list)
    np.save(folder_name+'//fluo_per_atom_2.npy',fluo_list_2)
    np.save(folder_name+'//loaded_per_atom.npy',loaded_list)
    np.save(folder_name+'//recap_per_atom.npy',recapture_list)
    np.save(folder_name+'//N_per_atom.npy',N_list)
    np.save(folder_name+'//N_run.npy',N_run_list)
    np.save(folder_name+'//std_list.npy',std_list)

def extract_probabilities_pair_of_atoms(folder_name,N_atoms = 20,threshold = 130, round_decimal = None, row = 40, col = 20):
    t0 = time.time()
    
    #How many parameters are being looped on
    with open(folder_name+'\\info.dat','r') as info_file:
        info = [x.strip().split('\t') for x in info_file]
    info_file.close()
    nb_loops = int(info[3][1])
    
    # Deal with NI analog input board
    if os.path.exists(folder_name+'\\analog_input.bin'):
        with open(folder_name+'\\analog_input.bin','rb') as f:
            analog_input = np.fromfile(f, dtype = np.float64)
            np.save(folder_name+'//analog_input.npy',analog_input)
    
    #if binary file exist use it (much faster)
    file_list = os.listdir(folder_name)
    
    if 'data.npy' in file_list:
        data = np.load(folder_name+'\\data.npy')
        data = data.reshape((-1,2*N_atoms))
        if nb_loops != 0:
            param_list = np.load(folder_name+'\\params.npy')
            param_list = param_list.reshape((-1,nb_loops))
        else:
            #if no loops, create a fake param_list with the parameter 0 repeated 
            #for the number of experimental run 
            N_run = len(data)
            param_list = np.zeros((N_run,1))
    
    #for analyzing data older than 26th April 2021
    elif 'data.bin' in file_list:
        
        with open(folder_name+'\\data.bin','rb') as f:
            data = np.fromfile(f, dtype = np.uint16)            
            #The data is currently a 1d array
            #Reshape into a 2d array with each rows corresponding to 1 experimental run
            #and containing the 2*N_atoms fluorescence of the first and second image
            data = data.reshape((-1,2*N_atoms))
            
        
        if nb_loops != 0:
            with open(folder_name+'\\params.bin','rb') as f:
                param_list = np.fromfile(f, dtype = np.float64)
                #The list of parameters is currently a 1d array
                #Reshape into a 2d array with each rows corresponding to 1 experimental run
                #and containing the nb_loops parameters that we are looping on 
                param_list = param_list.reshape((-1,nb_loops))
        else:
            #if no loops, create a fake param_list with the parameter 0 repeated 
            #for the number of experimental run 
            N_run = len(data)
            param_list = np.zeros((N_run,1))
       
    #else use text file (for analyzing older data)
    elif 'data.dat' in file_list:
        data = np.loadtxt(folder_name+'//data.dat',delimiter = '\t', usecols = np.arange(1,nb_loops+2*N_atoms+1,1))
        
        param_list = data[:,:nb_loops]
        #round the parameter
        if round_decimal != None: 
            param_list = np.round(param_list, decimals = round_decimal)
            
        #remove the parameters from the data
        data = data[:,nb_loops:]
        
    else:
        print('no data file : ( in %s'%folder_name)
    t1 = time.time()
#    print('Loading: %d ms'%((t1-t0)*1e3))

    fluo_list = []
    fluo_list_2 = []
    bck_list = []
    loaded_list = []
    recapture_list = []
    filling_list = []
    ghost_list = []
    N_list = []
    N_run_list = []
    
    #For pair of atoms
    loaded_single_per_pair_list = []
    loaded_double_per_pair_list = [] 
    recapture_single_per_pair_list = []
    recapture_double_per_pair_list = [] 
    parity_single_per_pair_list = []
    parity_double_per_pair_list = []
    std_list = []
    p00_list = []
    p11_list = []
    p01_list = []
    p10_list = []
    
    if param_list.shape[0] != data.shape[0]:
        print('STRANGE: more parameter than fluo...')
        param_list = param_list[:data.shape[0]]
        print('Ignore the other parameters')
        
    param_list_unique = np.unique(param_list, axis = 0)

    for i, param in enumerate(param_list_unique):
#        try:
            current_run = (param_list == param)
            current_run = np.product(current_run,axis = 1).astype(bool)
            N_run_list.append(np.sum(current_run))
    
            first = data[current_run,::2]
            second = data[current_run,1::2]

            loaded = first>threshold
            recap = second>threshold
                                
            fluo_1 = np.zeros_like(first).astype(float)
            fluo_1[loaded] = first[loaded]
            fluo_1[~loaded] = np.nan                
            
            bckg_1 = np.zeros_like(first).astype(float)
            bckg_1[loaded] = np.nan
            bckg_1[~loaded] = first[~loaded]
            
            fluo_2 = np.zeros_like(first).astype(float)
            fluo_2[recap] = second[recap]
            fluo_2[~recap] = np.nan                
            
            bckg_2 = np.zeros_like(first).astype(float)
            bckg_2[recap] = np.nan
            bckg_2[~recap] = second[~recap]
                 
            recap = recap.astype(float)
            recap[~loaded] = np.nan #if atom is not loaded, set
            
            fluo_per_atom = np.nanmean(fluo_1,axis = 0)
            fluo_per_atom_2 = np.nanmean(fluo_2,axis = 0)
            bck_per_atom = np.nanmean(bckg_1,axis = 0)
            N_per_atom = np.nansum(loaded, axis = 0)
            loaded_per_atom = np.nanmean(loaded, axis = 0)
            recapture_per_atom = np.nanmean(recap, axis = 0)
            

            fluo_list.append(fluo_per_atom)
            fluo_list_2.append(fluo_per_atom_2)
            bck_list.append(bck_per_atom)
            loaded_list.append(loaded_per_atom)
            recapture_list.append(recapture_per_atom)
            N_list.append(N_per_atom)
            
            # =============================================================================
            # Pair the atoms
            # =============================================================================
            '''
            We make pairs of atom aligned vertically. So two atoms in the same pair have the same column index
            but a row index 2i and 2i+1
            
            So the flatten index of the pair is index = column + row*N_column and column + (row+1)*N_column
            '''
            
            # loaded_2d = loaded.reshape((-1,row,col))
            recap = second>threshold
            # recap_2d = recap.reshape((-1,row,col))
            
            loaded_paired = np.array([loaded[:,::2],loaded[:,1::2]])
            recap_paired = np.array([recap[:,::2],recap[:,1::2]])
            
            loaded_paired = np.nanmean(loaded_paired, axis = 0)
            parity_paired = (2*recap[:,::2]-1)*(2*recap[:,1::2]-1)
            recap_paired = np.nanmean(recap_paired, axis = 0)

            loaded_single = loaded_paired == 0.5
            loaded_double = loaded_paired == 1.
            
            #chew
            # print(recap.shape)
            # print(loaded_double.shape)
            row1 = recap[:,::2]
            row2 = recap[:,1::2]
            # row1[~loaded_double] = np.nan
            # row2[~loaded_double] = np.nan
            p00 = row1*row2
            p00 = p00.astype('float64')
            p00[~loaded_double] = np.nan
            p00 = np.nanmean(p00,axis=0)
            
            p11 = np.invert(row1) * np.invert(row2)
            p11 = p11.astype('float64')
            p11[~loaded_double] = np.nan
            p11 = np.nanmean(p11,axis=0)
            
            p01 = row1 * np.invert(row2)
            p01 = p01.astype('float64')
            p01[~loaded_double] = np.nan
            p01 = np.nanmean(p01,axis=0)
            
            p10 = np.invert(row1) * row2
            p10 = p10.astype('float64')
            p10[~loaded_double] = np.nan
            p10 = np.nanmean(p10,axis=0)
            
            p11_list.append(p11)
            p01_list.append(p01)
            p10_list.append(p10)
            p00_list.append(p00)
            
            recap_single = np.zeros_like(recap_paired).astype('float')
            recap_double = np.zeros_like(recap_paired).astype('float')
            parity_single = np.zeros_like(recap_paired).astype('float')
            parity_double = np.zeros_like(recap_paired).astype('float')
            
            
            recap_single[loaded_single] = recap_paired[loaded_single]*2
            recap_single[recap_single == 2] = np.nan
            recap_single[~loaded_single] = np.nan
            
            parity_single[loaded_single] = parity_paired[loaded_single]
            parity_single[~loaded_single] = np.nan
            
            recap_double[loaded_double] = recap_paired[loaded_double]
            recap_double[~loaded_double] = np.nan
            
            parity_double[loaded_double] = parity_paired[loaded_double]
            parity_double[~loaded_double] = np.nan
            
            N_single_per_pair = np.nansum(loaded_single, axis = 0)
            N_double_per_pair = np.nansum(loaded_double, axis = 0)
            loaded_single_per_pair = np.nanmean(loaded_single, axis = 0)
            loaded_double_per_pair = np.nanmean(loaded_double, axis = 0)
            recapture_single_per_pair = np.nanmean(recap_single, axis = 0)
            recapture_double_per_pair = np.nanmean(recap_double, axis = 0)
            
            parity_single_per_pair = np.nanmean(parity_single, axis = 0)
            parity_double_per_pair = np.nanmean(parity_double, axis = 0)
            
            loaded_single_per_pair_list.append(loaded_single_per_pair)
            loaded_double_per_pair_list.append(loaded_double_per_pair)            
            recapture_single_per_pair_list.append(recapture_single_per_pair)
            recapture_double_per_pair_list.append(recapture_double_per_pair)
            std_list.append(np.nanstd(np.nanmean(recap_single, axis = 1)))
            parity_single_per_pair_list.append(parity_single_per_pair)
            parity_double_per_pair_list.append(parity_double_per_pair)

    loaded_list = np.asarray(loaded_list)
    recapture_list = np.asarray(recapture_list)     
    N_list = np.asarray(N_list)     
    
    loaded_single_per_pair_list = np.asarray(loaded_single_per_pair_list)
    loaded_double_per_pair_list = np.asarray(loaded_double_per_pair_list)
    recapture_single_per_pair_list = np.asarray(recapture_single_per_pair_list)
    recapture_double_per_pair_list = np.asarray(recapture_double_per_pair_list)
    parity_single_per_pair_list = np.asarray(parity_single_per_pair_list)
    parity_double_per_pair_list = np.asarray(parity_double_per_pair_list)
    
    p11_list = np.asarray(p11_list)
    p10_list = np.asarray(p10_list)
    p01_list = np.asarray(p01_list)
    p00_list = np.asarray(p00_list)
    
    t2 = time.time()
    
    np.save(folder_name+'//param_list.npy',param_list_unique)
    np.save(folder_name+'//bck_per_atom.npy',bck_list)
    np.save(folder_name+'//fluo_per_atom.npy',fluo_list)
    np.save(folder_name+'//fluo_per_atom_2.npy',fluo_list_2)
    np.save(folder_name+'//loaded_per_atom.npy',loaded_list)
    np.save(folder_name+'//recap_per_atom.npy',recapture_list)
    np.save(folder_name+'//N_per_atom.npy',N_list)
    np.save(folder_name+'//N_run.npy',N_run_list)
    np.save(folder_name+'//loaded_single_per_pair_list.npy',loaded_single_per_pair_list)
    np.save(folder_name+'//loaded_double_per_pair_list.npy',loaded_double_per_pair_list)
    np.save(folder_name+'//recapture_single_per_pair_list.npy',recapture_single_per_pair_list)
    np.save(folder_name+'//recapture_double_per_pair_list.npy',recapture_double_per_pair_list)
    np.save(folder_name+'//std_list.npy',std_list)
    np.save(folder_name+'//parity_single_per_pair_list.npy',parity_single_per_pair_list)
    np.save(folder_name+'//parity_double_per_pair_list.npy',parity_double_per_pair_list)
    np.save(folder_name+'//p11_list.npy',p11_list)  #chew
    np.save(folder_name+'//p10_list.npy',p10_list)  #chew
    np.save(folder_name+'//p01_list.npy',p01_list)  #chew
    np.save(folder_name+'//p00_list.npy',p00_list)  #chew
    
    