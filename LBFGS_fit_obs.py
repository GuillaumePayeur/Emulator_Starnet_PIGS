import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os

from train_and_search import *
################################################################################
# predicting stellar labels with the DNN. each spectrum is fit once, and the
# results are used to generate a better mask which is used in a second fit.
if __name__ == '__main__':
    # Name of the emulator to use
    model_name = 'emulator_v6.pth'
    # Observed data
    data_file = 'CSN_data_obs_all.h5'
    # File to save the results to
    outfile = 'test_results/sols_obs_v5_l1.npy'

    # settings
    n_spectra = 10996
    batch_size = 32
    loss = 'l1'
################################################################################
# Function to perform one fit on a batch of spectra
def fit_LBFGS(model,spectra_obs,weights,loss,batch_size):
    batch_size = spectra_obs.shape[0]
    params = torch.tensor(np.zeros((batch_size,1,23)),requires_grad=True,device='cuda:0',dtype=torch.float)
    params_found = False
    orig_params = params.clone()
    losses = []
    tol = 1e-10

    if loss == 'l1':
        loss_fn = torch.nn.L1Loss(reduction='mean')
    if loss == 'l2':
        loss_fn = torch.nn.MSELoss(reduction='mean')

    orig_loss = loss_fn(model(params)*weights, spectra_obs*weights)

    while params_found is False:
        optimizer = torch.optim.LBFGS([params], lr=1, max_iter=100, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        max_epochs = 100
        for epoch in range(max_epochs):
            def closure():
                optimizer.zero_grad()
                fit = model(params)

                loss = loss_fn(fit*weights, spectra_obs*weights)
                loss.backward()
                losses.append(loss)
                return loss
            optimizer.step(closure)
            scheduler.step()

        count_nans = 0
        final_loss = losses[-1]
        if np.isnan(final_loss.item()):
            params.data = orig_params.data
            final_loss = orig_loss
            count_nans+=1
        else:
            params_found=True
        if count_nans>5:
            params.data = orig_params.data
            final_loss = orig_loss
            params_found=True

        if (len(losses)>1 and abs(losses[-1]-losses[-2]) < tol):
            params_found=True

    valid = np.ones(batch_size)
    best_fits = model(params)

    return params, best_fits, np.reshape(valid,(batch_size,1))

# Function to perform the whole fitting procedure on spectra.
def fit_n_LBFGS(n,model,batch_size,loss,outfile):
    sols_array = np.zeros((n,23))

    for i in range(n//batch_size):
        # Getting the spectra and weights
        spectra_obs = torch.tensor(spectra[i*batch_size:(i+1)*batch_size],requires_grad=False,device='cuda:0',dtype=torch.float).view(batch_size,1,3829)
        weights = torch.tensor(1/(e_spectra[i*batch_size:(i+1)*batch_size]),requires_grad=False,device='cuda:0',dtype=torch.float).view(batch_size,1,3829)
        # Fitting a first time
        params, best_fits, valid = fit_LBFGS(model,spectra_obs,weights,loss,batch_size)
        # Masking the regions where the fit is bad
        abs_err = torch.abs(spectra_obs-best_fits)
        mask_threshold_red = 3*torch.std(abs_err[:,:,1980+25:1980+25+1791],dim=2).view(batch_size,1,1)
        mask_threshold_blue = 3*torch.std(abs_err[:,:,94:94+1791],dim=2).view(batch_size,1,1)
        weights[:,:,1980+25:1980+25+1791] = torch.where(abs_err[:,:,1980+25:1980+25+1791] < mask_threshold_red, weights[:,:,1980+25:1980+25+1791], torch.zeros_like(weights[:,:,1980+25:1980+25+1791]))
        weights[:,:,94:94+1791] = torch.where(abs_err[:,:,94:94+1791] < mask_threshold_blue, weights[:,:,94:94+1791], torch.zeros_like(weights[:,:,94:94+1791]))

        # Redoing the fit
        params, best_fits, valid = fit_LBFGS(model,spectra_obs,weights,loss,batch_size)
        # Writting to array
        params = params.detach().cpu().numpy()[:,0,:]
        sols_array[i*batch_size:(i+1)*batch_size,:] = params*valid
    # Saving results
    np.save(outfile,sols_array)

# Function to load observed spectra
def load_data_obs(data_file):
    with h5py.File(data_file,'r') as F:
        spectra = np.ones((F['spectra_red'].shape[0],3829))
        spectra[:,1980+25:1980+25+1791] = np.array(F['spectra_red'])
        spectra[:,94:94+1791] = np.array(F['spectra_blue'])
        e_spectra = 1e10*np.ones((F['e_spectra_red'].shape[0],3829))
        e_spectra[:,1980+25:1980+25+1791] = np.array(F['e_spectra_red'])
        e_spectra[:,94:94+1791] = np.array(F['e_spectra_blue'])

        # masking regions problematic to fits. In this case this allows masking
        # the red arm or the CaT
        e_spectra[:,1980+25:1980+25+1791] = 0*e_spectra[:,1980+25:1980+25+1791]+1e10
        # e_spectra[:,2471:2476] = 0*e_spectra[:,2471:2476]+1e10
        # e_spectra[:,2651:2656] = 0*e_spectra[:,2651:2656]+1e10
        # e_spectra[:,3142:3147] = 0*e_spectra[:,3142:3147]+1e10

        # also masking pixels with high errors
        e_spectra[np.where(e_spectra > 1.25)] = 1e10
    return spectra, e_spectra

if __name__=='__main__':
    # Loading data into ram
    spectra, e_spectra = load_data_obs(data_file)
    # Initializing the AE
    NN = DNN([200,268,397,569,739]).to('cuda:0')
    NN.load_state_dict(torch.load(model_name))
    NN.eval()

    # Doing the fits
    fit_n_LBFGS(n_spectra,NN,batch_size,loss,outfile)
