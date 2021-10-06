import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import time

from train_and_search import *
################################################################################
# predicting stellar labels of synthetic spectra with the emulator

# Emulator to use
model_name = 'models\\emulator_v6.pth'
# Datafile with the (synthetic) testing data
data_file = 'data_emulator_test.h5'
################################################################################
# Function to compute weighted residuals
def res(spectrum_true,err):
    def fun(labels):
        labels = torch.from_numpy(labels).to('cuda:0').float().view(-1,1,23)
        spectrum_pred = NN(labels).view((3829)).detach().cpu().numpy()
        residuals = (spectrum_true - spectrum_pred)/err
        return residuals
    return fun

# Forward pass on the pytorch model
def forward(labels):
    labels = torch.from_numpy(labels).to('cuda:0').float().view(-1,1,23)
    y = NN(labels).view((3829)).detach().cpu().numpy()
    return y

# Fitting on a single random spectrum with LBFGS, and plot results & stats
def fit_random_LBFGS(model,batch_size):
    while 1:
        params=0*np.random.rand(batch_size,1,23)
        params = torch.tensor(params,requires_grad=True,device='cuda:0',dtype=torch.float)
        params_found = False
        orig_params = params.clone()
        losses = []
        tol = 1e-10

        n = np.random.randint(0,inputs.shape[0]-batch_size)
        spectrum_true = torch.tensor(labels[n:n+batch_size],requires_grad=False,device='cuda:0',dtype=torch.float).view(batch_size,1,3829)
        # Masking red arm
        weights = torch.ones_like(spectrum_true)
        weights[:,:,1980:] = 0

        loss_fn = torch.nn.MSELoss(reduction='mean')
        orig_loss = loss_fn(model(params)*weights, spectrum_true*weights)

        start = time.time()
        while params_found is False:
            optimizer = torch.optim.LBFGS([params], lr=1, max_iter=100, line_search_fn='strong_wolfe')
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            max_epochs = 100
            for epoch in range(max_epochs):
                def closure():
                    optimizer.zero_grad()
                    fit = model(params)
                    loss = loss_fn(fit*weights, spectrum_true*weights)
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

            if ( len(losses)>1 and abs(losses[-1]-losses[-2]) < tol ):
                params_found=True

        end = time.time()

        best_fit = model(params).view(batch_size,3829).detach().cpu().numpy()
        stellar_labels = inputs[n:n+batch_size].reshape(batch_size,1,23)
        stellar_labels_tensor = torch.tensor(stellar_labels,requires_grad=False,device='cuda:0',dtype=torch.float)
        model_truth = model(stellar_labels_tensor)

        true_loss = loss_fn(model_truth, spectrum_true)
        print('fit loss', final_loss.item())
        print('truth loss', true_loss.detach().cpu().numpy())
        if batch_size == 1:
            print('fit', params.detach().cpu().numpy())
            print('truth', stellar_labels)
        print('time taken', end-start)

        spectrum_true = spectrum_true[0,0,:].view(3829).detach().cpu().numpy()
        model_truth = model_truth[0,0,:].detach().cpu().numpy()
        best_fit = np.reshape(best_fit[0], (3829))
        weights = weights[0,0,:].detach().cpu().numpy()

        plt.plot(spectrum_true,label='true spectrum')
        plt.plot(best_fit,label='fit')
        plt.plot(model_truth,label='NN(true params)')
        plt.plot(weights,label='weight')
        plt.legend()
        plt.show()

# Fitting on n spectra with LBFGS
def fit_LBFGS(n,model):
    sols_array = np.zeros((n,23))
    labels_array = np.zeros((n,23))
    err_array = np.zeros((n,23))

    tol = 1e-10
    for i in range(n):
        params = torch.tensor(np.zeros((1,1,23)),requires_grad=True,device='cuda:0',dtype=torch.float)
        params_found = False
        orig_params = params.clone()
        losses = []

        spectrum_true = torch.tensor(labels[i],requires_grad=False,device='cuda:0',dtype=torch.float).view(1,1,3829)
        # Masking red arm
        weights = torch.ones_like(spectrum_true)
        # weights[:,:,1980:] = 0

        loss_fn = torch.nn.MSELoss(reduction='mean')
        orig_loss = loss_fn(model(params)*weights, spectrum_true*weights)

        while params_found is False:
            optimizer = torch.optim.LBFGS([params], lr=1, max_iter=100, line_search_fn='strong_wolfe')
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            max_epochs = 100
            for epoch in range(max_epochs):
                def closure():
                    optimizer.zero_grad()
                    fit = model(params)

                    loss = loss_fn(fit*weights, spectrum_true*weights)
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

        stellar_labels = inputs[i]
        stellar_labels_tensor = torch.tensor(stellar_labels,requires_grad=False,device='cuda:0',dtype=torch.float)
        true_loss = loss_fn(model(stellar_labels_tensor), spectrum_true)

        best_sol = params.detach().cpu().numpy()[0,0]
        sols_array[i,:] = best_sol
        labels_array[i,:] = stellar_labels
        err_array[i,:] = best_sol-stellar_labels

    # Saving the results, labels and errors
    np.save('test_results\\sols_v2.npy',sols_array)
    np.save('test_results\\labels_v2.npy',labels_array)
    np.save('test_results\\err_v2.npy',err_array)


if __name__=='__main__':
    # Loading data into ram
    inputs,labels = load_data(data_file)
    # Initializing the AE
    NN = DNN([5,[200,268,397,569,739],2.3e-3,256]).to('cuda:0')
    NN.load_state_dict(torch.load(model_name))
    NN.eval()

    fit_random_LBFGS(NN,32)
    # fit_LBFGS(360,NN)
