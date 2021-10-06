import h5py
import numpy as np
################################################################################
# Code to pre-process the emulator data. This includes:
# -Removing any spectrum containing nans
# -Removing any spectrum where the continuum normalization gives abnormal flux values
# -Normalizing stellar labels and saving the normalization means and standard deviations

# Raw synthetic data
data_file = 'data_emulator.h5'
# Output files for training and testing
name_train = 'data_emulator_train.h5'
name_test = 'data_emulator_test.h5'
# Settings
test_frac = 0.1
label_names = ['Al','Ba','C','Ca','Co','Cr','Eu','Mg','Mn','N','Na','Ni','O','Si','Sr','Ti','Zn','logg','teff','m_h','vsini','vt','vrad']
################################################################################
# Loading all data
F = h5py.File(data_file,'r')
n_spectra = F['spectra_asymnorm_noiseless'].shape[0]

# Copying the spectra to the train and test files, and removing nans
spectra = np.array(F['spectra_asymnorm_noiseless'])
spectra[:,1979] = 1
spectra[:,3828] = 1
bad_indexes = list(np.unique(np.where(np.isnan(spectra)==True)[0]))
for i, spectrum in enumerate(spectra):
    if np.max(spectrum) > 2:
        bad_indexes.append(i)

good_indexes = []
for i in range(n_spectra):
    if i not in bad_indexes:
        good_indexes.append(i)

spectra = spectra[good_indexes]
n_spectra = len(good_indexes)
n_train = int(n_spectra*(1-test_frac))

F_train = h5py.File(name_train,'w')
F_test = h5py.File(name_test,'w')
F_train.create_dataset('spectra_asymnorm_noiseless',data=spectra[:n_train])
F_test.create_dataset('spectra_asymnorm_noiseless',data=spectra[n_train:])
# Copying the labels to the train and test files, normalizing them first
mean = np.zeros(23)
std = np.zeros(23)
for i, label in enumerate(label_names):
    data = np.array(F[label][good_indexes])
    mean[i] = np.mean(data)
    std[i] = np.std(data)
    data = (data - np.mean(data))/np.std(data)
    F_train.create_dataset(label,data=data[:n_train])
    F_test.create_dataset(label,data=data[n_train:])
# Saving arrays containing the means and standard deviations
np.save('mean.npy',mean)
np.save('std.npy',std)
