from Functions import GSFisher, kSZ_Fisher, load_derivatives
import numpy as np 
import matplotlib.pyplot as plt 

F_21cm = GSFisher(N_zbins=59, N_samples=59, nu_lims= np.array([109.23,200]))
fisher_21cm = F_21cm.compute_fisher_fgnd_cov()

print(F_21cm.z_linspace)
