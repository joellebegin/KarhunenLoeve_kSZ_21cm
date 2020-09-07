import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import inv,eigh,cholesky
from scipy import interpolate
import os

class KL_transform:
    '''Computes the Karhunen-Loeve transform for two matrices, M1 and M2, 
    generating the M1-to-M2 transformation matrix'''

    def __init__(self, M1, M2):
       

        #initializing some attributes
        self.M1 = M1
        self.M2 = M2 
        self.M_dim = self.M1.shape[0]
        

    def do_KL_transform(self, sort_eig = False):
        '''Comptues R, which is the transforamtion matrix of the KL transform. 

        sort_eig: Bool or str
            deciding order of eigenvalues
            
            sort_eig = False --> randomly ordered eigenvalues 
                     = "ascending" --> small to large 
                     = "descending" --> large to small
        '''
        self.sort_eig = sort_eig

        #diagonalizes the first matrix
        self.R1 = self.diagonalizing_matrix(self.M1)
        self.M1_prime = self.R1.T@self.M1@self.R1 
        
        #matrix such that R2^{-1} @ M1_prime @ R2 = Identity
        self.R2 = np.identity(self.M1.shape[0])*(1/np.sqrt(np.diag(self.M1_prime)))
        self.M2_tilde = self.R2.T@self.R1.T@self.M2@self.R1@self.R2

        #diagonalizes M2_tilde 
        self.R3 = self.diagonalizing_matrix(self.M2_tilde)
    
        self.R = self.R1@self.R2@self.R3  #the transformation matrix
        self.get_modes() #getting modes of transformation matrix

    def diagonalizing_matrix(self, matrix):
        '''returns matrix M such that ( M.T @ matrix @ M ) is diagonal'''
        eigvals, eigvects = np.linalg.eig(matrix)
        
        if self.sort_eig == "ascending": #[smallest eig,...,largest eig]
            M_diagonalizing = np.real((eigvects.T[np.argsort(eigvals)]).T)

        elif self.sort_eig == "descending": #[largest eig,...,smallest eig]
            M_diagonalizing = np.real((eigvects.T[np.flip(np.argsort(eigvals))]).T)
        
        else:        
            M_diagonalizing = np.real(eigvects)
        
        return M_diagonalizing


    def get_modes(self):
        '''Getting the KL modes'''
        self.KL_modes = []
        for i in range(self.M_dim):
            v = np.zeros(self.M_dim)
            v[i] = 1
            self.KL_modes.append(np.matmul(self.R, v))

    def plot_modes(self, n_modes, scatter = False, save = False, savename = "KL_modes.png"):
        "Plots n_modes in same axis"

        fig, ax = plt.subplots()
        for i in range(n_modes):
            lab = "Mode " + str(i)
            if scatter:
                ax.scatter(np.arange(self.M_dim),self.KL_modes[i], label = lab)
            else:
                ax.plot(np.arange(self.M_dim),self.KL_modes[i], label = lab)
        ax.grid()
        plt.legend()

        if save:
            plt.savefig(savename, bbox_inches = 'tight')

    def plot_mode(self, mode, scatter = False):
        "plots a single KL mode"
        fig, ax = plt.subplots()
        
        lab = "Mode " + str(mode)
        ax.set_title(lab)
        if scatter:
            ax.scatter(np.arange(self.M_dim),self.KL_modes[mode], label = lab)
        else:
            ax.plot(np.arange(self.M_dim),self.KL_modes[mode], label = lab)
        
        ax.grid()
        plt.legend()
        plt.show()


class KL_Choleski:

    def __init__(self,f1, f2, eps1=0, eps2=0):
        '''does f1-to-f2 signal'''

        self.f1 = f1
        self.f2 = f2
        self.eps1 = eps1
        self.eps2 = eps2

        self.cholesky_method()


    #=============================== Functions ================================#

    def perturb_matrix(self, matrix, epsilon):
        return matrix + epsilon*np.eye(matrix.shape[0])

    def get_sorted_eig(self,matrix):
        vals, vects = eigh(matrix)
        inds = np.flip(np.argsort(vals))
        return vals[inds], vects.T[inds]

    def cholesky_method(self):
        
        self.f1_perturbed = self.perturb_matrix(self.f1, self.eps1)
        self.f2_perturbed = self.perturb_matrix(self.f2, self.eps2)

        self.cov_1 = inv(self.f1_perturbed) #kSZ
        self.cov_2 = inv(self.f2_perturbed) #21cm

        L= cholesky(self.cov_2, lower=True)
        
        #the G matrix
        G = inv(L)@self.cov_1@inv(L.T)
        vals, vects = self.get_sorted_eig(G)
        psi = vects.T 
        
        #KL trasform matrix
        self.R = L@psi

        #transformed matrices
        self.cov_2_prime = psi.T@inv(L)@self.cov_2@inv(L.T)@psi
        self.cov_1_prime = psi.T@inv(L)@self.cov_1@inv(L.T)@psi

        #the modes
        modes = []
        for i in range(len(vals)):
            e_i = np.zeros_like(vects[0])
            e_i[i] =1
            v = L@psi@e_i
            modes.append(v)

        self.modes = np.array(modes)
