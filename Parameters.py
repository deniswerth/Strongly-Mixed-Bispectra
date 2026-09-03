import numpy as np

class parameters:
    """
    Precompute background functions on a grid and provide fast interpolators
    """

    def __init__(self, N_load, H_load, m2_load,
                 rho_load, kappa1_load, kappa2_load, lambda11_load, lambda12_load,
                 lambda2_load, lambda3_load):

        self.N_load = N_load

        # Store arrays
        self.H_load = H_load
        self.m2_load = m2_load
        self.rho_load = rho_load
        self.kappa1_load = kappa1_load
        self.kappa2_load = kappa2_load
        self.lambda11_load = lambda11_load
        self.lambda12_load = lambda12_load
        self.lambda2_load = lambda2_load
        self.lambda3_load = lambda3_load

        self.dH_load = np.gradient(H_load, N_load)

        self.k_load = 1
        self.a_load = np.exp(N_load)
        self.scale_load = self.a_load / (1. + self.a_load * self.H_load / self.k_load) / self.H_load
        self.dscale_load = (-self.dH_load / self.H_load**2 * self.a_load / (1. + self.a_load * self.H_load / self.k_load)
                + self.a_load / (1. + self.a_load * self.H_load / self.k_load)
                - self.a_load * (self.a_load * self.H_load**2 / self.k_load + self.a_load * self.dH_load / self.k_load) / (1. + self.a_load * self.H_load / self.k_load)**2 / self.H_load)


    # -------- FAST INTERPOLATORS --------

    def H_f(self, N):
        return np.interp(N, self.N_load, self.H_load)

    def m2_f(self, N):
        return np.interp(N, self.N_load, self.m2_load)

    def rho_f(self, N):
        return np.interp(N, self.N_load, self.rho_load)

    def kappa1_f(self, N):
        return np.interp(N, self.N_load, self.kappa1_load)

    def kappa2_f(self, N):
        return np.interp(N, self.N_load, self.kappa2_load)

    def lambda11_f(self, N):
        return np.interp(N, self.N_load, self.lambda11_load)

    def lambda12_f(self, N):
        return np.interp(N, self.N_load, self.lambda12_load)

    def lambda2_f(self, N):
        return np.interp(N, self.N_load, self.lambda2_load)

    def lambda3_f(self, N):
        return np.interp(N, self.N_load, self.lambda3_load)

    def dH_f(self, N):
        return np.interp(N, self.N_load, self.dH_load)

    def a_f(self, N):
        return np.interp(N, self.N_load, self.a_load)

    def scale_f(self, N):
        return np.interp(N, self.N_load, self.scale_load)

    def dscale_f(self, N):
        return np.interp(N, self.N_load, self.dscale_load)



    def output(self):
        return [self.H_f, self.m2_f, self.rho_f, self.kappa1_f, self.kappa2_f,
                self.lambda11_f, self.lambda12_f,
                self.lambda2_f, self.lambda3_f, self.dH_f, self.a_f, self.scale_f, self.dscale_f]





