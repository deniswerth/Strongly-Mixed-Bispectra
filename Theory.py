import numpy as np

class theory:

    def __init__(self, N, Nfield, interpolated):
        self.N = N
        self.Nfield = Nfield
        self.interpolated = interpolated

        # unpack interpolators
        self.H_f = interpolated[0]
        self.m2_f = interpolated[1]
        self.rho_f = interpolated[2]
        self.kappa1_f = interpolated[3]
        self.kappa2_f = interpolated[4]
        self.lambda11_f = interpolated[5]
        self.lambda12_f = interpolated[6]
        self.lambda2_f = interpolated[7]
        self.lambda3_f = interpolated[8]
        self.dH_interp = interpolated[9]
        self.a_interp = interpolated[10]
        self.scale_interp = interpolated[11]
        self.dscale_interp = interpolated[12]

        N = self.N

        # evaluate once
        self.H = self.H_f(N)
        self.m2 = self.m2_f(N)
        self.rho = self.rho_f(N)
        self.kappa1 = self.kappa1_f(N)
        self.kappa2 = self.kappa2_f(N)
        self.lambda11 = self.lambda11_f(N)
        self.lambda12 = self.lambda12_f(N)
        self.lambda2 = self.lambda2_f(N)
        self.lambda3 = self.lambda3_f(N)

        self.dH = self.dH_interp(N)
        self.a = self.a_interp(N)
        self.scale = self.scale_interp(N)
        self.dscale = self.dscale_interp(N)


    def k_mode(self, N_exit):
        return self.a_f(N_exit) * self.H_f(N_exit)


    def Delta_ab(self):
        Nfield = self.Nfield
        Deltaab = np.eye(Nfield) # Identity matrix of size Nfield.Nfield
        return Deltaab

    def I_ab(self):
    	Nfield = self.Nfield
    	Iab = np.zeros((Nfield, Nfield))
    	Iab[0, 1] = self.rho
    	return Iab

    def M_ab(self, k):
    	Nfield = self.Nfield
    	Mab = np.eye(Nfield)
    	Mab[0, 0] = -k**2/self.a**2
    	Mab[1, 1] = -k**2/self.a**2 - self.m2 - self.rho**2
    	return Mab


    # Define the cubic theory tensors
    def A_abc(self, k1, k2, k3):
    	Nfield = self.Nfield
    	Aabc = np.zeros((Nfield, Nfield, Nfield))
    	k1k2 = (k3**2 - k1**2 - k2**2)/2
    	k1k3 = (k2**2 - k1**2 - k3**2)/2
    	k2k3 = (k1**2 - k2**2 - k3**2)/2
    	Aabc[1, 1, 1] += -self.lambda3/3 + self.rho*self.lambda2 - self.lambda11 * self.rho**2 + 1*self.kappa1/3*self.rho**3
    	Aabc[0, 0, 1] += (self.lambda12/3 - self.kappa2*self.rho) * k1k2/self.a**2
    	Aabc[0, 1, 0] += (self.lambda12/3 - self.kappa2*self.rho) * k1k3/self.a**2
    	Aabc[1, 0, 0] += (self.lambda12/3 - self.kappa2*self.rho) * k2k3/self.a**2
    	return Aabc

    def A_abc_fast(self, k1, k2, k3):  # For initial conditions
        Nfield = self.Nfield
        Aabc = np.zeros((Nfield, Nfield, Nfield))
        k1k2 = (k3**2 - k1**2 - k2**2) / 2
        k1k3 = (k2**2 - k1**2 - k3**2) / 2
        k2k3 = (k1**2 - k2**2 - k3**2) / 2
        Aabc[0, 0, 1] += (self.lambda12/3 - self.kappa2 * self.rho) * k1k2 / self.a**2
        Aabc[0, 1, 0] += (self.lambda12/3 - self.kappa2 * self.rho) * k1k3 / self.a**2
        Aabc[1, 0, 0] += (self.lambda12/3 - self.kappa2 * self.rho) * k2k3 / self.a**2
        return Aabc

    def A_abc_slow(self, k1, k2, k3):
    	Nfield = self.Nfield
    	Aabc = np.zeros((Nfield, Nfield, Nfield))
    	Aabc[1, 1, 1] += -self.lambda3/3 + self.rho*self.lambda2 - self.lambda11 * self.rho**2 + 1*self.kappa1/3*self.rho**3
    	return Aabc

    def B_abc(self, k1, k2, k3):
        Nfield = self.Nfield
        Babc = np.zeros((Nfield, Nfield, Nfield))
        k1k2 = (k3**2 - k1**2 - k2**2)/2
        Babc[1, 1, 0] += -self.lambda2 + 2*self.lambda11 * self.rho - 1*self.kappa1*self.rho**2
        Babc[0, 0, 0] += self.kappa2*k1k2/self.a**2
        return Babc

    def C_abc(self, k1, k2, k3):
    	Nfield = self.Nfield
    	Cabc = np.zeros((Nfield, Nfield, Nfield))
    	Cabc[0, 0, 1] += -self.lambda11 + 1*self.kappa1*self.rho
    	return Cabc

    def D_abc(self, k1, k2, k3):
        Nfield = self.Nfield
        Dabc = np.zeros((Nfield, Nfield, Nfield))
        Dabc[0, 0, 0] += - self.kappa1/3
        return Dabc


    # Define the u-tensors
    def u_AB(self, k):
    	Nfield = self.Nfield
    	H = self.H
    	s = self.scale
    	ds = self.dscale
    	S = np.ones((Nfield, Nfield)) + (s-1)*np.eye(Nfield)
    	uAB = np.zeros((2*Nfield, 2*Nfield))
    	uAB[:Nfield, :Nfield] = -self.I_ab()/H
    	uAB[:Nfield, Nfield:] = self.Delta_ab()/H /s
    	uAB[Nfield:, :Nfield] = self.M_ab(k)/H *s
    	uAB[Nfield:, Nfield:] = (self.I_ab()).T/H - 3*self.H*np.eye(Nfield)/H + ds/s*np.eye(Nfield)/H 
    	return uAB


    def u_ABC(self, k1, k2, k3):
    	Nfield = self.Nfield
    	s = self.scale
    	S = np.ones((Nfield, Nfield, Nfield)) + (s-1)*np.eye(Nfield)
    	H = self.H
    	uABC = np.zeros((2*Nfield, 2*Nfield, 2*Nfield))

    	A123 = self.A_abc(k1, k2, k3)
    	B123 = self.B_abc(k1, k2, k3)
    	B132 = self.B_abc(k1, k3, k2)
    	B231 = self.B_abc(k2, k3, k1)
    	C123 = self.C_abc(k1, k2, k3)
    	C132 = self.C_abc(k1, k3, k2)
    	C321 = self.C_abc(k2, k3, k1)
    	D123 = self.D_abc(k1, k2, k3)

    	for i in range(Nfield):
    		for j in range(Nfield):
    			for k in range(Nfield):
    				uABC[i, j, k] = -B231[j, k, i]/H
    				uABC[i, Nfield+j, k] = -C123[i, j, k]/H/s
    				uABC[i, j, Nfield+k] = -C132[i, k, j]/H/s
    				uABC[Nfield+i, Nfield+j, Nfield+k] = C321[k, j, i]/H/s
    				uABC[i, Nfield+j, Nfield+k] = 3.*D123[i, j, k]/H/s/s
    				uABC[Nfield+i, j, k] = 3.*A123[i, j, k]/H*s
    				uABC[Nfield+i, Nfield+j, k] = B132[i, k, j]/H
    				uABC[Nfield+i, j, Nfield+k] = B123[i, j, k]/H
    	return uABC

















