



import emcee
import numpy as np
#import matplotlib.pyplot as plt
from nbodykit.lab import *
from numpy.linalg import inv
import scipy.integrate as integrate
import math
import scipy.optimize as op
from scipy.optimize import curve_fit
import numpy.linalg as linalg
from multiprocessing import Pool
import tqdm
import h5py


from argparse import ArgumentParser
ap = ArgumentParser(description='BAOfitter')
ap.add_argument('--inputpk', type=str, default='pk.json', help='input power spectrum in json format')
ap.add_argument('--covmatrix', type=str, default='covmax.txt', help='covariance matrix')
ap.add_argument('--redshift', type=float, default=1.5, help='eg: default redshift 1.5')
ap.add_argument('--outputMC', type=str, default='output.txt', help='output file for writing out the MC results')
ns = ap.parse_args()  

#inside python
inputpk = ns.inputpk
covpath = ns.covmatrix
redshift = ns.redshift
#outputmc = open(ns.outputmc,'w')
outputMC = ns.outputMC
r = ConvolvedFFTPower.load(inputpk)
poles = r.poles

#modes = poles['modes']
P0dat = poles['power_0'].real
P2dat = poles['power_2'].real
kdat = poles.coords['k'][1:poles.coords['k'].size]
kdat = poles['k']
kobs = kdat[1:24]
P0dat = P0dat[1:24]
P2dat = P2dat[1:24]


#dat = np.loadtxt('EZmock/PK_EZmock_eBOSS_QSO_NGC_v5_z0.8z2.2.dat')
#kobs = dat[1:24,1]
#P0dat = dat[1:24,2]
#P2dat = dat[1:24,4]

Pkdata = np.append(P0dat,P2dat)
size = Pkdata.size
half = int(size/2)


temp = np.loadtxt('/home/merz/workdir/emcee/EZmock/PlanckDM.linear.pk')
ktemp = temp[:,0]
Plintemp = temp[:,1]


cov = np.load(covpath)
covinv = inv(cov)



ell = [0,2]


print(ell,redshift)


cosmo = cosmology.Cosmology(h=0.676,Omega0_b=0.04814257).match(Omega0_m=0.31)   #eBOSS cosmology
#cosmo =cosmology.Cosmology(h=0.6777,Omega0_b=0.048206,n_s=0.9611).match(Omega0_m=0.307115) #EZmock cosmology
Plinfunc = cosmology.LinearPower(cosmo, redshift=redshift, transfer='CLASS')
Psmlinfunc = cosmology.LinearPower(cosmo, redshift=redshift, transfer='NoWiggleEisensteinHu')

muobs = np.linspace(-1,1,100)
sigpar = 8.
sigperp = 3.

# In[104]:

def polyf(j,k):

    if j==0 or j==5:
        h = 1./k**3
        return h
    if j ==1 or j==6:
        h = 1./k**2
        return h
    if j ==2 or j==7:
        h = 1./k
        return h
    if j==3 or j==8:
        h = 1.
        return h
    if j==4 or j==9:
        h = k
        return h
        
def polysolve(resmodel):
    C1 = linalg.pinv((np.matmul(Ht,np.matmul(covinv,H))))
    C2 = np.matmul(Ht,np.matmul(covinv,resmodel))
    theta = np.matmul(C1,C2)
    return(theta)
    
    
H = np.zeros((size,10))
for i in range(0,size):
    for j in range(0,5):
        if(i<half):
            H[i][j] = polyf(j,kobs[i])
        if(i>=half):
            H[i][j+5] = polyf(j+5,kobs[i-half])
        
Ht = H.transpose()

def Psmfitfunopt(k,a1,a2,a3,a4,a5):
    Psmfitpre = Psmlinfunc(ktemp[2900:5900]) + a1/ktemp[2900:5900]**3 + a2/ktemp[2900:5900]**2 
    + a3/ktemp[2900:5900] + a4 + a5*ktemp[2900:5900]
    Psmfit = np.interp(k,ktemp[2900:5900],Psmfitpre)
    return Psmfit


        
def Olin(k):
    #a1 = popt[0]
    #a2= popt[1]
    #a3 = popt[2]
    #a4 = popt[3]
    #a5 = popt[4]
    Olin = Plinfunc(k)/Psmfit(k)
    return Olin


popt,pcov = curve_fit(Psmfitfunopt,ktemp[2900:5900],Plinfunc(ktemp[2900:5900]))
asm1 = popt[0]
asm2= popt[1]
asm3 = popt[2]
asm4 = popt[3]
asm5 = popt[4]

Psmfitopt = Psmfitfunopt(ktemp[2900:5900],asm1,asm2,asm3,asm4,asm5)

def Psmfit(k):
	Psmfit = np.interp(k,ktemp[2900:5900],Psmfitopt)
	return Psmfit


#construct the model to compare to the data
class model:

    def __init__(self, params):

        self.B = params[0]
        self.beta = params[1]
        self.alpha_perp = params[2]
        self.alpha_par = params[3]
        self.sigs = params[4]

        
        self.F = self.alpha_par / self.alpha_perp

    def Legendre(self,ell):
        if ell == 0:
            L = 1
        elif ell ==2:
            L = 0.5 *(3*muobs**2-1)
        #elif ell ==4:
            #L = 1.0/8*(35*mu**4 - 30*mu**2 +3)
        return L

    def kprime(self,k):
        
        kp = k/self.alpha_perp * (1.0 + muobs**2 * (1.0/self.F**2 - 1.0))**0.5
        return kp
    
    def muprime(self):
        
        mup = muobs/self.F * (1.0 + muobs**2 * (1.0/self.F**2 - 1.0))**(-0.5)
        return mup

    def Pkmuf(self,kobs,ell):
        mup = self.muprime()
        Pkmuint = []
        for k in kobs:
            kp = self.kprime(k)
            Psmkmu = self.Psmkmuf(mup,kp)
            Pkmu = Psmkmu * (1+ (Olin(kp) -1) * 
               np.exp(-1*(kp**2 * mup**2 * sigpar**2 + kp**2*(1-mup**2)*sigperp**2)/2.0))
            Pkmuint.append(Pkmu)
        return np.asarray(Pkmuint)
        #return True


    def Psmkmuf(self,mu,k):
        R = 1.0
        Pskmu = np.exp(self.B) * (1+self.beta*mu**2 *R)**2 * Psmfit(k) * self.Ffogf(mu,k)
        return Pskmu



    def Ffogf(self,mu,k):
        Ffog = 1.0/(1+(k**2 * mu**2 * self.sigs**2)/2)**2
        return Ffog


    def run(self,kobs,ell):
        
        
        for ell in ell:
            L = self.Legendre(ell)
            Pkmu = self.Pkmuf(kobs,ell)
            integrand = Pkmu*L
            integral = integrate.simps(integrand,x=muobs,axis=1)
            if ell ==0:
                P_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integral

            if ell ==2:
                P_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integral
                
        P0_res = P0dat - P_0
        P2_res = P2dat - P_2
        
        res = np.append(P0_res,P2_res)
        Al = polysolve(res)
        A0 =0
        A2 = 0
        for i in range(0,5):
            A0 += Al[i]* polyf(i,kobs)
            A2 += Al[i+5]*polyf((i+5),kobs)
        return np.append(P_0+A0,P_2+A2)
        #return np.append(P_0,P_2)


# In[65]:


B = 2.12
beta = 0.5
alpha_perp = 1.0
alpha_par = 1.0
sigs = 3.0



paramstest = [B,beta,alpha_perp,alpha_par,sigs]
test = model(paramstest)



#calculate the chi-square
poles = [0,2]
def chi2f(params):
    modelP = model(params)
    #need to loop over k
    #Pkmodel = np.array(list(map(modelP.run, kobs)))
    #chi2 = np.sum(((P0dat[3:59]-Pkmodel)/error[3:59])**2))

    
    Pkmodel = modelP.run(kobs,poles)
    
    vec = Pkdata - Pkmodel
    vec = np.matrix(vec)
    vect = vec.transpose()
    intermediate = np.matmul(covinv,vect)
    chisq = np.matmul(vec,intermediate)

    if not np.isfinite(chisq):
         return -np.inf

    return -0.5*chisq.item() + log_prior(params)



def log_prior(params):
    B = params[0]
    beta = params[1]
    alpha_perp = params[2]
    alpha_par = params[3]
    sigs = params[4]  
    if 0.8 < alpha_perp < 1.2 and 0.8 < alpha_par < 1.2  and 0.15<beta<0.75 and 0.0<sigs<10 and 0.0<B<3:
        return 0.0
    return -np.inf

#Pkdata = P0dat
params = paramstest
pos0 = paramstest + 1e-4*np.random.randn(35, 5)
nwalkers, ndim = pos0.shape


# Set up the backend
# Don't forget to clear it in case the file already exists
filename = outputMC
#filename = '/home/workdir/emcee/mehdi/test.h5'
backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)
#print(backend.shape)
# Initialize the sampler
with Pool() as pool:
	sampler = emcee.EnsembleSampler(nwalkers, ndim, chi2f, backend=backend, pool=pool)
	sampler.run_mcmc(pos0, 10000, progress=True)

#outputmc.close()
