
import emcee
import numpy as np
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
from configobj import ConfigObj
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

import sys



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
    Hartlap = (1000-size-2)/(1000-1)
    if combined:
        Hartlap = (1000-size//2-2)/(1000-1)
    C1 = linalg.pinv((np.matmul(Ht,np.matmul(Hartlap*covinvh,H))))
    C2 = np.matmul(Ht,np.matmul(Hartlap*covinvh,resmodel))
    theta = np.matmul(C1,C2)
    return(theta)
    

def prepare_poly_k(ell,convolved):
    if not combined:
        km = np.linspace(0.001,0.4,endpoint=False,num=400)
        kmodel_vec = np.concatenate([km,km,km])
        kw = np.dot(M,kmodel_vec)
        kw = np.dot(W,kw)
        newkw = np.reshape(kw,(5,40))
        kslice10 = newkw[0][int(kmin/0.01):int(kmax/0.01)]
        kslice12 = newkw[2][int(kmin/0.01):int(kmax/0.01)]
        kslice14 = newkw[4][int(kmin/0.01):int(kmax/0.01)]
                                
        
        if 4 in ell and convolved:
            kwm = [kslice10,kslice12,kslice14]

        elif 4 in ell and not convolved:
            kwm = [kobs,kobs,kobs]
            km = kobs

        elif not 4 in ell and convolved:
            kwm = [kslice10,kslice12]
            
        elif not 4 in ell and not convolved:
            kwm = [kobs,kobs]
            km = kobs

                                                                                                                                                                                                                                                                                                                      

    if combined:

            km1 = np.linspace(0.001,0.4,endpoint=False,num=400)
            km2 = np.linspace(0.001,0.4,endpoint=False,num=400)
            modelhalf = km1.size
            km = np.concatenate([km1,km2])
            modelsize = km.size
            kmodel_vec = np.concatenate([km,km,km])

            kw = np.matmul(M,kmodel_vec)
            kw = np.matmul(W,kw)
            newkw = np.reshape(kw,(10,40))

            kslice10 = newkw[0][2:23]
            kslice12 = newkw[2][2:23]
            kslice14 = newkw[4][2:23]

            kslice20 = newkw[5][2:23]
            kslice22 = newkw[7][2:23]
            kslice24 = newkw[9][2:23]                                                                

            if 4 in ell and convolved:
                kwm = [kslice10,kslice12,kslice14,kslice20,kslice22,kslice24]

            elif 4 in ell and not convoled:
                kwm = [kobs,kobs,kobs,kobs,kobs,kobs]
                km = kobs

            elif not 4 in ell and convolved:
                kwm = [kslice10,kslice12,kslice20,kslice22]

            elif not 4 in ell and not convolved:
                kwm = [kobs,kobs,kobs,kobs]
                km = kobs
            
            
            
    return kwm,km

                      

def Psmfitfunopt(k,a1,a2,a3,a4,a5):
    Psmfitpre = Psmlinfunc(ktemp) + a1/ktemp**3 + a2/ktemp**2 
    + a3/ktemp + a4 + a5*ktemp
    #Psmfit = np.interp(k,ktemp[2900:5900],Psmfitpre)
    Pspl = IUS(ktemp,Psmfitpre)
    
    #return Psmfit
    return Pspl(k)



def Olin(k):
    #a1 = asm1
    #a2 = asm2
    #a3 = asm3
    #a4 = asm4
    #a5 = asm5
    Olin = Plinfunc(k)/Psmfit(k)
    return Olin




def Psmfit(k):
        #Psmfit = np.interp(k,ktemp[2900:5900],Psmfitopt)
        Psmfit = IUS(ktemp,Psmfitopt)
        #return Psmfit
        return Psmfit(k)

def Legendre(el):
    if el == 0:
        L = 1
    elif el ==2:
        L = 0.5 *(3*muobs**2-1)
    elif el ==4:
        L = 1.0/8*(35*muobs**4 - 30*muobs**2 +3)
    return L

class model:

        def __init__(self, params,combined):
                self.B = params[0]
                self.alpha_perp = params[1]
                self.alpha_par = params[2]
                self.beta = f/self.B
                #self.fn = params[3]
                #self.beta = self.fn/self.B
                #self.sigpar = params[4]
                #self.sigperp = params[5]
                #self.sigs = params[6]
                
                if combined:
                        self.B2 =params[3]
                        self.beta2 = f/self.B2
                        #self.B2 = params[4]
                        #self.fn2 = params[5]
                        #self.beta2 = self.B2/self.fn

                self.F = self.alpha_par / self.alpha_perp

                self.combined = combined

        def kprime(self,k):

                kp = k/self.alpha_perp * (1.0 + muobs**2 * (1.0/self.F**2 - 1.0))**0.5
                return kp

        def muprime(self):

                mup = muobs/self.F * (1.0 + muobs**2 * (1.0/self.F**2 - 1.0))**(-0.5)
                return mup

        def Pkmuf(self,kobs):

                combined = self.combined

                mup = self.muprime()
                Pkmuint = []

                if combined:
                        cap =1
                        for k in kobs[0:modelhalf]:
                                kp = self.kprime(k)
                                Psmkmu1 = self.Psmkmuf(mup,kp,k,cap)

                                Pkmu1 = Psmkmu1 * (1+ (Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * sigpar**2 + kp**2*(1-mup**2)*sigperp**2)/2.0))
                                Pkmuint.append(Pkmu1)
                        cap=2
                        for k in kobs[modelhalf:modelsize]:
                                kp = self.kprime(k)
                                Psmkmu2 = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu2 = Psmkmu2 * (1+ (Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * sigpar**2 + kp**2*(1-mup**2)*sigperp**2)/2.0))

                                Pkmuint.append(Pkmu2)	

                        return np.asarray(Pkmuint)
        
		
                else:
                        cap =1
                        for k in kobs:
                                kp = self.kprime(k)
                                Psmkmu = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu = Psmkmu * (1+ (Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 *sigpar**2 + kp**2*(1-mup**2)*sigperp**2)/2.0))
                                Pkmuint.append(Pkmu)
                        return np.asarray(Pkmuint)
		

        def Psmkmuf(self,mup,kp,k,cap):
                R = 1.0
                if cap ==1:
                        Pskmu = (self.B**2) * (1+self.beta*mup**2 *R)**2 * Psmfit(k) * self.Ffogf(mup,kp)

                if cap ==2:
                        Pskmu = (self.B2**2) * (1+self.beta2*mup**2 *R)**2 * Psmfit(k) * self.Ffogf(mup,kp)

                return Pskmu


        def Ffogf(self,mu,k):
		#Ffog = 1.0/(1+(k**2 * mu**2 * sigs**2)/2)**2
                Ffog = 1.0/(1+((k*mu*sigs)**2)/2)
                return Ffog


        def run(self,k):

                combined = self.combined
                if combined:
                        Pkmu = self.Pkmuf(k)
                        Pkmu1 = Pkmu[0:modelhalf]
                        Pkmu2 = Pkmu[modelhalf:modelsize]

                        integrand10 = Pkmu1*L0
                        integrand12 = Pkmu1*L2
                        integrand14 = Pkmu1*L4

                        integrand20 = Pkmu2*L0
                        integrand22 = Pkmu2*L2
                        integrand24 = Pkmu2*L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand12,x=muobs,axis=1)
                        P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand14,x=muobs,axis=1)

                        P_2_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand20,x=muobs,axis=1)
                        P_2_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand22,x=muobs,axis=1)
                        P_2_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand24,x=muobs,axis=1)	


                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4,P_2_0,P_2_2,P_2_4])
                        if convolved:
                            WPkm = np.dot(W,np.dot(M,Pkml))
                            newmod = np.reshape(WPkm,(10,40))
                            P_1_0 = newmod[0,2:23]
                            P_1_2 = newmod[2,2:23]
                            P_1_4 = newmod[4,2:23]
                            P_2_0 = newmod[5,2:23]
                            P_2_2 = newmod[7,2:23]
                            P_2_4 = newmod[9,2:23]
                            
                            
                            
                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4,P_2_0,P_2_2,P_2_4])
                            

                        res = Pkml-WPkm_cut
                        
                        BB = solver.BBk(res).flatten()
                       
                        Pkmodel = Pkml + BB
                        
                        return Pkmodel


                else:

                        Pkmu1 = self.Pkmuf(k)
                        integrand10 = Pkmu1*L0
                        integrand12 = Pkmu1*L2
                        integrand14 = Pkmu1*L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand12,x=muobs,axis=1)
                        P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand14,x=muobs,axis=1)
    
                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4])

        

                        if convolved:
                            convolved_model = np.dot(W,np.dot(M,Pkml))
                            newmod = np.reshape(convolved_model,(5,40)) 
                            P_1_0 = newmod[0,int(kmin/0.01):int(kmax/0.01)]
                            P_1_2 = newmod[2,int(kmin/0.01):int(kmax/0.01)]
                            P_1_4 = newmod[4,int(kmin/0.01):int(kmax/0.01)]
                        
        
                        if 4 in ell:
                            Pkml = np.concatenate([P_1_0,P_1_2,P_1_4])

                
                        elif not 4 in ell:
                            Pkml = np.concatenate([P_1_0,P_1_2])


                        
                        res = Pkdata-Pkml
                        BB = solver.BBk(res).flatten()
                        
                        Pkmodel = Pkml + BB
                        #Pkmodel = Pkm+WAkm


                        return Pkmodel
        








def chi2f(params):
    modelP = model(params,combined)

    if combined:
            hs = size//2
    else:
            hs = size
                        
    Hart = (1000-hs-2)/(1000-1)

    covinvhart = covinv*Hart
    Pkmodel = modelP.run(km)

    vec = Pkdata - Pkmodel
    
    chisq = np.dot(vec,np.dot(covinvhart,vec))

    if not np.isfinite(chisq):
         return np.inf

    return chisq-log_prior(params)
    #return -0.5*chisq + log_prior(params)



def lnlh(params):
        chi2 = chi2f(params)
        return -0.5*chi2 + log_prior(params)


def log_prior(params):
    if combined:
            B = params[0]
            #beta = params[1]
            alpha_perp = params[1]
            alpha_par = params[2]
            #f1 = params[3]
            B2 = params[3]
            #f2 = params[5]

            f1 = 1.0
            f2 = 1.0
                                        
            if 0.8 < alpha_perp < 1.2 and 0.8 < alpha_par < 1.2 and 0.0<B<10 and 0<f1<10 and 0<B2<10 and 0<f2<10:
                    return 0.0
    else:

            B = params[0]
            #beta = params[1]
            alpha_perp = params[1]
            alpha_par = params[2]
            #f = params[3]
            f = 1.0        
            if 0.8 < alpha_perp < 1.2 and 0.8 < alpha_par < 1.2 and 0.0<B<10 and 0<f<10.0:
                    return 0.0
    return -np.inf



if __name__ == "__main__":

    import sys
    sys.path.append("./")
    sys.path.insert(0, '/home/merz/workdir/emcee/BAOfitter/')
    from analyticBBsolver import LLSQsolver

    pardict = ConfigObj('/home/merz/workdir/emcee/BAOfitter/config.ini')

    #Cosmo params
    redshift = float(pardict["z"])
    h = float(pardict["h"])
    n_s = float(pardict["n_s"])
    omb0 = float(pardict["omega0_b"])
    Om0 = float(pardict["omega0_m"])
    sig8 = float(pardict["sigma_8"])



    linearpk = pardict["linearpk"]
    inputpk = pardict["inputpk"]
    window = pardict["window"]
    wideangle = pardict["wideangle"]

    covpath = pardict["covmatrix"]
    outputMC = pardict["outputMC"]


    combined = int(pardict["combined"])
    poles = list(pardict["poles"])
    ell = list(map(int, poles))
    
    deg = list(pardict["degrees"])
    degrees = list(map(int, deg))
    kmin = float(pardict["kmin"])
    kmax = float(pardict["kmax"])
    json = int(pardict["json"])
    convolved = int(pardict["convolve"])

    smooth = False
    print('smooth: ',smooth)



    if json:
        r = ConvolvedFFTPower.load(inputpk)
        poles = r.poles
        shot = poles.attrs['shotnoise']

        P0dat = poles['power_0'].real-shot
        P2dat = poles['power_2'].real
        P4dat = poles['power_4'].real
        kdat = poles['k']

    else:
        r = np.loadtxt(inputpk)
        kdat = r[:,0]
        P0dat = r[:,1]
        P2dat = r[:,2]
        P4dat = r[:,3]

    valid = (kdat>kmin) & (kdat<kmax)
    kobs = kdat[valid]
    ksize = kobs.size
    P0dat = P0dat[valid]
    P2dat = P2dat[valid]
    P4dat = P4dat[valid]

    if 4 in ell:
            Pkdata = np.concatenate([P0dat,P2dat,P4dat])

    else:
            Pkdata = np.concatenate([P0dat,P2dat])

    if json and combined:
        r1 = ConvolvedFFTPower.load(inputpk)
        poles1 = r1.poles
        shot1 = poles1.attrs['shotnoise']
        P0dat1 = poles1['power_0'].real-shot1
        P2dat1 = poles1['power_2'].real
        P4dat1 = poles1['power_4'].real
        kdat1 = poles1['k']
        valid1 = (kdat1>0.02) & (kdat1<0.23)
        kobs1 = kdat1[valid1]
        ksize = kobs1.size
        P0dat1 = P0dat1[valid1]
        P2dat1 = P2dat1[valid1]
        P4dat1 = P4dat1[valid1]



        r2 = ConvolvedFFTPower.load(inputpk2)
        poles2 = r2.poles
        shot2 = poles2.attrs['shotnoise']
        P0dat2 = poles2['power_0'].real-shot2
        P2dat2 = poles2['power_2'].real
        P4dat2 = poles2['power_4'].real
        kdat2 = poles2['k']
        valid2 = (kdat2>0.02) & (kdat2<0.23)
        kobs2 = kdat2[valid2]
        P0dat2 = P0dat2[valid2]
        P2dat2 = P2dat2[valid2]
        P4dat2 = P4dat2[valid2]
        kobs = np.concatenate([kobs1,kobs2])
        Pkdata = np.concatenate([P0dat1,P2dat1,P4dat1,P0dat2,P2dat2,P4dat2])

    size = Pkdata.size
    print('size of kobs ',ksize)
    half = int(size/2)

    temp = np.loadtxt(linearpk)
    ktemp = temp[:,0]
    Plintemp = temp[:,1]

    if 4 in ell:
            cov = np.load(covpath)
            covinv = inv(cov)

    else:
            cov = np.load(covpath)
            cov = cov[0:2*ksize,0:2*ksize]
            covinv = inv(cov)



    Wfile = window
    Mfile = wideangle
    W = np.loadtxt(Wfile)
    M = np.loadtxt(Mfile)


    
    print('poles: ', ell,'redshift: ',redshift)

    #h=0.676
    #Om0 = 0.31
    #cosmo = cosmology.Cosmology(h=0.676,Omega0_b=0.022/h**2,n_s=0.97).match(Omega0_m=Om0)   #eBOSS cosmology
    #new_cosmo = cosmo.match(sigma8=0.8)
    cosmo = cosmology.Cosmology(h=h,Omega0_b=omb0/h**2,n_s=0.97).match(Omega0_m=Om0)  
    new_cosmo = cosmo.match(sigma8=sig8)



    Rtemp = np.loadtxt('/home/merz/workdir/emcee/eBOSS/pk_matter_power_spectrum_bao.dat')
    Plinfunc =  IUS(Rtemp[0],Rtemp[1])
    #Plinfunc = cosmology.LinearPower(new_cosmo, redshift=redshift, transfer='CLASS')
    Psmlinfunc = cosmology.LinearPower(new_cosmo, redshift=redshift, transfer='NoWiggleEisensteinHu')
    
    
    
    popt,pcov = curve_fit(Psmfitfunopt,ktemp,Plinfunc(ktemp))
    asm1 = popt[0]
    asm2= popt[1]
    asm3 = popt[2]
    asm4 = popt[3]
    asm5 = popt[4]

    Psmfitopt = Psmfitfunopt(ktemp,asm1,asm2,asm3,asm4,asm5)
    

    muobs = np.linspace(-1,1,100)
    sigpar = 8.
    sigperp = 3.

    if smooth:
            sigpar = 100.
            sigperp = 100.

    print(sigpar,sigperp)

    sigs = 4.0

    z = redshift
    Omv0 = 1-Om0
    Omz = Om0*(1+z)**3/(Om0*(1+z)**3 + .69)
    f = Omz**0.55

    print('calculated f: ', f)

    
    L0 = Legendre(0)
    L2 = Legendre(2)
    L4 = Legendre(4)


    start = np.array([2.0,1.00,1.00])

    if combined:
        start = np.array([2.0,1.0,1.0,1.0])


    pos0 = start + 1e-4*np.random.randn(8*start.size, start.size)
    nwalkers, ndim = pos0.shape

    
    kbb,km = prepare_poly_k(ell,convolved)
    solver = LLSQsolver(degrees,ell,cov,kbb)

    

    print('Running best fit....')
    result = op.minimize(chi2f,start,method='Powell')



    print(result)


    pbf = result.x
    mP = model(pbf,combined)
    Pm= mP.run(km)
    chi2bf = chi2f(pbf)




    np.savetxt(outputMC+'_bf_params.txt',[*pbf,chi2bf])


                                                
    if not combined:
        np.savetxt(outputMC+'_best_pk.txt',np.column_stack([kobs,Pm[0:ksize],Pm[ksize:2*ksize],Pm[2*kobs.size:3*kobs.size]]))

    else:
                np.savetxt(outputMC+'_best_pk.txt',np.column_stack([kobs1,Pm[0:ksize],Pm[ksize:2*ksize],Pm[2*ksize:3*ksize],kobs2,Pm[half:half+ksize],Pm[half+ksize:half+2*ksize],Pm[half+2*ksize:half+3*ksize]]),header='NGC k \t P0 \t P2')




    # Don't forget to clear it in case the file already exists
    filename = outputMC+'.h5'
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # Initialize the sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlh, pool=pool,backend=backend)
        #sampler = zeus.sampler(nwalkers,ndim,chi2f,pool=pool)
        sampler.run_mcmc(pos0, 5000, progress=True)

    reader = emcee.backends.HDFBackend(outputMC+'.h5')
    samples1 = reader.get_chain(flat=True,discard=200,thin=30)
    B1m = np.mean(samples1[:,0])
    aperm = np.mean(samples1[:,1])
    aparm = np.mean(samples1[:,2])
    #f1m = np.mean(samples1[:,3])
    #B2m = np.mean(samples1[:,3])
    #f2m = np.mean(samples1[:,5])
    paramMC = [B1m,aperm,aparm]

    chi2MC = chi2f(paramMC)
    print(chi2MC)
    np.savetxt(outputMC+'_MC_params.txt',[*paramMC,chi2MC])











