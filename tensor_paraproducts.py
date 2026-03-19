

import numpy as np
import math

class haar_paraproduct:

    def __init__(self,data,orig):

        """
        orig = smooth composition of Holder data
        data = holder data
        """

        # Define Holder data
        self.f = data
        self.Af = orig
        
        return


    def haar(self,j,npts):
    
        # create haar function at scale j
        dx = int(npts/(2**j))
        dxd2 = int(dx/2)
        left = np.ones(dxd2).reshape(1,dxd2)
        left = left.reshape(1,dxd2)
        right = np.ones(dxd2)*-1
        right = right.reshape(1,dxd2)
        haar = np.squeeze(np.concatenate((left,right),axis=1))
        nhaar = haar/np.linalg.norm(haar)
        
        # Generate basis for subspace at scale j using location parameter
        veclist = []
        for k in range(0,npts,dx):
            vec = np.zeros((npts))
            vec[k:k + dx] = nhaar
            veclist.append(vec)
        veclist = np.array(veclist)
    
        return veclist


    def scl(self,j,npts):
    
        # Create scaling function at scale j
        dx = int(npts/(2**j))
        scl = np.ones(dx)
        nscl = scl/np.linalg.norm(scl)
        
        # Generate basis for subspace at scale j using location parameter
        veclist = []
        for k in range(0,npts,dx):
            vec = np.zeros((npts))
            vec[k:k + dx] = nscl
            veclist.append(vec)
        veclist = np.array(veclist)
    
        return veclist
    

    def genconvs(self,jx,jy):

        NY = self.f.shape[0]
        NX = self.f.shape[1]
        
        Wx = self.haar(jx,NX)
        Wy = self.haar(jy,NY)
        Vx = self.scl(jx,NX)
        Vy = self.scl(jy,NY)
        
        nyloc = Vy.shape[0]
        nxloc = Vx.shape[0]
        
        convopWxWy = np.ones((NY,NX))
        convopWxVy = np.ones((NY,NX))
        convopVxWy = np.ones((NY,NX))
        convopVxVy = np.ones((NY,NX)) 
        
        
        for kx in range(0,nxloc):
            for ky in range(0,nyloc):
                
                tens = np.kron(Wx[kx,:].reshape(1,NX),Wy[ky,:].reshape(NY,1))
                rowidx, colidx = np.nonzero(tens)
                coef = np.sum(self.f[rowidx,colidx] * tens[rowidx,colidx])
                suppsize = len(rowidx)*len(colidx)
                #print(f"support size {suppsize}")
                convopWxWy[rowidx,colidx] = coef/suppsize
        
        
                tens = np.kron(Wx[kx,:].reshape(1,NX),Vy[ky,:].reshape(NY,1))
                rowidx, colidx = np.nonzero(tens)
                coef = np.sum(self.f[rowidx,colidx] * tens[rowidx,colidx])
                convopWxVy[rowidx,colidx] = coef/suppsize
        
        
                tens = np.kron(Vx[kx,:].reshape(1,NX),Wy[ky,:].reshape(NY,1))
                rowidx, colidx = np.nonzero(tens)
                coef = np.sum(self.f[rowidx,colidx] * tens[rowidx,colidx])
                convopVxWy[rowidx,colidx] = coef/suppsize
    
        
                tens = np.kron(Vx[kx,:].reshape(1,NX),Vy[ky,:].reshape(NY,1))
                rowidx, colidx = np.nonzero(tens)
                coef = np.sum(self.f[rowidx,colidx] * tens[rowidx,colidx])
                convopVxVy[rowidx,colidx] = coef/suppsize
                
    
        return convopWxWy, convopWxVy, convopVxWy, convopVxVy


    def para_approx(self,mNjx,mNjy):


        # Specify number of points
        NY = self.f.shape[0]
        NX = self.f.shape[1]
        App = np.zeros((NY,NX))
        if self.kernel == 'schrod':
            App = np.zeros((NY,NX),dtype=complex)

        # Starting number of scales
        smNjx = 3
        smNjy = 3

        for jdx in range(smNjx,mNjx):
            for jdy in range(smNjy,mNjy):
        
                WxWy, WxVy, VxWy, VxVy = self.genconvs(jdx,jdy)


                # Nonlinearity (Heat Kernel)
                if self.kernel == 'heat':
                    escl = 0.2
                    Ap = -escl*np.exp(-escl*VxVy)
                    Adp = escl*escl*np.exp(-escl*VxVy)

                # Nonlinearity (Potential)
                if self.kernel == 'pot':
                    Ap = -VxVy**(-2)
                    Adp = 2*VxVy**(-3)

                # Nonlinearity (Parametrix)
                if self.kernel == 'parametrix':
                    Ap = -0.5*VxVy**(-1.5)
                    Adp = 0.75*VxVy**(-2.5)

                if self.kernel == 'log':

                    # Constant Coefficient
                    #Ap = VxVy**(-1)
                    #Adp = VxVy**(-2)

                    # NOTE: if the degree of the polynomial is too high you could get infinite values (well behaved for up to degree 25)

                    # Variable coefficient 
                    termone = -(self.deg + 1)*self.coeffn*self.eps**(-1)*(VxVy**(-(self.deg+2)))
                    termtwo = (self.coeffn*(VxVy**(-(self.deg+1)))*self.eps**(-1)) - (self.coeffzro*self.eps**(-1))
                    Ap = termone/ (termtwo + 1e-10)
                    Adp = ((self.eps - 1)*(self.deg**2 + 2*self.deg + 2)*VxVy**(-2)) + (self.deg*VxVy**(-2)) - (self.coeffzro*(self.coeffn**(-1))*(VxVy**(self.deg + 1)))


                if self.kernel == 'schrod':
                    #Ap = np.exp((1j*(VxVy**2))/self.t)*(((2*1j)/self.t)*(VxVy))
                    #Adp = np.exp((1j*(VxVy**2))/self.t)*(-4*(VxVy**2)/(self.t**2))

                    Ap = 1j*np.exp(1j*VxVy) 
                    Adp = -np.exp(1j*VxVy) 

                # Build approximation
                frst = Ap * WxWy
                scnd = Adp * WxVy * VxWy
                App = App + frst + scnd

        self.App = App


        return App
        

    def twave_approx(self,data,mNjx,mNjy):

        NY = data.shape[0]
        NX = data.shape[1]
        Afrecon = np.zeros((NY,NX))
        self.coefs = []
        self.norms = []
        self.supp = []
   
        for jxi in range(0,mNjx):
            for jyi in range(0,mNjy):
        
                hxv = self.haar(jxi,NX) 
                hyv = self.haar(jyi,NY)
                nxloc = hxv.shape[0]
                nyloc = hyv.shape[0]
                
                for kx in range(nxloc):
                    for ky in range(nyloc):                    
                        
                        wvtens = np.kron(hxv[kx,:].reshape(1,NX),hyv[ky,:].reshape(NY,1))
                        acoef = np.abs(np.sum(data * wvtens))
                        #dnm = 2**(-(jxi + jyi))*(self.alf + 0.5)
                        dnm = (2**(-(jxi + jyi)))**(self.alf + 0.5)
                        self.norms.append(acoef/dnm)
                        self.coefs.append(acoef)
                        Afrecon += np.sum(data * wvtens) * wvtens
                        self.supp.append(dnm)

        # Scaling Function
        char = np.ones((NY,NX))
        Afrecon += (np.mean(data) * char)
        

        return Afrecon, self.coefs, self.norms


    def compute_holder_norm(self,pararesid,appresid,odata):

        # Need to use finest scales available
        mXs = int(math.log2(pararesid.shape[0])-1) 
        mYs = int(math.log2(pararesid.shape[1])-1)
        #mXs = 2
        #mYs = 2

        # Compute approximation
        para_resid_wave, para_resid_coefs, para_resid_norms = self.twave_approx(pararesid,mXs,mYs)
        app_resid_wave, app_resid_coefs, app_resid_norms = self.twave_approx(appresid,mXs,mYs)
        f_resid, f_coefs, f_norms = self.twave_approx(odata,mXs,mYs)

        # Compute decomposition
        self.decomp_results = {'paracoeffs': [para_resid_wave, para_resid_coefs, para_resid_norms],'twavecoeffs':[app_resid_wave, app_resid_coefs, app_resid_norms],'datacoeffs':[f_resid, f_coefs, f_norms]}

        return

        

        
        
     