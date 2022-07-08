import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.optimize import curve_fit
import os
import copy
plt.rcParams['figure.figsize'] = (9, 6.5)
plt.rcParams['font.size'] = 12

class Simulation(object):
    
    def __init__(self, AorM):
        '''
        The constructor of the Simulation class for solving Poisson's equation.
        The constructor calls all the appropriate user prompts and initialises the 
        system (electrostatic potential phi) accordingly, saving it as an attribute.        
        '''
        
        self.AorM = AorM
        if(self.AorM == 'a'):
            self.alg, self.EorM = self.alg_prompt()              
            self.N = self.size_prompt()    
            self.tol = self.tol_prompt()
        elif(self.AorM == 'm'):
            self.alg = 's'             
            self.N = 100  
            self.tol = 0.001   
            self.EorM ='e'       
        
        if(self.EorM == 'e'):          
            self.rho = np.zeros((self.N, self.N, self.N))   
            mid = int(self.N/2)
            self.rho[mid, mid, mid] = 1       # central charge 
            #self.rho[mid+10, mid, mid] = -1       # second opposite charge 
        elif(self.EorM == 'm'):  
            self.rho = np.zeros((self.N, self.N, self.N))
            mid = int(self.N/2)
            self.rho[mid, mid, :] = 1       # central wire (vertical)
            
        self.phi = np.zeros((self.N, self.N, self.N))
        self.dt = 1 
        self.dx = 1 
        if(self.alg == 'j'):
            print("\nJacobi's algorithm:")
        elif(self.alg == 'g'):
            print("\nGauss-Seidel algorithm:")    
        elif(self.alg == 's'):
            print("\nSuccessive Over-Relaxation method:")  
        
    def boundaries(self, array):
        '''
        A function to set the boundaries of the input array to 0, as required by the
        Dirichlet condition for solving BVPs.
        ''' 
        array[0,:,:] = 0
        array[:,0,:] = 0
        array[:,:,0] = 0        
        array[self.N-1,:,:] = 0
        array[:,self.N-1,:] = 0
        array[:,:,self.N-1] = 0
        
        return array
    
    def jacobi_update(self):
        '''
        Discretised version of the update rule for the Jacobi algorithm in 3D.
        '''         
        self.newPhi = (1/6)*(self.boundaries(np.roll(self.phi,-1,axis=0)) + self.boundaries(np.roll(self.phi,+1,axis=0)) + self.boundaries(np.roll(self.phi,-1,axis=1)) + 
                              self.boundaries(np.roll(self.phi,+1,axis=1)) + self.boundaries(np.roll(self.phi,-1,axis=2)) + self.boundaries(np.roll(self.phi,+1,axis=2)) + self.rho)
    
    def GS_update(self):
        '''
        Discretised version of the update rule for the Gauss-Seidel algorithm in 3D, 
        which uses the most recent value for each timestep, as opposed to updating the 
        whole lattice from the old values.
        ''' 
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                for k in range(1, self.N-1):
                    self.phi[i,j,k] = (1/6)*(self.phi[i-1,j,k] + self.phi[i+1,j,k] + self.phi[i,j-1,k] 
                                             + self.phi[i,j+1,k] + self.phi[i,j,k-1] + self.phi[i,j,k+1] + self.rho[i,j,k])
    
    def SOR(self):
        '''
        Successive Over-Relaxation method with GS algorithm in 2D.
        '''
        
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                
                self.phi[i,j] = (1-self.omega)*(self.phi[i,j]) + (self.omega)*(1/4)*(
                    self.phi[i-1,j] + self.phi[i+1,j] + self.phi[i,j-1] + self.phi[i,j+1] + self.rho[i,j,int(self.N/2)])  # fixed k
                
        
    def size_prompt(self):
        '''
         A function to prompt for the system size N, catching any possible errors on the way. 
        '''    
        
        N = input("Type in the phi size N: ")
        wrong = True
        while(wrong):
            try:                         
                N = int(N)
                if(N>0):      # ensure N is an integer larger than 0
                    wrong = False
                else:
                    N = int("zero")  # ValueError
            except ValueError:
                print("Please enter a valid integer larger than 0.")
                N = input("Type in the phi size N: ")
        
        return N
    
    def tol_prompt(self):    
        '''
        A function to prompt for the desired tolerance, catching any possible errors on the way.     
        '''    
        
        p = input("Type in the desired tolerance: ")
        wrong = True
        while(wrong):
            try:                         
                p = float(p)
                if(p>0):    # ensure p is a float larger than 0
                    wrong = False
                else:
                    p = int("zero")  # ValueError
            except ValueError:
                print("Please enter a valid float larger than 0.")
                p = input("Type in the desired tolerance: ")
        
        return p   
    
    def alg_prompt(self):
        '''
        A function to prompt for the desired algorithm to solve Poisson's equation.
        '''
        
        EorM = input("Type in the desired problem to solve\n('e' for Electric or 'm' for Magnetic): ")
        while(EorM!='e' and EorM!='m'):
            print("Invalid entry. Please try again.")
            EorM = input("Type in the desired problem to solve\n('e' for Electric or 'm' for Magnetic): ")
    
        algorithm = input("Type in the desired algorithm to be used\n('j' for Jacobi's or 'g' for Gauss-Seidel): ")
        while(algorithm!='j' and algorithm!='g' and algorithm!='s'):
            print("Invalid entry for algorithm. Please try again.")
            algorithm = input("Type in the desired algorithm to be used\n('j' for Jacobi's or 'g' for Gauss-Seidel): ")
        
        return algorithm, EorM  
 
    def filewrite(self, title, data):
        '''
        A function to write the data out to a file.
        '''             
        if not os.path.isdir('Datafiles'):
            os.mkdir('Datafiles')
        outdir = 'Datafiles'
        
        f=open(f'{outdir}/{title}.txt','w')      
        
        #f.write('%s\n\n'%(title))            
        for i in range(len(data[0])):
            for j in range(len(data)):
                f.write('%lf '%(data[j][i]))
            f.write('\n')
        f.close()
            
    def animate(self):
        '''
        A function for running the simulation and animating on the run.  
        '''   
        N = self.N
#         fig, ax = plt.subplots()
#         vmax = 0.25
#         ticks = np.linspace(0,vmax,6)
#         print(ticks)
#         vis = ax.pcolormesh(self.phi[:,:,int(N/2)], vmin=0, vmax=vmax, cmap = plt.cm.get_cmap("viridis"))  # plot the initial state
#         fig.colorbar(vis, label="Phi", ticks=ticks, ax=ax)   # colorbar to indicate the spins
        error = 1000
        
        while(error>self.tol):
            
            if(self.alg == 'j'):
                self.jacobi_update()
                error = np.sum(np.abs(self.newPhi - self.phi))
                print(error)                
                #self.phi = copy.deepcopy(self.newPhi)
                self.phi = self.newPhi
                
            elif(self.alg == 'g'):
                old_phi = np.copy(self.phi)
                self.GS_update()
                error = np.sum(np.abs(self.phi - old_phi))
                print(error)     
                
        if(self.EorM == 'e'):
            self.phi_plot()
            self.E_field_plot()
        elif(self.EorM == 'm'):
            self.phi_plot()
            self.B_field_plot()
            
    def measurements(self):
        '''
        This function runs the SOR method for a range of omegas (1 to 2), 
        saving the total iterations needed for convergence and plotting the 
        result, to determine the optimum omega.
        '''                    
                          
        omegas = np.arange(1,2,0.01)
        sweeps = []
                
        for omega in omegas:            
            self.omega = omega 
            self.phi = np.zeros((self.N, self.N))
            error = 1000
            sweep = 0
            
            while(error>self.tol):
                self.oldPhi = np.copy(self.phi)
                self.SOR()
                error = np.sum(np.abs(self.phi - self.oldPhi))
                print(error)
                sweep+=1
                
                if sweep > 10000:   # prevents infinite loops                    
                    break
                
            sweeps.append(sweep)
        
        self.omegas_plot(omegas, sweeps)
            
    def omegas_plot(self, omegas, sweeps):
        '''
        A function to plot the total sweeps needed for convergence over the
        relaxation parameter omega, to find the optimal value.
        '''    
            
        plt.plot(omegas, sweeps)
        plt.title(r"Total sweeps needed over the relaxation parameter $\omega$")
        plt.ylabel("Total sweeps for convergence")
        plt.xlabel(r"$\omega$")
        self.filewrite(f"Omegas", [omegas, sweeps])
        plt.savefig(f'Datafiles/Omegas.png')
        plt.clf()
        
    def phi_plot(self):
        '''
        This function plots the contour of the electric or magnetic potentials
        in the middle slice along z.
        '''
        
        N = self.N
        fig, ax = plt.subplots()
        vmin = np.min(self.phi[:,:,int(N/2)])
        vmax = np.max(self.phi[:,:,int(N/2)])
        ticks = np.linspace(vmin,vmax,6)
        
        if(self.EorM == 'e'):
            potential = 'Electric'
            label= r"$\Phi$"
        elif(self.EorM == 'm'):
            potential = 'Magnetic'
            label= r"$A_{z}$"
            
        ax.set_title(f"Contour plot of the {potential} potential\nfor a {N}x{N} square lattice")
#         ax.set_xticks(np.arange(0, N+1, 1))
#         ax.set_yticks(np.arange(0, N+1, 1))
        vis = ax.pcolormesh(self.phi[:,:,int(N/2)], vmin=vmin, vmax=vmax, cmap = plt.cm.get_cmap("viridis"))  # plot the new state
#         plt.draw()
        fig.colorbar(vis, label=label, ticks=ticks, ax=ax)

        plt.xlabel('x')
        plt.ylabel('y')
        
        data = []
        z = int(N/2)
        for x in range(N):
            for y in range(N):
                data.append([x, y, self.phi[x,y,z]])
        
        data = np.array(data)    
        self.filewrite(f"{potential}Potential", data.T ) 
        plt.savefig(f'Datafiles/{potential}Potential.png')
        if(self.AorM=="a"):  
            plt.show()
        plt.close()
        
    def E_field_plot(self):
        '''
        This function calculates the electric field (negative gradient of the
        electric potential) and plots the normalised vector field in the middle
        slice along z.
        '''
        N= self.N
        fig, ax = plt.subplots()
        grad = np.gradient(np.array(self.phi))
        dx = np.array(grad[0])
        dy = np.array(grad[1])
        dz = np.array(grad[2])
        
        vectors = []
        rs = []
        magns = []
        phis = []
        mid = int(N/2)
        z = mid        
        for x in range(N):
            for y in range(N):
#                 charges = np.where(self.rho != 0)
#                 print(charges)
#                 for c in range(len(charges[0])):                
#                     if not(x==charges[0][c] and y==charges[0][c]):
                if not(x==mid and y==mid):
                    E_x = -dx[x,y,z]
                    E_y = -dy[x,y,z]
                    E_z = -dz[x,y,z]
                    E_magn = np.sqrt(E_x**2 + E_y**2 + E_z**2)
                    vector = [x, y, E_x/E_magn, E_y/E_magn]
                    vectors.append(vector)
                    r = np.sqrt( (x-mid)**2 + (y-mid)**2 + (z-mid)**2)
                    rs.append(r)
                    magns.append(E_magn)
                    phis.append(self.phi[x,y,z])
        
        vectors = np.array(vectors)  
        rs = np.array(rs) 
        magns = np.array(magns) 
        phis = np.array(phis)
              
        plt.quiver(vectors[:,0], vectors[:,1], vectors[:,2], vectors[:,3]) # scale=7, scale_units='inches'
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Electric (vector) field on the middle slice along z')
        self.filewrite(f"ElectricVectorField", vectors.T)
        plt.savefig(f'Datafiles/ElectricVectorField.png')
        if(self.AorM=="a"):  
            plt.show()
        plt.close()
        
        plt.cla()
        data = tuple(zip(rs[np.where(rs!=0)], phis[np.where(rs!=0)]))
        data = sorted(data, key = lambda t: t[0])
        data = np.array(data)
        self.filewrite(f"Phi_r", [data[:,0], data[:,1]])
        data = np.log(data)
        l = len(data)
        
        #fit = curve_fit(self.powerLaw, data[2:10,0], data[2:10,1])
        fit = curve_fit(self.line, data[5:100,0], data[5:100,1])
        print(fit[0])
        print(f"The best fit power is {fit[0][1]}")
        plt.scatter(data[:,0], data[:,1], s=3, label = 'Data')
        plt.plot(data[:,0], self.line(data[:,0], fit[0][0], fit[0][1]), label = f'Fit (m = {fit[0][1]:.2f})', color='r')
        plt.title('Electric scalar potential over distance from the central charge')
        plt.xlabel('ln(r)')
        plt.ylabel(r'ln($\Phi$)')
        plt.legend()
        #plt.xscale('log')
        #plt.yscale('log') 
        plt.savefig(f'Datafiles/Phi_r.png')
        if(self.AorM=="a"):  
            plt.show()
        plt.close()
        
        plt.cla()
        data = tuple( zip( rs[np.where(rs!=0)], magns[np.where(rs!=0)] ))
        data = sorted(data, key = lambda t: t[0])
        data = np.array(data)
        self.filewrite(f"E_r", [data[:,0], data[:,1]])
        data = np.log(data)
        
        #fit = curve_fit(self.powerLaw, data[2:30,0], data[2:30,1])        
        fit = curve_fit(self.line, data[:100,0], data[:100,1])
        print(fit[0])
        print(f"The best fit power is {fit[0][1]}")
        
        #plt.plot(data[:,0], self.powerLaw(data[:,0], fit[0][0], fit[0][1]), label = 'Fit', color='r')       
        plt.scatter(data[:,0], data[:,1], s=3, label = 'Data')
        plt.plot(data[:,0], self.line(data[:,0], fit[0][0], fit[0][1]), label = f'Fit (m = {fit[0][1]:.2f})', color='r')
        plt.title('Electric field magnitude over distance from the central charge')
        plt.xlabel('ln(r)')
        plt.ylabel('ln(|E|)')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
        plt.savefig(f'Datafiles/E_r.png')
        if(self.AorM=="a"):  
            plt.show()
        plt.close()        
    
    def B_field_plot(self):
        '''
        This function calculates the magnetic field (curl of the magnetic 
        vector potential) and plots the normalised vector field in the middle
        slice along z.
        '''
        N= self.N
        fig, ax = plt.subplots()
        grad = np.gradient(np.array(self.phi))
        dx = np.array(grad[0])
        dy = np.array(grad[1])
        dz = np.array(grad[2])
        
        vectors = []
        rs = []
        magns = []
        phis = []
        mid = int(N/2)
        z = mid        
        for x in range(N):
            for y in range(N):
#                 charges = np.where(self.rho != 0)
#                 print(charges)
#                 for c in range(len(charges[0])):                
#                     if not(x==charges[0][c] and y==charges[0][c]):
                if not(x==mid and y==mid):
                    B_x = dy[x,y,z]
                    B_y = -dx[x,y,z]
                    B_magn = np.sqrt(B_x**2 + B_y**2) # + B_z**2)
                    vector = [x, y, B_x/B_magn, B_y/B_magn]
                    vectors.append(vector)
                    r = np.sqrt( (x-mid)**2 + (y-mid)**2 + (z-mid)**2)
                    rs.append(r)
                    magns.append(B_magn)
                    phis.append(self.phi[x,y,z])
        
        vectors = np.array(vectors)  
        rs = np.array(rs) 
        magns = np.array(magns) 
        phis = np.array(phis)
              
        plt.quiver(vectors[:,0], vectors[:,1], vectors[:,2], vectors[:,3]) # scale=7, scale_units='inches'
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Magnetic (vector) field on the middle slice along z')
        self.filewrite(f"MagneticVectorField", vectors.T)
        plt.savefig(f'Datafiles/MagneticVectorField.png')
        if(self.AorM=="a"):  
            plt.show()
        plt.close()
        
        plt.cla()
        data = tuple(zip(rs[np.where(rs!=0)], phis[np.where(rs!=0)]))
        data = sorted(data, key = lambda t: t[0])
        data = np.array(data)
        self.filewrite(f"Az_r", [data[:,0], data[:,1]])
        data[:,0] = np.log(data[:,0])
        l = len(data)
        
        #fit = curve_fit(self.powerLaw, data[10:int(l/3),0], data[10:int(l/3),1])
        fit = curve_fit(self.line, data[:100,0], data[:100,1])
        print(fit[0])
        print(f"The best fit power is {fit[0][1]}")
        plt.scatter(data[:,0], data[:,1], s=3, label = 'Data')
        plt.plot(data[:,0], self.line(data[:,0], fit[0][0], fit[0][1]), label = f'Fit (m = {fit[0][1]:.2f})', color='r')
        plt.title('Magnetic potential over distance from the central wire')
        plt.xlabel('ln(r)')
        plt.ylabel(r'$A_{z}$')
        plt.ylim(0)
        plt.legend()
        #plt.xscale('log')
        #plt.yscale('log')        
        plt.savefig(f'Datafiles/Az_r.png')
        if(self.AorM=="a"):  
            plt.show()
        plt.close()
        
        plt.cla()
        data = tuple( zip( rs[np.where(rs!=0)], magns[np.where(rs!=0)] ))
        data = sorted(data, key = lambda t: t[0])
        data = np.array(data)
        self.filewrite(f"B_r", [data[:,0], data[:,1]])
        data = np.log(data)
        
        fit = curve_fit(self.line, data[:100,0], data[:100,1])
        print(fit[0])
        print(f"The best fit power is {fit[0][1]}")
        plt.scatter(data[:,0], data[:,1], s=3, label = 'Data')
        plt.plot(data[:,0], self.line(data[:,0], fit[0][0], fit[0][1]), label = f'Fit (m = {fit[0][1]:.2f})', color='r')
        plt.title('Magnetic field magnitude over distance from the central wire')
        plt.xlabel('ln(r)')
        plt.ylabel('ln(|B|)')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
        plt.savefig(f'Datafiles/B_r.png')
        if(self.AorM=="a"):  
            plt.show()
        plt.close()        
    
    '''
    The following functions were used for fitting the data to get the relationship
    between the Electric and Magnetic fields over distance from the charge/wire.
    '''        
    def powerLaw(self, r, C, p):
        
        return C * r**p   
    
    def line(self, r, C, m):
        
        return m*r + C
    
def main():
    
    '''
    The main function for controlling the flow of the simulation. Allows the user to choose
    between animation and measurements, and runs the code accordingly.
    '''             
    AorM = input("Please type 'a' for Animation or 'm' for Measurements: ")
    
    if(AorM=='a'):
        
        print("Animation:")
        poissons = Simulation(AorM)
        poissons.animate()
        
    elif(AorM=='m'):
        
        print("Measurements:")
        poissons = Simulation(AorM)
        poissons.measurements()
        
    else:
        print("Invalid entry. Please try again.")
        main()
        
if __name__ == "__main__":
    main()
     