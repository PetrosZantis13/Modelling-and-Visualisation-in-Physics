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
        The constructor of the Simulation class for solving the Cahn-Hilliard equation.
        The constructor calls all the appropriate user prompts and initialises the 
        system (order parameter phi) accordingly, saving it as an attribute.        
        '''
        self.AorM = AorM
        if(self.AorM=='a'):
            self.N = self.size_prompt()          
            self.phi0 = self.init_prompt()
        elif(self.AorM=='m'):
            self.N = 100
            self.phi0 = 0
            
        self.phi = np.ones((self.N, self.N)) * self.phi0       # initial condition
        self.phi += np.random.uniform(-0.1, +0.1, (self.N,self.N))      # small random noise
        self.newPhi = copy.deepcopy(self.phi)
        self.dt = 1 
        self.dx = 1.5 
        self.M = 0.1  
        self.K = 0.1
        self.a= self.b= 0.1   
      
    def update_mu(self):
        '''
        Discretised equation for the chemical potential (μ) using centered differences to
        discretise the Laplacian, and also a term (K) to include the surface tension of 
        an oil-water interface.  
        '''         
        mu1 = -self.a*self.phi + self.b*(self.phi**3)
        mu2 = -(self.K/self.dx**2)*(np.roll(self.phi,-1,axis=0) + np.roll(self.phi,+1,axis=0) + 
                                    np.roll(self.phi,-1,axis=1) + np.roll(self.phi,+1,axis=1) - 4*self.phi)
        
        self.mu = mu1+mu2
    
    def update_phi(self):
        '''
        Discretised version of the update rule for the Cahn-Hilliard equation.
        '''         
        self.newPhi += (self.M*self.dt/(self.dx**2))*(
            np.roll(self.mu,-1,axis=0) + np.roll(self.mu,+1,axis=0) + np.roll(self.mu,-1,axis=1) + 
            np.roll(self.mu,+1,axis=1) - 4*self.mu)
    
    def free_energy(self):
        '''
        Discretised free energy density calculation based on the order parameter, phi.
        '''             
        f = -(self.a/2)*(self.phi**2) + (self.a/4)*(self.phi**4)+ (self.K/(8*(self.dx**2)))*( 
            (np.roll(self.phi,-1,axis=0) - self.phi)**2 + (np.roll(self.phi,-1,axis=1) - self.phi)**2)
        return f
     
    def size_prompt(self):
        '''
         A function to prompt for the system size N, catching any possible errors on the way. 
        '''       
         
        N = input("Type in the system size N: ")
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
    
    def init_prompt(self):    
        '''
        A function to prompt for the initial condition, catching any possible errors on the way.     
        '''    
        
        p = input("Type in the initial condition phi_0: ")
        wrong = True
        while(wrong):
            try:                         
                p = float(p)
                if(p>=-1 and p<=1):    # ensure p is a float between -1 and 1
                    wrong = False
                else:
                    p = int("zero")  # ValueError
            except ValueError:
                print("Please enter a valid float between -1 and 1.")
                p = input("Type in the initial condition phi_0: ")
        
        return p   
    
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
        fig, ax = plt.subplots()
        vis = ax.pcolormesh(self.phi, vmin=-1, vmax=1, cmap = plt.cm.get_cmap("viridis"))  # plot the initial state
        fig.colorbar(vis, label= r"$\Phi$", ax=ax) 
        
        free_energy = []
        while(plt.fignum_exists(fig.number)):
            
            for sweep in range(100):
                self.update_mu()
                self.update_phi()                
                #print(self.phi)
                #print(self.newPhi)                
                self.phi = copy.deepcopy(self.newPhi)
                #print(np.min(self.phi) , np.max(self.phi))    

            free_energy.append( np.sum(self.free_energy() ))
            plt.cla()
            ax.set_title(f"Cahn-Hilliard equation for a {N} x {N} square lattice")
            vis = ax.pcolormesh(self.phi, vmin=-1, vmax=1, cmap = plt.cm.get_cmap("viridis"))  # plot the new state
    #         plt.draw()
            plt.pause(0.0001)
            
        self.plot_freeEnergy(free_energy)
    
    def measurements(self):
        '''
        A function for running simulations with different initial conditions (phi0)
        and taking measurements of the free energy density without animating.  
        '''  
        
        for phi0 in [0, 0.5]:
            
            self.phi0 = phi0
            free_energy = []
            
            self.phi = np.ones((self.N, self.N)) * self.phi0     # initial condition
            self.phi += np.random.uniform(-0.1, +0.1, (self.N,self.N))    # small random noise
            self.newPhi = copy.deepcopy(self.phi)
            
            for total in range(5001):    
                for sweep in range(100):
                    self.update_mu()
                    self.update_phi()                
                    #print(self.phi)
                    #print(self.newPhi)                
                    self.phi = copy.deepcopy(self.newPhi)
                    #print(np.min(self.phi) , np.max(self.phi))    
    
                free_energy.append( np.sum(self.free_energy() ))
            
            self.plot_freeEnergy(free_energy)
        
    def plot_freeEnergy(self, free_energy):
        '''
        A function to plot the free energy density of the system over the 
        number of sweeps (in hundreds). 
        '''    
            
        plt.plot(free_energy)
        plt.title(f"Free energy density of the system over number of sweeps\nwith phi_0 = {self.phi0}")
        plt.ylabel("Free energy density")
        plt.xlabel("Sweeps (in hundreds)")
        self.filewrite(f"FreeEnergy(phi0={self.phi0})", [free_energy])
        plt.savefig(f'Datafiles/FreeEnergy(phi0={self.phi0}).png')
        if(self.AorM=="a"):  
            plt.show()
        plt.clf()
        
def main():
    
    '''
    The main function for controlling the flow of the simulation. Allows the user to choose
    between animation and measurements, and runs the code accordingly.
    '''             
    AorM = input("Please type 'a' for Animation or 'm' for Measurements: ")
    
    if(AorM=='a'):
        
        print("Animation:")
        CH = Simulation(AorM)
        CH.animate()
        
    elif(AorM=='m'):
        
        print("Measurements:")
        CH = Simulation(AorM)
        CH.measurements()
        
    else:
        print("Invalid entry. Please try again.")
        main()
        
if __name__ == "__main__":
    main()
    