import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.optimize import curve_fit
import os
plt.rcParams['figure.figsize'] = (9, 6.5)
plt.rcParams['font.size'] = 12

class Simulation(object):
    
    def __init__(self, AorM):
        '''
        The constructor of the Simulation class takes as argument the AorM variable
        which indicates if this will be an animation or measurements simulation.
        Next, the constructor calls all the appropriate user prompts and 
        initialises the lattice accordingly, saving it as an attribute.        
        ''' 
        name = self.dyn_prompt()
        print(f"{name} simulation")
        self.name = name
        self.N = self.size_prompt()
        lattice = np.zeros((self.N,self.N))
        self.init = self.init_prompt(AorM)  
        
        if(self.init == 'r'):
            print("Random initial state\n")
            lattice = self.randomise(lattice)
            
        elif(self.init == 'o'):
            print("Initial state for oscillators\n")
            quarter = int(self.N/4)
            lattice[quarter,quarter] = lattice[quarter,quarter+1] = lattice[quarter,quarter-1] = 1
            
            lattice[quarter,3*quarter] = lattice[quarter,3*quarter +1] = 1            
            lattice[quarter,3*quarter -1] = lattice[quarter-1,3*quarter-1] = 1            
            lattice[quarter-1,3*quarter] = lattice[quarter-1,3*quarter-2] = 1            
            
            lattice[3*quarter,quarter-1] = lattice[3*quarter+1,quarter] = 1            
            lattice[3*quarter+1,quarter-1] = lattice[3*quarter-1,quarter+2] = 1            
            lattice[3*quarter-2,quarter+1] = lattice[3*quarter-2,quarter+2] = 1            
            
            lattice[3*quarter,3*quarter] = lattice[3*quarter +1,3*quarter] = lattice[3*quarter -1,3*quarter] = 1 
                                         
        elif(self.init == 'g'):
            print("Initial state for moving spaceship (glider)\n")
            centre = int(self.N/2)
            lattice[centre,centre] = lattice[centre,centre+1] = lattice[centre,centre-1] = 1
            lattice[centre+1,centre-1] = lattice[centre+2,centre] = 1
        
        elif(self.init == 's'):
            print("Initial state for still states\n")
            quarter = int(self.N/4)
            lattice[quarter,quarter] = lattice[quarter,quarter+1] = 1
            lattice[quarter+1,quarter] = lattice[quarter+1,quarter+1] = 1
            
            lattice[quarter,3*quarter] = lattice[quarter,3*quarter +1] = 1
            lattice[quarter+1,3*quarter -1] = lattice[quarter+1,3*quarter +2] = 1            
            lattice[quarter+2,3*quarter] = lattice[quarter+2,3*quarter+1] = 1
            
            lattice[3*quarter,quarter] = lattice[3*quarter+2,quarter] = lattice[3*quarter+1,quarter+2] = 1            
            lattice[3*quarter+1,quarter-1] = lattice[3*quarter+1,quarter+2] = 1            
            lattice[3*quarter-1,quarter+1] = lattice[3*quarter,quarter+2] = lattice[3*quarter+2,quarter+1] = 1            
            
            lattice[3*quarter,3*quarter] = lattice[3*quarter +2,3*quarter] = 1            
            lattice[3*quarter +1,3*quarter +1] = lattice[3*quarter +1,3*quarter -1] = 1 
             
        elif(self.init != 'p' and self.init != 'v' and self.init != 'i'):
            print("Random initial state")
            lattice = self.randomise(lattice)
            p1,p2,p3 = self.init
            print(f"p1 = {p1}, p2 = {p2}, p3 = {p3} \n")            
                   
        self.lattice = lattice
    
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
                N = input("Type in the system size N: ")
        
        return N
    
    def dyn_prompt(self):
        '''
        A function to prompt for the desired dynamics, catching any possible errors on the way.     
        '''    
            
        dynamics = input("Type in the desired simulation\n('g' for Game of Life or 's' for SIRS): ")
        while(dynamics!='g' and dynamics!='s'):
            print("Invalid entry for dynamics. Please try again.")
            dynamics = input("Type in the desired dynamics to be used\n('g' for Game of Life or 's' for SIRS): ")
        
        if(dynamics=='g'):
            dynamics = 'Game of Life'
        else:
            dynamics = 'SIRS'
        
        return dynamics 
    
    def init_prompt(self, AorM):
        '''
        A function to prompt for the initial conditions, catching any possible errors on the way.     
        '''
        
        if(self.name == 'Game of Life'):
            init = input("Type in the desired initial condition\n('r' for Random, 'o' for Oscillators, 'g' for moving Spaceship or 's' for Still states):")
            while(init!='r' and init!='o' and init!='g' and init!='s'):
                print("Invalid Initial condition. Please try again.")
                init = input("Type in the desired initial condition\n('r' for Random, 'o' for Oscillators, 'g' for moving Spaceship or 's' for Still states):")
        
        elif(self.name == 'SIRS' and AorM =='a'):
            p1 = self.prob_prompt("p1")
            p2 = self.prob_prompt("p2")
            p3 = self.prob_prompt("p3")
            
            init = p1,p2,p3
            
        elif(self.name == 'SIRS' and AorM =='m'):
            init = input("Type in the desired measurement\n('p' for Phase Diagram, 'v' for Variance Cut or 'i' for Immunity/Vaccination): ")
            while(init!='p' and init!='v' and init!='i' ):
                print("Invalid Initial condition. Please try again.")
                init = input("Type in the desired measurement\n('p' for Phase Diagram, 'v' for Variance Cut or 'i' for Immunity/Vaccination): ")
           
        return init    
    
    def prob_prompt(self, name):    
        '''
        A function to prompt for the probabilities, catching any possible errors on the way.     
        '''    
        
        p = input(f"Type in {name}: ")
        wrong = True
        while(wrong):
            try:                         
                p = float(p)
                if(p>=0 and p<=1):    # ensure p is a float between 0 and 1
                    wrong = False
                else:
                    p = int("zero")  # ValueError
            except ValueError:
                print("Please enter a valid float between 0 and 1.")
                p = input(f"Type in {name}: ")
        
        return p         
    
    def randomise(self, state):
        '''
        A function to randomise the given state (according to the simulation dynamics)
        '''
        
        if(self.name =='Game of Life'):
            spins = [0,1]
            
        elif(self.name =='SIRS'):
            spins = [-1,0,1]   
        
        for i in range(self.N):
            for j in range(self.N):
                r = np.random.randint(spins[0],spins[-1]+1)
                state[i,j] = r
                    
        return state      

    def neighbours(self, state, i, j, nns):
        """
        This function checks how many nearest neighbours are alive/infected
        (4 or 8 is indicated by the nns variable)
        """
        
        N = self.N        
        east = state[i,(j+1)%N] 
        west =  state[i,(j-1)%N] 
        north = state[(i+1)%N,j] 
        south = state[(i-1)%N,j]
        NE = state[(i+1)%N,(j+1)%N] 
        NW =  state[(i+1)%N,(j-1)%N] 
        SE = state[(i-1)%N,(j+1)%N] 
        SW = state[(i-1)%N,(j-1)%N]
        
        alive = 0
        neigbours = [east, west, north, south]
        if(nns==8):            
            neigbours += [NE, NW, SE, SW]
            
        for nn in neigbours:
            if nn==1:
                alive+=1
                
        return alive
    
    def active(self,state):
        '''
        A function to measure the total alive/infected cells in the system.
        '''
        
        sum = 0
        for i in range(self.N):
            for j in range(self.N):
                if(state[i,j]==1):
                    sum+=1
        
        return sum
    
    def centreOfMass(self,state):
        '''
        A function to calculate the position of the centre of mass of the glider,
        rejecting points when it crosses the periodic boundaries.
        '''        
        glider = np.where(state==1)
        if( np.all(glider[0] != 0) and np.all(glider[1] != 0) ):
            x_cm = np.sum(glider[0]) / len(glider[0])
            y_cm = np.sum(glider[1]) / len(glider[1])
            print(f"Centre of mass ({ x_cm}, {y_cm})")
            r_cm = np.sqrt(x_cm**2 + y_cm**2)
            return x_cm, y_cm, r_cm        

    def GameOfLife(self, state):
        '''
        This function contains the rules for Conway's Game of Life. First, a temporary
        copy of the given state is created and then the function loops through all cells
        in the lattice to calculate their next state. The updated lattice is returned.
        '''         
        temp = np.copy(state)
        
        for i in range(self.N):
            for j in range(self.N):
                cell = state[i,j]
                alive = self.neighbours(state, i, j, 8)
                if((cell==1) and (alive < 2 or alive > 3)):
                    temp[i,j] = 0
                elif(cell==0 and alive==3):
                    temp[i,j] = 1
        
        return temp
    
    def SIRS(self, state):
        '''
        This function contains the rules for the SIRS model. The function 
        chooses a random cell site and updates its state according to the rules below.
        '''           
        i,j = np.random.randint(0,self.N,2)   # chooses a random cell site
        infected = self.neighbours(state, i, j, 4)
        cell = state[i,j]
        p1,p2,p3 = self.init
        r = np.random.random()
        
        if(cell==0 and infected>0 and r<p1):
            state[i,j] = 1
        elif(cell==1 and r<p2):
            state[i,j] = -1
        elif(cell==-1 and r<p3):
            state[i,j] = 0
        
        return state
    
    def animate(self, state):
        '''
        This function runs the simulation and animates on the run.
        '''  
        
        if(self.name =='Game of Life'):
            label = "Dead = 0 , Alive = 1"
            spins = [0,1]
            pause = 0.01
            
        elif(self.name =='SIRS'):
            label = "R = -1, S = 0, I = 1 "
            spins = [-1,0,1]  
            pause = 0.0001
        
        N = len(state)
        fig, ax = plt.subplots()
        vis = ax.pcolormesh(state, vmin=spins[0], vmax=spins[-1], cmap = plt.cm.get_cmap("viridis",len(spins)))  # plot the initial state
        fig.colorbar(vis, label=label, ticks=spins, ax=ax)   # colorbar to indicate the spins
        plt.pause(pause)        
        
        while(plt.fignum_exists(fig.number)):
           
            if(self.name=='Game of Life'):  
                alive = self.active(state)
                state = self.GameOfLife(state)  # evolve state according to Game of Life

            elif(self.name=='SIRS'):
                for sweep in range(N**2):
                    state = self.SIRS(state)  # evolve state according to SIRS model
     
            plt.cla()
            ax.set_title(f"{self.name} simulation on a {N} x {N} square lattice")
    #         ax.set_xticks(np.arange(0, N+1, 1))
    #         ax.set_yticks(np.arange(0, N+1, 1))
            vis = ax.pcolormesh(state, vmin=spins[0], vmax=spins[-1], cmap = plt.cm.get_cmap("viridis",len(spins)))  # plot the new state
    #         plt.draw()
            plt.pause(pause)
    
    # a straight line (y=Ax+B) function used for fitting
    def f(self, x, A, B): 
        return A*x + B
                   
    def measurements(self, state):
        '''
        This function runs the simulation and takes the specified measurements.
        It also creates plots and saves the datafiles.
        '''  
        
        if(self.name=='Game of Life' and self.init == 'r'):  
        
            t1 = time.time()
            generations = []
            total = 500  
            for i in range(total):
                
                t2 = time.time()  
                state = self.randomise(state)
                prev = 0
                streak = 0
                gen = 0
                
                while(streak!=10):  # 10 consecutive equal measurements
                
                    alive = self.active(state)
                    if(alive==prev):
                        streak += 1
                    else:
                        streak = 0  
                    prev = alive                              
                    state = self.GameOfLife(state)
                    
                    if(gen>5000):  # prevents infinite runs in the rare case of a remaining glider
                        break
                    gen +=1                                 
    
                t3 = time.time() 
                print(f"This run took {t3-t2} seconds.")
                generations.append(gen)
                     
            plt.hist(generations, rwidth=0.92, bins = np.arange(0,5100,100))
            plt.title(f"Histogram showing the time needed for\n {total} systems to reach equilibrium")             
            plt.ylabel("Count")   
            plt.xlabel("Time (in generations/sweeps)") 
            plt.xticks(np.arange(0,5500,500))
            self.filewrite('Histogram', [generations])
            plt.savefig('Datafiles/Histogram.png')
            print(f"Total histogram simulation took {t3-t1} seconds.")
            plt.show()            
            plt.clf()
            
        elif(self.name=='Game of Life' and self.init == 'g'):
            x_cm =[]
            y_cm =[]
            r_cm =[]
            #for i in range(500):
            while(self.centreOfMass(state) != None):  # stops when glider hits the boundary
                x, y, r = self.centreOfMass(state)
                x_cm.append(x)
                y_cm.append(y)
                r_cm.append(r)
                state = self.GameOfLife(state) 
            
            fig, ax = plt.subplots(1,3, figsize = (16,4.5)) 
            plt.rcParams['font.size'] = 10
            t = np.arange(0,len(r_cm),1)   
            popt, pcov = curve_fit(self.f, t, x_cm)   # fits to a straight line
            print(f"\nThe glider speed in x is {np.abs(popt[0])}")
            ax[0].set_title("Glider centre of mass x position over time")             
            ax[0].set_ylabel("Position x")   
            ax[0].set_xlabel("Time (in generations)")  
            label = f'y = {popt[0]:.3f}t + {popt[1]:.3f}'
            ax[0].plot(x_cm, 'bs', markersize=2, label=label)
            ax[0].legend(loc="upper right")
            popt, pcov = curve_fit(self.f, t, y_cm)   # fits to a straight line
            print(f"\nThe glider speed in y is {np.abs(popt[0])}")
            ax[1].set_title("Glider centre of mass y position over time")             
            ax[1].set_ylabel("Position y")   
            ax[1].set_xlabel("Time (in generations)")
            label = f'y = {popt[0]:.3f}t + {popt[1]:.3f}'
            ax[1].plot(y_cm, 'ro', markersize=2, label=label)
            ax[1].legend(loc="upper right")
            popt, pcov = curve_fit(self.f, t, r_cm)   # fits to a straight line
            print(f"\nThe glider total speed is {np.abs(popt[0])}")
            ax[2].set_title("Glider centre of mass r position over time")             
            ax[2].set_ylabel("Position r")   
            ax[2].set_xlabel("Time (in generations)") 
            label = f'y = {popt[0]:.3f}t + {popt[1]:.3f}' 
            ax[2].plot(r_cm, 'yo', markersize=2, label=label)
            self.filewrite('Glider', [x_cm, y_cm, r_cm])
            ax[2].legend(loc="upper right")
            plt.savefig('Datafiles/Glider.png')
            plt.show()            
            plt.clf()            
            
        elif(self.name=='SIRS' and self.init == 'p'):
            
            t1 = time.time()        
            res = 0.05      
            p1s = np.arange(0,1+res,res)
            p2 = 0.5
            p3s = np.arange(0,1+res,res)
            I_avg = []
            I_var = []
            
            for p1 in p1s:
                for p3 in p3s:
                    print(f"Now running with p1={p1}, p2={p2}, p3={p3}")
                    self.init = p1,p2,p3
                    state = self.randomise(self.lattice)
                    infected = []
                    
                    for sweep in range(1000):
                        for flip in range(self.N**2):                    
                            state = self.SIRS(state)  
                        
                        if(sweep>100):   # equilibration
                            active = self.active(state)
                            if(active!=0):
                                infected.append(active)
                            else:
                                infected.append(0) # stop when reach absorbing state
                                break
                    
                    infected = np.array(infected)
                    print(f"<I> / N = {np.mean(infected) / (self.N**2)}")
                    I_avg.append(np.mean(infected) / (self.N**2))
                    I_var.append((np.mean(infected**2)-np.mean(infected)**2) / (self.N**2))
            
            fig, ax = plt.subplots()            
            I_avg = np.array(I_avg).reshape(len(p1s),len(p1s))
            I_var = np.array(I_var).reshape(len(p1s),len(p1s))
            P1S, P3S = np.meshgrid(p1s,p3s)
            #vis = ax.contourf(P1S, P3S, I_avg)
            vis = plt.imshow(I_avg, origin='lower', extent=(0,1,0,1))
            plt.xlabel("p3")
            plt.ylabel("p1")
            plt.title(f"Phase diagram with p2 fixed at {p2}")
            fig.colorbar(vis, label=r"< I > / N", ax=ax)
            self.filewrite('Contour', [P3S.flatten(), P1S.flatten(), I_avg.flatten(), I_var.flatten()])
            plt.savefig('Datafiles/Phase.png')
            plt.show()            
            plt.clf()            
            
            fig, ax = plt.subplots()            
            #vis = ax.contourf(P1S, P3S, I_var)
            vis = plt.imshow(I_var, origin='lower', extent=(0,1,0,1))
            plt.xlabel("p3")
            plt.ylabel("p1")
            plt.title(f"Contour plot of variance with p2 fixed at {p2}")
            fig.colorbar(vis, label=r"$< I^2 > - < I >^2 \; / \; N$", ax=ax)
            plt.savefig('Datafiles/VarianceContour.png')    
            t2 = time.time()
            print(f"Phase plots simulation took {t2-t1} seconds.")
            plt.show()            
            plt.clf() 
            
        elif(self.name=='SIRS' and self.init == 'v'):
            
            t1 = time.time()      
            res = 0.01
            p1s = np.arange(0.2,0.5+res,res)
            p2 = 0.5
            p3 = 0.5
            I_var = []
            I_err = []
            
            for p1 in p1s:
                print(f"Now running with p1={p1}, p2={p2}, p3={p3}")
                self.init = p1,p2,p3
                state = self.randomise(self.lattice)
                infected = []
                
                for sweep in range(10000):  # increased for the Cut plot
                    for flip in range(self.N**2):                    
                        state = self.SIRS(state)  
                    
                    if(sweep>100):  # equilibration
                        active = self.active(state)
                        if(active!=0):
                            infected.append(active)
                        else:
                            infected.append(0) # stop when reach absorbing state
                            break
                        
                infected = np.array(infected)
                print(f"<I> / N = {np.mean(infected) / (self.N**2)}")
                I_var.append((np.mean(infected**2)-np.mean(infected)**2) / (self.N**2))
                I_err.append(self.resampling(infected))
            
            self.filewrite('VarianceCut', [p1s, I_var, I_err])     
            fig, ax = plt.subplots()
            plt.errorbar(p1s, I_var, yerr=I_err, fmt='bo', markersize=3)
            plt.xlabel("p1")
            plt.ylabel(r"$< I^2 > - < I >^2 \; / \; N$")
            plt.title(f"Variance along the cut with p2 fixed\nat {p2} and p3 fixed at {p3}")
            plt.savefig('Datafiles/VarianceCut.png')                       
            t2 = time.time()
            print(f"Total variance cut simulation took {t2-t1} seconds.")
            plt.show()            
            plt.clf()
        
        elif(self.name=='SIRS' and self.init == 'i'):
            
            p1 = p2 = p3 = 0.5
            self.init = p1,p2,p3
            self.immunity(5)
                
            p1 = 0.8
            p2 = 0.1
            p3 = 0.02
            self.init = p1,p2,p3
            self.N = 100
            self.lattice = np.zeros((self.N,self.N))
            self.immunity(5)
    
    def immunity(self, runs):
        '''
        A function for making the immunity measurements
        '''
        
        t1 = time.time()
        fims = np.linspace(0,1,101)  # 101 equally spaced measurements
        p1,p2,p3 = self.init
        print(f"\nNow running with p1={p1}, p2={p2}, p3={p3} and N={self.N}")
        I_scaled = []
        I_err = []
        
        for fim in fims:   
            I_avg = []
            
            for i in range(runs):
                print(f"\nRun {i}")                                
                state = self.randomise(self.lattice)
                self.vaccinate(state, fim)                
                infected=[]
                
                for sweep in range(3000):   # 3000 sweeps
                    for flip in range(self.N**2):                    
                        state = self.SIRS(state)  
                    
                    if(sweep>100):   # equilibration
                        active = self.active(state)
                        if(active!=0):
                            infected.append(active)
                        else:
                            infected.append(0)  # stop when reach absorbing state
                            break
                        
                infected = np.array(infected)
                print(f"<I> / N = {np.mean(infected) / (self.N**2)}")
                I_avg.append(np.mean(infected))
            
            I_avg = np.array(I_avg)
            I_scaled.append(np.mean(I_avg) / (self.N**2))
            error = np.std(I_avg) / np.sqrt(len(I_avg))
            print(f"scaled error value = {error / (self.N**2)}")
            I_err.append(error / (self.N**2))
            
        fig, ax = plt.subplots()
        plt.errorbar(fims, I_scaled, yerr=I_err, fmt='bo', markersize=3)
        plt.xlabel("Fraction of immune agents")
        plt.ylabel(r"$< I > \; / \; N$")
        plt.title(f"Average infected fraction vs immune fraction\n with p1={p1}, p2={p2}, p3={p3}")
        self.filewrite(f'Immunity{(p1,p2,p3)}', [fims, I_scaled, I_err])
        plt.savefig(f'Datafiles/Immunity{(p1,p2,p3)}.png')                   
        t2 = time.time()
        print(f"\nImmunity simulation took {t2-t1} seconds.")          
        plt.show()            
        plt.clf()    
    
    def vaccinate(self, state, fim):
        '''
        A function to vaccinate a specific proportion of the population,
        based on this provided fraction.
        '''
        
        print(f"\nVaccination with Fim = {fim}")
        for vacc in range(int(fim*(self.N**2))):
            i,j = np.random.randint(0,self.N,2)   # chooses a random cell site
            while(state[i,j]== -2):     # does not count if already vaccinated                
                i,j = np.random.randint(0,self.N,2)   # chooses a random cell site
            state[i,j]= -2  # vaccinate cell
            
        return state  
    
    def resampling(self, data):
        '''
        This function calculates the errors based on the Bootstrap method.
        '''          
        n = len(data)
        values = []
        for k in range(1000):   
            r = (np.random.random(n)*n).astype(int)   # take n random measurements
            resample = data[r]
            value = (np.mean(resample**2)-np.mean(resample)**2)/(self.N**2)
            values.append(value)
        
        values = np.array(values)
        error = np.sqrt(np.mean(values**2)-np.mean(values)**2)  # calculate the error
        
        return error   
    
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
           
def main():
    '''
    The main function for controlling the flow of the simulation. Allows the user to choose
    between animation and measurements, and runs the code accordingly.
    '''             
    AorM = input("Please type 'a' for Animation or 'm' for Measurements: ")
    
    if(AorM=='a'):
        
        print("Animation:")
        cellular = Simulation(AorM)
        cellular.animate(cellular.lattice)
        
    elif(AorM=='m'):
        
        print("Measurements:")
        cellular = Simulation(AorM)
        cellular.measurements(cellular.lattice)
        
    else:
        print("Invalid entry. Please try again.")
        main()

main()      