import matplotlib.pyplot as plt
import numpy as np
import time
import math
import matplotlib.animation as animation

# J = kB = 1  (for this simulation)

# A function to calculate the whole energy of the state needed for measurements.
def energy(state):
    
    N = len(state)
    sum = 0
    for i in range(N):
        for j in range(N):
            spin = state[i,j]
            spin_r = state[i, (j+1)%N]
            spin_d = state[(i+1)%N, j]
            sum += spin*spin_r + spin*spin_d
    
    return -sum

# A function to calculate the whole magnetisation of the state needed for measurements.
def magnetisation(state):
    
    N = len(state)
    sum = 0
    for i in range(N):
        for j in range(N):
            sum += state[i,j]
            
    return sum

# A function to randomise the spins of the state.
def randomise(state):
    
    N = len(state)
    for i in range(N):
        for j in range(N):
            r = np.random.random()
            if(r<0.5):
                state[i,j] *= -1
    
#     print(state)
    print(f"\nThe energy of the randomised state is {energy(state)},")
    print(f"and its magnetisation is {magnetisation(state)}.")
    return state

# A function to calculate the energy change (ΔE) for a flip in Glauber dynamics,
# using the 4 nearest neighbours.
def energy_change(state, i, j):
    
    N = len(state)
    spin = state[i,j]
    nn_sum = state[i,(j+1)%N] + state[i,(j-1)%N] + state[(i+1)%N,j] + state[(i-1)%N,j]
    delta_e = 2 * spin * nn_sum
    
    return delta_e

# Glauber dynamics with the Metropolis algorithm.
def glauber(state, T):
    
    N = len(state)
    i,j = (np.random.random(2)*N).astype(int)   # chooses a random spin site
    delta_e = energy_change(state, i, j)      # calculates ΔE 
    
    p = math.exp(-(delta_e)/T)  # Boltzmann weight
    prob = np.minimum(1, p)     # Metropolis algorithm
    r = np.random.random()
    if(r<prob):
        state[i,j] *= -1  # flip the chosen spin
        
    return(state)

# Kawasaki dynamics with the Metropolis algorithm, using ΔE from Glauber
# with a correction in the case of nearest neighbours.
def kawasaki(state, T):
    
    N = len(state)
    i1,j1,i2,j2 = (np.random.random(4)*N).astype(int)  # chooses two random spin sites
    
    while(state[i1,j1] == state[i2,j2]):   # avoid calculation if same spins
        i1,j1,i2,j2 = (np.random.random(4)*N).astype(int)
    
    delta_e1 = energy_change(state, i1, j1)   # calculates ΔE1 as if a Glauber flip
    delta_e2 = energy_change(state, i2, j2)   # calculates ΔE2 as if a Glauber flip
    delta_e = delta_e1 + delta_e2
    
    horz_nn = (i1==i2 and np.abs(j1-j2)==1)  # for horizontal nearest neighbours
    vert_nn = (np.abs(i1-i2)==1 and j1==j2)  # for vertical nearest neighbours
    
    if(horz_nn or vert_nn):  # if the two spin sites are nearest neighbours 
        delta_e += 4     # correction for nearest neighbours 
        
    p = math.exp(-(delta_e)/T)  # Boltzmann weight
    prob = np.minimum(1, p)     # Metropolis algorithm
    r = np.random.random()
    if(r<prob):
        state[i1,j1] *= -1  # equivalent to swapping the two opposite spins
        state[i2,j2] *= -1  # equivalent to swapping the two opposite spins
        
    return(state)

# A function for running the simulation and taking measurements of Energy 
# and Magnetisation of the states. 
def measurements(state, dynamics):   
    
    t1 = time.time()
    temps = np.arange(1,3.1,0.1)  # Temperature range from 1 to 3 in steps of 0.1
    N = len(state)
    
    # arrays to store the observables and the corresponding errors
    E_avg = []
    E_error = []
    Esqrd = []
    c_error = []
    M_avg = []
    M_error = []
    Msqrd = []
    chi_error = []
    
    for T in temps:
        print(f"Now measuring temperature: {T:.2f}")
        energies = []
        mags = []
        
        for sweep in range(10000):
            for flip in range(N**2):
                
                if(dynamics=='g'):
                    state = glauber(state, T)  # evolve state according to Glauber dynamics
                elif(dynamics=='k'):
                    state = kawasaki(state, T)  # evolve state according to Kawasaki dynamics
                else:
                    print("Invalid entry for dynamics. Please try again.")
                    break
                
            if(sweep>100 and (sweep%10)==0):  # equilibration and autocorrelation
                energies.append(energy(state))
                mags.append(np.abs(magnetisation(state)))
    
        E_avg.append(np.mean(energies))     # average energy <E>
        E_error.append(np.std(energies)/np.sqrt(len(energies)))  # error for energy
        Esqrd.append(np.mean(np.array(energies)**2))        # <E^2>
        c_error.append(resampling(np.array(energies), N**2, T**2))   # error for c
        
        M_avg.append(np.mean(mags))     # average absolute magnetisation  <|M|>
        M_error.append(np.std(mags)/np.sqrt(len(mags)))     # error for magnetisation
        Msqrd.append(np.mean(np.array(mags)**2))        #<|M|^2>
        chi_error.append(resampling(np.array(mags), N**2, T))     # error for chi
        
    E_avg = np.array(E_avg)
    M_avg = np.array(M_avg)
    t2 = time.time()
    print(f"\nAll measurements took {(t2-t1):.10f} seconds.") 
    
    plots(temps, E_avg, E_error, M_avg, M_error, Esqrd, c_error, Msqrd, chi_error, N**2)

# A function for calculating the errors based on the Bootstrap method.
def resampling(data, N, T):
    
    n = len(data)
    values = []
    for k in range(1000):   
        r = (np.random.random(n)*n).astype(int)   # take n random measurements
        resample = data[r]
        value = (np.mean(resample**2)-np.mean(resample)**2)/(N*T)  # calculate c or χ accordingly
        values.append(value)
    
    values = np.array(values)
    error = np.sqrt(np.mean(values**2)-np.mean(values)**2)  # calculate the error
    
    return error
    
# A function for producing the desired plots and outputting the data on a datafile.
def plots(T, E, E_err, M, M_err, Esqrd, c_err, Msqrd, chi_err, N):
    
    c = (Esqrd - (E**2))/(N* T**2)    # equation for scaled Heat Capacity
    chi = (Msqrd - (M**2))/(N*T)      # equation for susceptibility
    
    f=open('datafile.txt','w')   
    titles = 'Temperature, <E>, E_error, <|M|>, M_error, Heat Capacity c, c_error, Susceptibility chi, chi_error'
    
    f.write('%s\n\n'%(titles))    
    for i in range(len(T)):
        f.write('%lf %lf %lf %lf %lf %lf %lf %lf %lf\n'%(T[i], E[i], E_err[i], M[i], M_err[i], c[i], c_err[i], chi[i], chi_err[i]))
    f.close()
    
    plt.rcParams['figure.figsize'] = (9, 6)
    plt.rcParams['font.size'] = 13
    plt.errorbar(T, E, yerr=E_err, fmt='ro', markersize=3)
    plt.title("Average energy over temperature")
    plt.xlabel("Temperature T")
    plt.ylabel("Average energy <E>")
    plt.savefig('Energy.png')
    plt.show()
    plt.errorbar(T, c , yerr=c_err, fmt='ro', markersize=3)
    plt.title("Scaled Heat Capacity over Temperature")
    plt.xlabel("Temperature T")
    plt.ylabel("Scaled Heat Capacity c")
    plt.savefig('HeatCapacity.png')
    plt.show()
    plt.errorbar(T, M, yerr=M_err, fmt='bo', markersize=3)
    plt.title("Average absolute magnetisation over temperature")
    plt.xlabel("Temperature T")
    plt.ylabel("Average absolute magnetisation <|M|>")
    plt.savefig('Magnetisation.png')
    plt.show()
    plt.errorbar(T, chi, yerr=chi_err, fmt='bo', markersize=3)
    plt.title("Susceptibility over temperature")
    plt.xlabel("Temperature T")
    plt.ylabel("Susceptibility χ")
    plt.savefig('Susceptibility.png')
    plt.show()
    
# A function for running the simulation and animating on the run. The size, temperature
# and dynamics for the run are chosen from the user when running the program.     
def animate(state, T, dynamics):
    
    N = len(state)
    fig, ax = plt.subplots()
    vis = ax.pcolormesh(state, vmin=-1, vmax=1, cmap = plt.cm.get_cmap("viridis",2))  # plot the initial state
    fig.colorbar(vis, label="Spin", ticks = [-1,1], ax=ax)   # colorbar to indicate the spins
    
    while(plt.fignum_exists(fig.number)):
        for sweep in range(N**2):       # plot only every sweep
        
            if(dynamics=='g'):
                state = glauber(state, T)  # evolve state according to Glauber dynamics
            elif(dynamics=='k'):
                state = kawasaki(state, T)  # evolve state according to Kawasaki dynamics
            else:
                print("Invalid entry for dynamics. Please try again.")
                break
 
        plt.cla()
        ax.set_title(f"Ising model for a {N} x {N} square lattice")
#         ax.set_xticks(np.arange(0, N+1, 1))
#         ax.set_yticks(np.arange(0, N+1, 1))
        vis = ax.pcolormesh(state, vmin=-1, vmax=1, cmap = plt.cm.get_cmap("viridis",2))  # plot the new state
#         plt.draw()
        plt.pause(0.00005)

# A function to prompt for the system size N, catching any possible errors on the way.     
def size_prompt():
    
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

# A function to prompt for the temperature T, catching any possible errors on the way.     
def temp_prompt():
    
    T = input("Type in the temperature T: ")
    wrong = True
    while(wrong):
        try:                         
            T = float(T)
            if(T>0):      # ensure T is a float larger than 0
                wrong = False
            else:
                T = int("zero")  # ValueError
        except ValueError:
            print("Please enter a valid float larger than 0.")
            T = input("Type in the temperature T: ")
    
    return T  

# A function to prompt for the desired dynamics, catching any possible errors on the way.     
def dyn_prompt():
    
    dynamics = input("Type in the desired dynamics to be used\n('g' for Glauber or 'k' for Kawasaki): ")
    while(dynamics!='g' and dynamics!='k'):
        print("Invalid entry for dynamics. Please try again.")
        dynamics = input("Type in the desired dynamics to be used\n('g' for Glauber or 'k' for Kawasaki): ")
    
    return dynamics    
       
# The main function for controlling the flow of the simulation. Allows the user to choose
# between animation and measurements, and runs the code accordingly.    
def main():
    
    # prompt for required values
    AorM = input("Please type 'a' for Animation or 'm' for Measurements: ")
    
    if(AorM=='a'):
        
        print("Animation:")
        N = size_prompt()
        T = temp_prompt()    
        dynamics = dyn_prompt()
        
        print(f"\nSystem size = {N}") 
        print(f"Temperature = {T}")  
        print(f"Dynamics = {dynamics}")
        
        lattice = np.ones((N,N))  # initial state of all spins up
        print(f"\nThe calculated energy of the initial configuration is {energy(lattice)},")
        print(f"the corresponding theoretical value is {-2*N**2}.")
        print(f"The magnetisation is {magnetisation(lattice)}.")
        
        state = randomise(lattice)   # randomise the spins of the initial state       
        animate(state, T, dynamics)   # run the animation code
        
    elif(AorM=='m'):
        
        print("Measurements:")
        N = size_prompt()
        dynamics = dyn_prompt()
        
        state = np.ones((N,N))   # initial state of all spins up
        if(dynamics=='k'):
            state[:,int(N/2):] *= -1   # only used for initial condition of Kawasaki
        print(state)
        print(f"\nThe calculated energy of the initial configuration is {energy(state)},")
        print(f"the corresponding theoretical value is {-2*N**2}.")
        print(f"The magnetisation is {magnetisation(state)}.\n")
        
        measurements(state, dynamics)   # run the measurements code
        
    else:
        print("Invalid entry. Please try again.")
        main()

main()