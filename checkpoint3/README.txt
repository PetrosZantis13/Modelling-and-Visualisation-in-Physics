Modelling and Visualisation in Physics 2021
Checkpoint 3
Petros Zantis s1703728

Instructions for the Cahn-Hilliard simulation:

1. Run the file CahnHilliard.py
2. Select between Animation('a') or Measurements('m')
3. Select the system size N (resulting in an NxN square system)
4. Select the initial condition phi_0 (e.g. +/- 0.5 (for droplet growth), 0 (for spinodal decomposition))
5. Enjoy the animation! The figure remains open until the user closes it.
6. In the case of measurements, the free energy density output datafiles and plots will be saved in the created Datafiles folder.


Instructions for the Poisson simulation:

1. Run the file Poisson.py
2. Select between Animation('a') or Measurements('m')
3. Select the desired problem to solve ('e' for Electric or 'm' for Magnetic)
4. Select the desired algorithm to be used ('j' for Jacobi's or 'g' for Gauss-Seidel)
5. Select the system size N (resulting in an NxN square system)
6. Select the desired tolerance for stopping (e.g. 0.001)
7. Enjoy the animation! The figure remains open until the user closes it.
8. In the case of measurements, the SOR method will be run for various omega values to determine the optimum.
   In all cases, the output datafiles and plots will be saved in the created Datafiles folder.
