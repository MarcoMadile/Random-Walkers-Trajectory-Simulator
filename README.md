# Random Walkers Trajectory Simulator

This repository contains Python scripts for simulating the trajectories of random walkers. Unique to this simulator is the capability to determine the length of each step and the waiting time after each step based on user-defined distributions. It also has functionalities to plot these trajectories, identify intersections between them, and subsequently plot a graph based on these intersections.

## Features

- Simulation of trajectories of multiple individuals based on user-defined step size and waiting time distributions
- Intersection detection between trajectories (applicable for more than one walker)
- Plotting of individual trajectories with their intersections
- Creation and plotting of a network graph showing the nodes (individuals) and edges (intersections)

## Repository Contents

- `CPU_code.py`: This script uses NumPy for generating and handling the trajectories and NetworkX for creating and plotting the network graph. It also uses SciPy for calculating the distance matrix and Matplotlib for plotting.

## Dependencies

- NumPy
- SciPy
- Matplotlib
- NetworkX

## Setup and Usage

1. Clone this repository to your local machine using `git clone https://github.com/<your_username>/<repo_name>.git`.
2. Install the necessary Python packages if not already installed. 

### Usage:
Open the CPU_code.py script in your preferred text editor.

To generate a random walk simulation, you need to create an Individual or Ensemble instance. Here's an example:

data_dist_CPU = np.random.exponential(scale=1.0, size=1000) //or any other distribution
hist, bin_edges = np.histogram(data_dist_CPU, bins=10, density=True)
ind_CPU = CPU_code.Individual(initial_position=[0,0],waiting_time_dist=[hist,bin_edges],step_size_dist=[hist,bin_edges]) // this creates an individual with the specified distributions
time_ = 1000 // this is the total time of the simulation
ind_CPU.move(time=time_) // this moves the individual for the specified time
ind_CPU.plot_trajectory() // this plots the trajectory of the individual

You can accses the positions using ind_CPU.positions(). 


If you want to make an Ensamble of individuals, you can do it like this:

N_ind = 10 // number of individuals in the ensamble
initial_pos = np.zeros((N_ind,2)) // or you can specify the initial positions of the individuals 
ensamble_CPU = CPU_code.Ensemble(N_ind,initial_positions=initial_pos ,waiting_time_dist=[hist,bin_edges],step_size_dist=[hist,bin_edges])
ensamble_CPU.move(time=time_)// this moves the ensamble for the specified time
intersections= population.find_intersections(t0=5,x0=2)// this finds the intersections between the individuals when the pairs of points are separated by at least t0 and x0
population.plot(intersections)// plots trayectorys and intersections
population.plot_network(intersections,time_filter=30) // plot the network from intersection after the time_filter 

Then run the script.

## System Requirements

The CPU code should run on any system that supports Python and the necessary packages.


## Contact

Marco Madile Hjelt – marco.madile@ib.edu.ar – [MarcoMadile](https://github.com/MarcoMadile)
