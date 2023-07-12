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
2. Install the necessary Python packages if not already installed. You can use pip: `pip install numpy scipy matplotlib networkx`.
3. Run the script: `python CPU_code.py`.

## System Requirements

The CPU code should run on any system that supports Python and the necessary packages.

## Known Issues and Limitations

(Provide any known issues or limitations here)

## License

(Provide information about the project's license here)

## Contact

(Your Name – Your Email – [Your GitHub Profile](https://github.com/<your_username>))
