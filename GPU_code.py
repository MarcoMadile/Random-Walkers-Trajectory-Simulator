import cupy as cp
import cupyx as cpx
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np




class Individual:
    def __init__(self, initial_position, waiting_time_dist, step_size_dist):
        self.ini_position = cp.array(initial_position)
        self.waiting_time_hist = waiting_time_dist[0]/cp.sum(waiting_time_dist[0])
        self.waiting_time_edges = (waiting_time_dist[1][:-1]+waiting_time_dist[1][1:])/2
        self.waiting_time_mean =  cp.sum(self.waiting_time_edges * self.waiting_time_hist)
        self.step_size_hist = step_size_dist[0]/cp.sum(step_size_dist[0])
        self.step_size_edges = (step_size_dist[1][:-1]+step_size_dist[1][1:])/2
        self.time = 0
        self.positions= [self.ini_position]


    def generate_waiting_time(self,size_=1):
        # Generate a random waiting time based on the waiting_time_dist
        # Replace the function below with your desired waiting time distribution function
        return cp.random.choice(self.waiting_time_edges,size=size_, p=self.waiting_time_hist)

    def generate_step_size(self,size_=1):
        # Generate a random step size based on the step_size_dist
        # Replace the function below with your desired step size distribution function
        return cp.random.choice(self.step_size_edges,size=size_, p=self.step_size_hist)

    def move(self,time):
        # Generate a random direction
        positions= cp.zeros((2,time))
        # get mean of waiting time distribution 
        posible_size_w_t = int(2*time/self.waiting_time_mean)  
        waiting_time= self.generate_waiting_time(size_=posible_size_w_t)                                              
        while cp.sum(waiting_time)<=time: 
            waiting_time = cp.hstack((waiting_time,self.generate_waiting_time(size_=posible_size_w_t)))
        cumulative_sum = cp.cumsum(waiting_time)
        index = cp.where(cumulative_sum > time)[0][0]
        filtered_waiting_time = waiting_time[:index]
        step_size = self.generate_step_size(size_=len(filtered_waiting_time))
        theta = cp.random.uniform(0, 2 * cp.pi,size=len(filtered_waiting_time))
        direction = [cp.cos(theta), cp.sin(theta)]
        # Update position based on this simple model where individual stays still for each waiting time (updating positions with the actual position) and when the waiting time finishes move with step size in direction. 
        filtered_waiting_time = filtered_waiting_time.astype(int)
        # check if it has 0 in the waiting time 
                
        filtered_waiting_time[filtered_waiting_time == 0] += 1
        # Generate movements for each waiting time
        movements = step_size[:, cp.newaxis] * cp.array(direction).T
        # Repeat each movement for the corresponding waiting time
        repeated_movements = cp.repeat(movements, filtered_waiting_time.tolist(), axis=0)
        # add initial position to repeated_movements
        repeated_movements = cp.vstack((self.ini_position,repeated_movements))
        # Calculate positions
        positions = cp.cumsum(repeated_movements, axis=0)[:time]
        # Update positions
        self.positions=positions    


## Ensemble of individuals
class Ensemble:
    def __init__(self, n, initial_positions, waiting_time_dist, step_size_dist):
        self.n = n
        self.ini_positions= cp.array(initial_positions)
        self.waiting_time_hist = waiting_time_dist[0]/cp.sum(waiting_time_dist[0])
        self.waiting_time_edges = (waiting_time_dist[1][:-1]+waiting_time_dist[1][1:])/2
        self.waiting_time_mean =  cp.sum(self.waiting_time_edges * self.waiting_time_hist)
        self.step_size_hist = step_size_dist[0]/cp.sum(step_size_dist[0])
        self.step_size_edges = (step_size_dist[1][:-1]+step_size_dist[1][1:])/2
        self.time = 0
        self.positions = cp.copy(self.ini_positions)


    def generate_waiting_time(self,size_=1):
        # Generate a random waiting time based on the waiting_time_dist
        # Replace the function below with your desired waiting time distribution function
        return cp.random.choice(self.waiting_time_edges,size=size_, p=self.waiting_time_hist)

    def generate_step_size(self,size_=1):
        # Generate a random step size based on the step_size_dist
        # Replace the function below with your desired step size distribution function
        return cp.random.choice(self.step_size_edges,size=size_, p=self.step_size_hist)


    def move(self, time):
        # Generate random directions, step sizes, and waiting times for all individuals at once
        posible_size_w_t = cp.int(4*time/self.waiting_time_mean) 
        thetas = cp.random.uniform(0, 2 * cp.pi, size=(self.n, posible_size_w_t))

        directions = cp.array([cp.cos(thetas), cp.sin(thetas)])

        step_sizes = self.generate_step_size(size_=(self.n, posible_size_w_t))

        waiting_times = self.generate_waiting_time(size_=(self.n, posible_size_w_t))
        waiting_times[waiting_times == 0] += 1
        result_movements = step_sizes.T[:, :, cp.newaxis] * directions.T
        # result_movements[i,j,k]= eje i (i=0 es x,i=1 es y ), individuo j, paso k 
        reshaped_result_movements = result_movements.reshape(( self.n, posible_size_w_t,2))
        # now result_movements[i,j,k]=  individuo , paso j, eje k 

        #now we use numpy functions once calculated all the movements
        reshaped_result_movements_np = cp.asnumpy(reshaped_result_movements)
        waiting_times_np = cp.asnumpy(waiting_times).astype(int)
        ensamble_positions = []
        for i in range(self.n):
            repeated_movements_individual = np.repeat(reshaped_result_movements_np[i], waiting_times_np[i], axis=0)
            where_are_equal = np.equal(repeated_movements_individual[1:,0],repeated_movements_individual[:-1,0])
            changes_in_positions= np.where(where_are_equal[:,np.newaxis],np.zeros(repeated_movements_individual.shape)[1:],repeated_movements_individual[1:])
            positions= np.cumsum(changes_in_positions, axis=0)[:time]
            ensamble_positions.append(positions)
        self.positions = cp.array(ensamble_positions)
        

# #Tests 

# data = cp.random.exponential(scale=1, size=1000)
# hist, bin_edges = cp.histogram(data, bins=10, density=True)

# time = 2*60

# n_ind= 30000
# initial_positions= cp.zeros((n_ind,2))

# population= Ensemble(n_ind,initial_positions,waiting_time_dist=[hist,bin_edges],step_size_dist=[hist,bin_edges])
# start_gpu = cp.cuda.Event()
# start_gpu.record()
# ensamble_positions = population.move(time) 

# end_gpu = cp.cuda.Event()
# end_gpu.record()
# end_gpu.synchronize()
# t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
# print("Tiempo GPU: ", round(t_gpu/1000,1),"s for N = ",n_ind)
