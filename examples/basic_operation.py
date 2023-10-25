import numpy as np
from pyoof import Simulation  

# Define grid size
gridsize = 3

# Create a Simulation object
simulation = Simulation(gridsize)
init_vals = simulation.get_doublevar()
print("Initialized values should be all -1.0 as set by the Fortran side")
print(init_vals)

# Create a NumPy array for elevation data
elev_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)

# Set the elevation data using the set_doublevar method
simulation.set_doublevar(elev_data)

# Retrieve and print the elevation data using the get_doublevar method
retrieved_elev_data = simulation.get_doublevar()
print("Elevation Data:")
print(retrieved_elev_data)

# Verify that the retrieved data matches the original data
if np.array_equal(elev_data, retrieved_elev_data):
    print("Elevation data matches.")
else:
    print("Elevation data does not match.")
