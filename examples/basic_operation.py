import numpy as np
from pyoof import Surface  

# Define grid size
gridsize = 3

# Create a Surface object
surface = Surface(gridsize)
init_vals = surface.get_elev()
print("Initialized values should be all -1.0 as set by the Fortran side")
print(init_vals)

# Create a NumPy array for elevation data
elev_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)

# Set the elevation data using the set_elev method
surface.set_elev(elev_data)

# Retrieve and print the elevation data using the get_elev method
retrieved_elev_data = surface.get_elev()
print("Elevation Data:")
print(retrieved_elev_data)

# Verify that the retrieved data matches the original data
if np.array_equal(elev_data, retrieved_elev_data):
    print("Elevation data matches.")
else:
    print("Elevation data does not match.")
