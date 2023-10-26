import numpy as np
from pyoof import Simulation  

# Define some values to test
new_arr_expected = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float64)
init_arr_expected = -np.ones_like(new_arr_expected, dtype=np.float64)
arr_shape = new_arr_expected.shape
init_str_expected = "Initialized in Fortran"
new_str_expected = "Modified in Python"

print(f"Initializing the Simulation object with an array with shape {arr_shape}")
simulation = Simulation(arr_shape)
print("Simulation object has been initialized successfully!")

print("Retrieving the current contents of the doublevar variable that was initialized in Fortran to be -1.0")
init_arr_retrieved = simulation.get_doublevar()
print("Variable retrieval succeeded!\nHere are the properties of the retrieved array:")
print(f"Type:  {type(init_arr_retrieved)}")
print(f"Shape: {init_arr_retrieved.shape}")
print(f"Size:  {init_arr_retrieved.size}")
print(f"Value: {init_arr_retrieved}")
assert np.array_equal(init_arr_retrieved, init_arr_expected), "Initial array value does not match expected"
print("Initial array values match")

init_str_retrieved = simulation.get_stringvar()
print(f"Initialized string variable should say: {init_str_expected}")
print(f"Retrieved: {init_str_retrieved}")
assert init_str_retrieved == init_str_expected, "Initial string value does not match"
print("Initial string variable matches")

# Set the elevation data using the set_doublevar method
simulation.set_doublevar(new_arr_expected)

# Set a new value for the string
simulation.set_stringvar(new_str_expected)

# Retrieve and print the array data using the get_doublevar method
new_arr_retrieved = simulation.get_doublevar()
print("Array Data:")
print(new_arr_retrieved)

new_str_retrieved = simulation.get_stringvar()
print("String Data:")
print(new_str_retrieved)

# Verify that the retrieved data matches the original data
assert np.array_equal(new_arr_retrieved, new_arr_expected), "Array data does not match."
print("Array data matches.")

assert new_str_retrieved == new_str_expected, "String variable does not match."
print("String variable matches.")
