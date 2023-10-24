# PythonOOF: Bridging Python and Modern Fortran OOP

PythonOOF is a working example of how one could go about building a software bridge facilitating seamless interaction between Python class objects and Modern Fortran's derived-type classes, focusing especially on those with allocatable array components and type-bound procedures. 

## Key Features

- **Object-Oriented Interoperability**: Create and manipulate Python objects that allocate array components of a Fortran derived type, fully equipped with type-bound procedures (OOF).
- **Data Manipulation**: Set the values of the Fortran derived type's allocatable arrays using Numpy arrays, and manipulate them efficiently with Fortran subroutines.
- **Automatic Memory Management**: Fortran's side of the bridge handles automatic deallocation of derived types with finalizers, ensuring memory efficiency.
- **Cython Integration**: PythonOOF uses Cython for smooth data interchange between Python and Fortran, eliminating the need for f2py.
- **CMAKE & scikit-build-core**: The build process is managed through CMAKE, integrated with scikit-build-core, simplifying the compilation and installation process.

## Installation

PythonOOF requires a working Python environment, a Fortran compiler, and CMAKE installed on your system.

To install PythonOOF, you need to clone the repository and then use pip. Here are the steps:

```bash
# Clone the repository
git clone https://github.itap.purdue.edu/daminton/pythonOOF

# Change directory to the repository folder
cd pythonOOF

# Install PythonOOF
pip install .
```

# Example
A basic example of how the code operates is included in `pythonOOF/examples`

```python
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
```


