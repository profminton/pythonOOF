# PythonOOF: Bridging Python and Modern Fortran OOP

PythonOOF is a working example of how one could go about building a software bridge facilitating seamless interaction between Python class objects and Modern Fortran's derived-type classes, focusing especially on those with allocatable array components and type-bound procedures. 

## Key Features

- **Object-Oriented Interoperability**: Create and manipulate Python objects that allocate array components of a Fortran derived type, fully equipped with type-bound procedures (OOF).
- **Data Manipulation**: Set the values of the Fortran derived type's allocatable arrays using Numpy arrays, and manipulate them efficiently with Fortran subroutines.
- **Automatic Memory Management**: Fortran's side of the bridge handles automatic deallocation of derived types with finalizers, ensuring memory efficiency.
- **Cython Integration**: PythonOOF uses Cython for smooth data interchange between Python and Fortran, eliminating the need for f2py.
- **CMAKE & scikit-build-core**: The build process is managed through CMAKE, integrated with scikit-build-core, simplifying the compilation and installation process.

## PythonOOF Project Structure

- 📄 **CMakeLists.txt** - Main CMake configuration file.
- 📄 **LICENSE** - Project license information.
- 🖼 **PyOOF_Programmers_at_work.png** - Totally accurate picture of what it's like to develop software in a mixed Fortran/Python environment.
- 📄 **README.md** - Project documentation and overview.
- 📁 **cmake**
  - Project-specific CMake modules.
  - 📁 **Modules**
    - 📄 **FindCoarray_Fortran.cmake** - Determine compiler flags to use Coarray Fortran parallelization.
    - 📄 **FindMKL.cmake** - Locate the Intel MKL library.
    - 📄 **FindOpenMP_Fortran.cmake** - Determine compiler flags to use OpenMP parallelization.
    - 📄 **SetCompilerFlag.cmake** - Determine whether a particular compiler flag is valid by testing each one individually.
    - 📄 **SetParallelizationLibrary.cmake** - Configure the parallelization options and find any and all appropriate libraries.
    - 📄 **SetPyOOFFlags.cmake** - The complete set of possible compiler flags, for various combinations of compilers, operating systems, and build configurations (Release, Debug, Testing, Profiling).
- 📄 **distclean.cmake** - Script for cleaning up build artifacts.
- 📁 **examples**
  - Sample scripts demonstrating project usage.
  - 📄 **basic_operation.py** - An example to demonstrate the basic functionality of the PythonOOF project.
- 📁 **pyoof**
  - Python module main folder.
  - 📄 **CMakeLists.txt** - CMake config for Python module.
  - 📄 **__init__.py** - Python package initializer.
  - 📄 **simulation.h** - Header file for Cython bindings.
  - 📄 **simulation.pyx** - Cython file for Python bindings and defines a class called Simulation to be used in a notional project
- 📄 **pyproject.toml** - Python project metadata and build tool config.
- 📁 **src**
  - Fortran source files.
  - 📄 **CMakeLists.txt** - CMake config for source directory.
  - 📁 **bind**
    - 📄 **bind_module.f90** - Defines the C to Fortran bindings that allow the Python module to interface with the Fortran derived type.
  - 📁 **globals**
    - 📄 **globals_module.f90** - Defines global variables, including variable data types and the project version.
    - 📄 **globals_module.f90.in** - A configuration file used by CMake in order to automatically set the project version based on the value in the pyproject.toml file at the time the project is built.
  - 📁 **simulation**
    - 📄 **simulation_module.f90** - Defines a derived type that contains the allocatable arrays and type-bound procedures that are manipulated with the Python module.
- 📄 **version.txt** - Current version of the project.


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
```


