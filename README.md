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

## How it works
PythonOOF is a complete project that can be built and installed using a simple `pip install .` command, and therefore contains a number of files that are not strictly relevant to the Python-Fortran bridge.
This section will focus on only those components of the project that are relevant for the main problem.

Our basic problem is this: We want to use the Object Oriented Programming features of Modern Fortran (heretofore referred to as OOF) in a Fortran library that can be seemlessly accessed by a Python script. 
Fortran and Python have some fundamental differences in how data is represented in the two languages, particularly in how the two language represent multi-dimensional arrays and strings. We use Cython to act 
as an an intermediary between the two language, and so our Fortran code will make use of the `iso_c_binding` intrinsic module to allow our Fortran code to communicate with the C code that Cython generates.

We start by defining some data types that will be used throughout the Fortran codebase. To aid in the ease of conversion between the Fortran and Cython, we define the double precision type using `iso_c_binding` 
type definitions.  These are found in `pythonOOF/src/globals/globals_module.f90`:

```Fortran
module globals
   !! author: David A. Minton
   !!
   !! Basic parameters, definitions, and global type definitions used throughout the PyOOF project
   use, intrinsic :: iso_c_binding  ! Use the intrinsic kind definitions
   implicit none
   public

   integer, parameter :: I8B = c_int_least64_t !! Symbolic name for kind types of 8-byte integers
   integer, parameter :: I4B = c_int_least32_t !! Symbolic name for kind types of 4-byte integers
   integer, parameter :: I2B = c_int_least16_t !! Symbolic name for kind types of 2-byte integers
   integer, parameter :: I1B = c_int_least8_t  !! Symbolic name for kind types of 1-byte integers

   integer, parameter :: SP = c_float  !! Symbolic name for kind types of single-precision reals
   integer, parameter :: DP = c_double  !! Symbolic name for kind types of double-precision reals
   integer, parameter :: QP = c_long_double !! Symbolic name for kind types of quad-precision reals

   character(*,kind=c_char), parameter :: VERSION = "2023.10.0" !! Cratermaker version

   integer(I4B), parameter :: STRMAX = 512 !! Maximum size of character strings 

end module globals
```


We next Fortran derived-type that contains a 2D allocatable array of double precision floating point values, and a fixed-length string. It also contains some type-bound procedures that handle allocation and deallocation
of the allocatables, as well as a finalizer. This defintion is in the file `pythonOOF/src/simulation/simulation_module.f90`

```Fortran
   type  :: simulation_type
      real(DP), dimension(:,:), allocatable :: doublevar    !! A placeholder 2D array. 
      character(len=STRMAX)                 :: stringvar    !! A placeholder for a string component variable
   contains
      procedure :: allocate   => simulation_allocate   !! Allocate the allocatable components of the class
      procedure :: deallocate => simulation_deallocate !! Deallocate all allocatable components of the class
      final     ::               simulation_final      !! Finalizer (calls deallocate)
   end type simulation_type
```

Due to the allocatable components, this derived type is not C interoperable, so we cannot simply bind `simulation_type` to C to make it work. Therefore we need to write a set of Fortran functions that can bind the component variables
to C individually. In order to keep the Fortran functionality independent from the Fortran<->Python bridge, we implement our C binding procedures in a separate module defined in `pythonOOF/src/bind/bind_module.f90`.

The `bind` module provides several procedures for interfacing Cython code with Fortran. These are:

### `bind_simulation_init(ny, nx)`

- **Purpose**: Initializes the `simulation_type` derived type object in Fortran and returns a pointer to the object. This pointer can be used as a C struct and is ultimately linked to the Python class object via Cython.
  
- **Arguments**:
  - `ny, nx`: The dimensions of the array to allocate. Note that this expects row-major ordering.

### `bind_simulation_final(sim)`

- **Purpose**: Deallocates the pointer that links the C struct to the Fortran derived type object.

- **Arguments**:
  - `sim`: A C pointer to the Fortran simulation structure.

### `bind_simulation_get_stringvar(c_sim)`

- **Purpose**: Retrieves the string variable from the Fortran `simulation_type` derived type and passes it to C and ultimately to Python via Cython.

- **Arguments**:
  - `c_sim`: A C pointer to the Fortran simulation structure.

### `bind_simulation_set_stringvar(c_sim, c_string)`

- **Purpose**: Sets the value of the string variable in the Fortran `simulation_type` derived type.

- **Arguments**:
  - `c_sim`: A C pointer to the Fortran simulation structure.
  - `c_string`: A C-style string.

### `bind_c2f_string(c_string, f_string)`

- **Purpose**: Converts C-style strings to Fortran-style strings, enabling Python strings to be passed as arguments to Fortran functions.

- **Arguments**:
  - `c_string`: An input C-style string.
  - `f_string`: An output Fortran-style string.

### `bind_f2c_string(f_string, c_string)`

- **Purpose**: Converts Fortran-style strings to C-style strings, allowing Python to read strings that were created in Fortran procedures.

- **Arguments**:
  - `f_string`: An output Fortran-style string.
  - `c_string`: An output C-style string.


The Cython file `simulation.pyx` (and its companion C header `simulation.h`) define the functions that allow the Python code to interact with Fortran. 

# `simulation.pyx` Documentation

## Import Statements

The Cython code starts by importing necessary modules:

- `cython`
- `cpython`
- `numpy` as `cnp`
- Standard C libraries for memory allocation and string manipulation

## C-External Definitions

The `simulation.h` header is imported and the structure `c_simulation_type` is defined to map to the Fortran derived type. Additionally, the functions `bind_simulation_init`, `bind_simulation_final`, `bind_simulation_set_stringvar`, and `bind_simulation_get_stringvar` are declared.

## `Simulation` Class

This Cython class wraps the Fortran `simulation_type` derived type.

### `__cinit__(self, tuple shape)`

- **Purpose**: Initializes the object by calling Fortran's initializer.
- **Parameters**: 
  - `shape`: Shape of the Numpy array to initialize in Fortran
- **Returns**: Sets the `fobj` class variable.

### `__dealloc__(self)`

- **Purpose**: Deallocates the Fortran object.
- **Parameters**: None
- **Returns**: Deallocates `fobj`.

### `get_doublevar(self)`

- **Purpose**: Retrieves the `doublevar` array from Fortran and returns it as a Numpy array.
- **Parameters**: None
- **Returns**: 
  - `doublevar_array`: 2D Numpy array.

### `set_doublevar(self, cnp.ndarray[cnp.float64_t, ndim=2] doublevar_array)`

- **Purpose**: Sets the `doublevar` array in Fortran from a Numpy array.
- **Parameters**: 
  - `doublevar_array`: 2D Numpy array
- **Returns**: None

### `get_stringvar(self)`

- **Purpose**: Retrieves the `stringvar` from Fortran and returns it as a Python string.
- **Parameters**: None
- **Returns**: 
  - `string`: Python string.

### `set_stringvar(self, str string)`

- **Purpose**: Sets the `stringvar` in Fortran from a Python string.
- **Parameters**: 
  - `string`: Python string.
- **Returns**: None



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

## Example
A basic example of how the code operates is included in `pythonOOF/examples/basic_operation.py`. This is a Python script that will instantiate a Python class that contains the Fortran derived type as a component variable. 
It will show how one can manipulate the components of the Fortran object in either Fortran or Python and confirm that the conversions between the two data structures are handled correctly.



