# cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii
cimport cython
from cpython cimport PyUnicode_AsUTF8AndSize, PyUnicode_FromString
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset 

"""
    The definitions below provides the Cython versions of the definitions found in the C header. They should exactly mirror the contents of header.h, just with a modified Cython syntax.

    One important note: In order to map the data structure from Python/Cython to Fortran, it is important that the order in which component variables are defined is the same here, as it is in `header.h` and `simulation_module.f90`. The only difference between the struct definition here and the type definition in Fortran is that the allocatable arrays require a "hidden" shape (or length) variable in the Cython and C definitions that immediately follow the pointer to the data itself, hence the need for a definition of *doublevar and doublevar_shape[2] for the allocatable doublevar(:,:) in Fortran.
"""
cdef extern from "simulation.h":
    ctypedef struct c_simulation_type:
        double *doublevar
        int doublevar_shape[2]
        char *stringvar
        int stringvar_len

    c_simulation_type* bind_simulation_init(int ny, int nx)
    void bind_simulation_final(c_simulation_type *obj)
    void bind_simulation_set_stringvar(c_simulation_type *obj, const char *c_string)
    char* bind_simulation_get_stringvar(c_simulation_type *obj)


cdef class _SimulationBind:
    """
    This defines the class that contains all of the methods and variables used to bind Cython to Fortran. It is not meant to be used direclty, but instead serves as an intermediary. 

    """

    cdef c_simulation_type* fobj

    def __cinit__(self, tuple shape):
        """
        Initializes the Simulation object by calling the Fortran initializer, which will allocate the array in Fortran, set some initial values, and return a pointer that connects the Fortran derived type class variable to the Python object.

        Parameters
        ----------
            Shape of the allocatable Numpy array to initialize in Fortran 

        Returns
        -------
            Sets the fobj component variable containing the components set by the Fortran object.
        """

        # Check to make sure we are passing a correct 2D array for the shape.
        if len(shape) != 2:
            raise ValueError("Expected a tuple of length 2 for shape")

        print("Cython: calling bind_simulation_init")
        self.fobj = bind_simulation_init(shape[0],shape[1])  
        print("Cython: Successfully returned")

        # Do some basic checks to make sure the object variable and all its components were allocated succesfully
        if self.fobj is NULL:
            raise MemoryError("Failed to allocate Fortran object.")
        else:
            print("The Fortran object was allocated successfully ")
        print(f"self.fobj           = {<unsigned long>self.fobj          }")
        print(f"self.fobj.doublevar = {<unsigned long>self.fobj.doublevar}")
        print(f"self.fobj.stringvar = {<unsigned long>self.fobj.stringvar}")

        if self.fobj.doublevar is NULL:
            raise MemoryError("Failed to allocate component variable 'doublevar' in the Fortran object.")
        else:
            print("The component variable 'doublevar' was allocated successfuly in the Fortran object")

        if self.fobj.stringvar is NULL: 
            raise MemoryError("Failed to allocate component variable 'stringvar' in the Fortran object.")
        else:
            print("The component variable 'stringvar' was allocated successfuly in the Fortran object")

        # Manually set the shape of the 2D component array in the Python object
        self.fobj.doublevar_shape[0] = shape[0]
        self.fobj.doublevar_shape[1] = shape[1]

        return


    def __dealloc__(self):
        """
        Finalizes the Fortran component variable.

        Parameters
        ----------
            None
        """
        if self.fobj is not NULL:
            bind_simulation_final(self.fobj)


    def get_doublevar(self):
        """
        A getter method that retrieves the doublevar allocatable array from Fortran and returns it as a Numpy array

        Parameters
        ----------
            None
        Returns
        -------
            doublevar_array : [y,x] Numpy array
        """

        # Retrieve the shape of the elevation data
        cdef cnp.npy_intp shape[2]
        shape[0] = self.fobj.doublevar_shape[0]
        shape[1] = self.fobj.doublevar_shape[1]

        # Create a NumPy array from the Fortran array
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] doublevar_array
        doublevar_array = cnp.PyArray_SimpleNewFromData(2, shape, cnp.NPY_FLOAT64, <void*>(self.fobj.doublevar))

        return doublevar_array
 
    def set_doublevar(self, cnp.ndarray[cnp.float64_t, ndim=2] doublevar_array):
        """
        A setter method that sets the value of the doublevar allocatable array in Fortran from a Numpy array

        Parameters
        ----------
            doublevar_array : [y,x] Numpy array
        Returns
        -------
            None : Sets the values of self.fobj
        """
        cdef cnp.npy_intp old_shape[2]
        cdef cnp.npy_intp new_shape[2]
        old_shape[0] = self.fobj.doublevar_shape[0]
        old_shape[1] = self.fobj.doublevar_shape[1]
        new_shape[0] = doublevar_array.shape[0]
        new_shape[1] = doublevar_array.shape[1]

        if new_shape[0] != old_shape[0] or new_shape[1] != old_shape[1]:
            raise ValueError(f"Invalid shape for doublevar array: {new_shape} does not match {old_shape}")

        # Get the dimensions of the doublevar_array
        cdef int rows = doublevar_array.shape[0]
        cdef int cols = doublevar_array.shape[1]

        # Manually copy data from the NumPy array to the Fortran array
        cdef double* c_doublevar = self.fobj.doublevar
        for row in range(rows):
            for col in range(cols):
                c_doublevar[row * cols + col] = doublevar_array[row, col]

    def get_stringvar(self):
        """
        A getter method that retrieves the stringvar from Fortran and returns it as a Python string 

        Parameters
        ----------
            None
        Returns
        -------
            string : str
        """
        cdef char *c_string
        c_string = bind_simulation_get_stringvar(self.fobj)
        
        if c_string == NULL:
            return None
        else:
            py_string = PyUnicode_FromString(c_string)
            return py_string

    
    def set_stringvar(self, str string_value):
        """
        A setter method that sets the value of the stringvar in Fortran from a Python string. 

        Parameters
        ----------
            string : str 
                Input string
        Returns
        -------
            None : Sets the values of self.fobj
        """
        cdef const char *c_string
        cdef Py_ssize_t length

        c_string = PyUnicode_AsUTF8AndSize(string_value, &length)
        bind_simulation_set_stringvar(self.fobj, c_string)
    


class Simulation:
    """
    This defines the class that enapsulates the Cython intermediary class into a Python class. In a real-world version of this code, consider implementing this in its own separate Python file to isolate the Python implementation from the Cython one. This allows us to create extendable pure Python objects that make use of our Cython bridge to the Fortran library
    """
    def __init__(self, tuple shape):
        self.c_simulation = _SimulationBind(shape)

    @property
    def doublevar(self):
        return self.c_simulation.fobj.doublevar
    
    @property
    def stringvar(self):
        return self.c_simulation.fobj.stringvar

    def get_doublevar(self):
        return self.c_simulation.get_doublevar()
    
    def set_doublevar(self, doublevar_array):
        self.c_simulation.set_doublevar(doublevar_array)

    def get_stringvar(self):
        return self.c_simulation.get_stringvar()
    
    def set_stringvar(self, string_value):
        self.c_simulation.set_stringvar(string_value)



