# cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii
cimport cython
from cpython cimport PyUnicode_AsUTF8AndSize, PyUnicode_FromString
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset 
cdef int STRMAX = 512

cdef extern from "simulation.h":
    ctypedef struct c_simulation_type:
        double *doublevar
        int doublevar_shape[2]
        char *stringvar
        int stringvar_len

    c_simulation_type* bind_simulation_init(int ny, int nx)
    void bind_simulation_final(c_simulation_type *obj)
    void bind_simulation_set_stringvar(c_simulation_type *obj, char *c_string)
    char* bind_simulation_get_stringvar(c_simulation_type *obj)


cdef class Simulation:
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

        if self.fobj.doublevar is NULL:
            raise MemoryError("Failed to allocate component variable 'doublevar' in the Fortran object.")
        else:
            print("The component variable 'doublevar' was allocated successfuly in the Fortran object")

        print("fobj points to:", <unsigned long>self.fobj)
        print("stringvar points to:", <unsigned long>self.fobj.stringvar)

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
        Returns
        -------
            Deallocates the fobj component variables from Fortran.
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
            # Don't forget to free the C string if allocated in Fortran
            #free(c_string)
            return py_string

    
    def set_stringvar(self, str string):
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
        cdef char *c_string
        cdef Py_ssize_t length

        c_string = PyUnicode_AsUTF8AndSize(string, &length)
        bind_simulation_set_stringvar(self.fobj, c_string)
