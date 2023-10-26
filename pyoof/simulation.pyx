# cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii
cimport cython
cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset 

cdef extern from "simulation.h":
    ctypedef packed struct c_simulation_type:
        double *doublevar_data
        int doublevar_shape[2]
        char *stringvar

    c_simulation_type* bind_simulation_init(int ny, int nx)
    void bind_simulation_final(c_simulation_type *obj)
    void bind_c2f_string(char* c_string, char* f_string)
    void bind_f2c_string(char* f_string, char* c_string)


def f2c_string(str f_string):
    # Convert Python string to bytes for compatibility with C char*
    f_string_bytes = f_string.encode('utf-8')

    # Allocate a buffer with the size of the Fortran string + 1 (for the null terminator)
    cdef char* c_string = <char*> malloc((len(f_string_bytes) + 1) * sizeof(char))
    if not c_string:
        raise MemoryError("Failed to allocate memory for c_string")

    # Call the Fortran subroutine
    bind_f2c_string(f_string_bytes, c_string)

    # Convert C string (char*) to Python string and free the allocated memory
    result = c_string.decode('utf-8')
    free(c_string)

    return result

cdef char* c2f_string(str c_string):
    # Convert Python string to bytes for compatibility with C char*
    c_string_bytes = c_string.encode('utf-8')
    
    # Assume a maximum Fortran string length for buffer allocation.
    # Adjust this length based on your requirements.
    cdef int max_fortran_length = 512
    cdef char* f_string = <char*> malloc(max_fortran_length * sizeof(char))
    if not f_string:
        raise MemoryError("Failed to allocate memory for f_string")
    
    # Fill buffer with spaces as Fortran strings might be space-padded. 
    # Use 32 as the ASCII value of space, which is more efficient than calling ord(' ')
    memset(f_string, 32, max_fortran_length)
    
    # Call the Fortran subroutine
    bind_c2f_string(c_string_bytes, f_string)
    
    return f_string


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
    
        if len(shape) != 2:
            raise ValueError("Expected a tuple of length 2 for shape")

        self.fobj = bind_simulation_init(shape[0],shape[1])  # <- I'd like to be able to pass the tuple values as separate arguments if possible

        self.fobj.doublevar_shape[0] = shape[0]
        self.fobj.doublevar_shape[1] = shape[1]

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
        # Free memory for stringvar if it was allocated
        if self.fobj.stringvar is not NULL:
            free(self.fobj.stringvar)

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
        doublevar_array = cnp.PyArray_SimpleNewFromData(2, shape, cnp.NPY_FLOAT64, <void*>(self.fobj.doublevar_data))

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
        cdef double* c_doublevar_data = self.fobj.doublevar_data
        for row in range(rows):
            for col in range(cols):
                c_doublevar_data[row * cols + col] = doublevar_array[row, col]
