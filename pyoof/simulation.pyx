# cython: language_level=3, c_string_type=unicode, c_string_encoding=ascii
cimport cython
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset 

cdef extern from "simulation.h":
    ctypedef packed struct c_simulation_type:
        double *doublevar_data
        int doublevar_shape[2]
 
    c_simulation_type* bind_simulation_init(int gridsize)
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

cdef class Simulation:
    cdef c_simulation_type* c_sim

    def __cinit__(self, int gridsize):
        self.c_sim = bind_simulation_init(gridsize)
        if self.c_sim is NULL:
            raise MemoryError("Failed to allocate simulation object")

        # Manually set the shape values based on gridsize
        self.c_sim.doublevar_shape[0] = gridsize
        self.c_sim.doublevar_shape[1] = gridsize

    def __dealloc__(self):
        if self.c_sim is not NULL:
            bind_simulation_final(self.c_sim)

    def get_doublevar(self):
        # Retrieve the shape of the elevation data
        cdef np.npy_intp shape[2]
        shape[0] = self.c_sim.doublevar_shape[0]
        shape[1] = self.c_sim.doublevar_shape[1]

        # Create a NumPy array from the Fortran array
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] doublevar_array
        doublevar_array = np.PyArray_SimpleNewFromData(2, shape, np.NPY_FLOAT64, <void*>(self.c_sim.doublevar_data))

        return doublevar_array
 
    def set_doublevar(self, np.ndarray[np.float64_t, ndim=2] doublevar_array):
        cdef np.npy_intp old_shape[2]
        cdef np.npy_intp new_shape[2]
        old_shape[0] = self.c_sim.doublevar_shape[0]
        old_shape[1] = self.c_sim.doublevar_shape[1]
        new_shape[0] = doublevar_array.shape[0]
        new_shape[1] = doublevar_array.shape[1]

        if new_shape[0] != old_shape[0] or new_shape[1] != old_shape[1]:
            raise ValueError(f"Invalid shape for doublevar array: {new_shape} does not match {old_shape}")

        # Get the dimensions of the doublevar_array
        cdef int rows = doublevar_array.shape[0]
        cdef int cols = doublevar_array.shape[1]

        # Manually copy data from the NumPy array to the Fortran array
        cdef double* c_doublevar_data = self.c_sim.doublevar_data
        for row in range(rows):
            for col in range(cols):
                c_doublevar_data[row * cols + col] = doublevar_array[row, col]
