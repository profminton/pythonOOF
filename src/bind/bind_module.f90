!! Copyright 2023 - David Minton
!! This file is part of PyOOF
!! PyOOF is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
!! as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
!! pyoof is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
!! of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
!! You should have received a copy of the GNU General Public License along with pyoof. 
!! If not, see: https://www.gnu.org/licenses. 


module bind_module
   !! author: David A. Minton
   !!
   !! This module defines the set of routines that connect the Cython code to the Fortran. Because Fortran derived types with
   !! allocatable arrays and type-bound procedures are not interoperable with C, you have to write a set of functions that allow
   !! you to access the Fortran structures. On the Fortran side, you can make use of the full suite of OOF features from F2003+ 
   !! e.g. classes, inheritence, polymorphism, type-bound procedures, etc., but any communication back to Python must be done 
   !! via a set of binding functions. 
   !! 
   !! The following implementation was adapted from _Modern Fortran Explained: Incorporating Fortran 2018_ by Metcalf, Reid, & 
   !! Cohen (see Fig. 19.8)
   use iso_c_binding
   use globals
   use simulation
   implicit none

contains

  type(c_ptr) function bind_simulation_init(ny,nx) bind(c)
      !! author: David A. Minton
      !!
      !! This function is used to initialize the simulation_type derived type object in Fortran and return a pointer to the object 
      !! that can be used as a struct in C, and ultimately to the Python class object via Cython.
      implicit none
      ! Arguments
      integer(I4B),          value   :: ny, nx   !! The dimensions of the array to create. Note, this expects row-major ordering 
      ! Internals
      type(simulation_type), pointer :: sim_ptr  !! A pointer to the simulation type variable that will be passed to Cython
      integer(I4B) :: i
      type(c_ptr) :: c_ptr_var

      nullify(sim_ptr)
      allocate(sim_ptr)
      call sim_ptr%allocate(nx, ny) 
      bind_simulation_init = c_loc(sim_ptr)

      write(*,*) 
      write(*,*) "***************** FORTRAN *******************"
      write(*,*) "Inside bind_simulation_init" 
      write(*,*) "stringvar       : ",trim(adjustl(sim_ptr%stringvar))
      write(*,*) "len(stringvar)  : ",len(sim_ptr%stringvar)
      write(*,*) "doublevar       : "
      do i = 1,ny
         write(*,*) sim_ptr%doublevar(:,i)
      end do
      write(*,*) "shape(doublevar): ", shape(sim_ptr%doublevar)
      c_ptr_var = c_loc(sim_ptr)
      write(*,*) "Address of sim_ptr:", transfer(c_ptr_var, 0_C_INTPTR_T)
      c_ptr_var = c_loc(sim_ptr%stringvar)
      write(*,*) "Address of sim_ptr%stringvar:", transfer(c_ptr_var, 0_C_INTPTR_T)
      
      write(*,*) "*************** END FORTRAN *****************"
      write(*,*) 

      return
   end function bind_simulation_init


   subroutine bind_simulation_final(sim) bind(c)
      !! author: David A. Minton
      !!
      !! This subroutine is used to deallocate the pointer that links the C struct to the Fortran derived type object. 
      implicit none
      ! Arguments
      type(c_ptr), intent(in), value :: sim
      ! Internals
      type(simulation_type), pointer :: sim_ptr

      call c_f_pointer(sim, sim_ptr)
      deallocate(sim_ptr)

      return
   end subroutine bind_simulation_final


   subroutine bind_c2f_string(c_string, f_string) bind(c)
      !! author: David A. Minton
      !!
      !! This subroutine is used to convert C style strings into Fortran. This allows one to pass Python strings as arguments to 
      !! Fortran functions.
      implicit none
      ! Arguments
      character(kind=c_char), dimension(:), intent(in)  :: c_string
      character(kind=c_char), dimension(:), intent(out) :: f_string
      ! Internals
      integer :: i
      character(len=STRMAX,kind=c_char) :: tmp_string

      i=1
      tmp_string = ''
      do while(c_string(i) /= c_null_char .and. i <= STRMAX)
         tmp_string(i:i) = c_string(i)
         i=i+1
      end do

      if (i > 1) then
         f_string = trim(tmp_string)
      else
         f_string = ""
      end if

      return
   end subroutine bind_c2f_string


   subroutine bind_f2c_string(f_string, c_string) bind(c)
      !! author: David A. Minton
      !!
      !! This subroutine is used to convert Fortran style strings to C. This allows the Python module to read strings that were 
      !! created in Fortran procedures.
      implicit none
      ! Arguments
      character(kind=c_char), dimension(:), intent(in)  :: f_string
      character(kind=c_char), dimension(:), intent(out) :: c_string
      ! Internals
      integer :: i

      i = 1
      do while (f_string(i) /= c_null_char)
         c_string(i) = f_string(i)
         i = i + 1
      end do
      c_string(i) = c_null_char
      
      return
   end subroutine bind_f2c_string

   
end module bind_module