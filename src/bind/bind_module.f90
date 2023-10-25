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

   type(c_ptr) function bind_simulation_init(gridsize) bind(c)
      integer(I4B), value :: gridsize
      type(simulation_type), pointer :: sim_ptr

      allocate(sim_ptr)
      call sim_ptr%allocate(gridsize) 
      bind_simulation_init = c_loc(sim_ptr)
   end function bind_simulation_init

   subroutine bind_simulation_final(surf) bind(c)
      type(c_ptr), intent(in), value :: surf
      type(simulation_type), pointer :: sim_ptr

      call c_f_pointer(surf, sim_ptr)
      deallocate(sim_ptr)
   end subroutine bind_simulation_final


end module bind_module