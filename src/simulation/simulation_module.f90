!! Copyright 2023 - David Minton
!! This file is part of PyOOF
!! PyOOF is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
!! as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
!! pyoof is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
!! of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
!! You should have received a copy of the GNU General Public License along with pyoof. 
!! If not, see: https://www.gnu.org/licenses. 

module simulation
   use globals

   type  :: simulation_type
      real(DP), dimension(:,:), allocatable :: doublevar  ! simulation elevation
   contains
      procedure :: allocate   => simulation_allocate   !! Allocate the allocatable components of the class
      procedure :: deallocate => simulation_deallocate !! Deallocate all allocatable components of the class
      final     ::               simulation_final      !! Finalizer (calls deallocate)
   end type simulation_type


contains

   subroutine simulation_allocate(self, gridsize)
      !! author: David A. Minton
      !!
      !! Allocate the allocatable components of the class
      implicit none
      ! Arguments
      class(simulation_type), intent(inout) :: self     !! Simulation object
      integer(I4B),        intent(in)    :: gridsize !! Size of the grid

      allocate(self%doublevar(gridsize,gridsize))

      self%doublevar(:,:) = -1.0_DP

      return
   end subroutine simulation_allocate


   subroutine simulation_deallocate(self) 
      !! author: David A. Minton
      !!
      !! Deallocate the allocatable components of the class
      implicit none
      ! Arguments
      class(simulation_type), intent(inout) :: self     !! Simulation object

      deallocate(self%doublevar)

      return
   end subroutine simulation_deallocate


   subroutine simulation_final(self)
      !! author: David A. Minton
      !!
      !! Finalizer for the simulation object
      implicit none
      ! Arguments
      type(simulation_type), intent(inout) :: self

      call self%deallocate()
      return
   end subroutine simulation_final


end module simulation