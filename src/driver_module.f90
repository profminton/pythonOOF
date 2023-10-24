!! Copyright 2023 - David Minton
!! This file is part of Pyoof
!! Pyoof is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
!! as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
!! pyoof is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
!! of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
!! You should have received a copy of the GNU General Public License along with pyoof. 
!! If not, see: https://www.gnu.org/licenses. 

module driver
   contains

   module subroutine pyoof_driver()

      !! author: David A. Minton
      !!
      !! This just verifies that we can create an executable using the library.
      implicit none

      write(*,*) "Use Python!"
      return
   end subroutine pyoof_driver

end module driver
