
  !---------------------------------------------------------------------------!
  !                                                                           !
  !     MLatom: a Package for Atomistic Simulations with Machine Learning     !
  !                               Version 2.0                                 !
  !                           http://mlatom.com/                              !
  !                                                                           !
  !                  Copyright (c) 2013-2021 Pavlo O. Dral                    !
  !                           http://dr-dral.com/                             !
  !                                                                           !
  ! All rights reserved. No part of MLatom may be used, published or          !
  ! redistributed without written permission by Pavlo Dral.                   ! 
  !                                                                           !
  ! The above copyright notice and this permission notice shall be included   !
  ! in all copies or substantial portions of the Software.                    !
  !                                                                           !
  ! The software is provided "as is", without warranty of any kind, express   !
  ! or implied, including but not limited to the warranties of                !
  ! merchantability, fitness for a particular purpose and noninfringement. In !
  ! no event shall the authors or copyright holders be liable for any claim,  !
  ! damages or other liability, whether in an action of contract, tort or     !
  ! otherwise, arising from, out of or in connection with the software or the !
  ! use or other dealings in the software.                                    !
  !                                                                           !
  !                                Cite as:                                   !
  ! Pavlo O. Dral, J. Comput. Chem. 2019, 40, 2339-2347                       !
  ! Pavlo O. Dral, Fuchun Ge, Bao-Xin Xue, Yi-Fan Hou, Max Pinheiro Jr,       !
  ! Jianxing Huang, Mario Barbatti, Top. Curr. Chem. 2021, 379, 27            !
  !                                                                           !
  ! Pavlo O. Dral, Bao-Xin Xue, Fuchun Ge, Yi-Fan Hou,                        !
  ! MLatom: A Package for Atomistic Simulations with Machine Learning         !
  ! version 2.0, Xiamen University, Xiamen, China, 2013-2021                  !
  !                                                                           !  
  !---------------------------------------------------------------------------!
  
!===========================================================
program MLatomF
!===========================================================

  !---------------------------------------------------------------------------!
  !                                                                           !
  ! This is a Fortran + OpenMP part of MLatom, previously called MLler        !
  !                                                                           !
  !                        Computer-Chemie-Centrum                            !
  !                     University Erlangen-Nuremberg                         !
  !                         Naegelsbachstrasse 25                             !
  !                        91052 Erlangen, Germany                            !
  !                September 10th, 2013 - October 31, 2013                    !
  !                                                                           !
  !               Max-Planck-Institut fuer Kohlenforschung                    !
  !                         Kaiser-Wilhelm-Platz 1                            !
  !                  45470 Muelheim an der Ruhr, Germany                      !
  !                November 1, 2013 - September 30, 2019                      !
  !                                                                           !
  !            College of Chemistry and Chemical Engineering                  !
  !                          Xiamen University                                !
  !                        Xiamen 361005, China                               !
  !                        From October 1, 2019                               !
  !                                                                           !
  !---------------------------------------------------------------------------!

!===========================================================
  use analysis,      only : analyze
  use MLmodel,       only : getX, useMLmodel, createMLmodel, estAccMLmodel
  use optionsModule, only : option
  use sampling,      only : sample
  implicit none
  ! Local variables
  integer*8 :: dateStart(1:8) ! Time and date, when MLatomF starts

  ! Initialize MLatomF
  call initializeMLatomF(dateStart(1:8))
  
  ! Perform required calculations
  if     (option%XYZ2X) then
    call getX()
  elseif (option%analyze) then
    call analyze()
  elseif (option%sample) then
    call getX()
    call sample()
  elseif (option%useMLmodel) then
    call useMLmodel()
  elseif (option%createMLmodel) then
    call createMLmodel()
  elseif (option%estAccMLmodel) then
    call estAccMLmodel()
  end if

  ! Terminate MLatomF
  call finishMLatomF(dateStart(1:8))

end program MLatomF
!==============================================================================

!==============================================================================
subroutine initializeMLatomF(dateStart)
!==============================================================================
  use MLatomFInfo,   only : header
  use optionsModule, only : writeOptions, getOptions, option
  use timing,        only : timeTaken
  implicit none
  ! Agruments
  integer*8, intent(out) :: dateStart(1:8) ! Time and date, when MLatomF starts

  ! Get options
  call getOptions()
  
  ! Write header
  call header()

  ! Start time
  call date_and_time(values=dateStart)
  write(6,'(/" MLatomF started on",1x,2(i2.2,"."),i4.4," at ",2(i2.2,":"),i2.2/)') &
             dateStart(3),dateStart(2),dateStart(1),dateStart(5),dateStart(6),dateStart(7)
            !  Day         Month         Year        Hour          Minute       Second

  ! Write options
  call writeOptions(6)

end subroutine initializeMLatomF
!==============================================================================

!==============================================================================
subroutine finishMLatomF(dateStart)
!==============================================================================
  use A_KRR,   only : cleanUp_KRR
  use dataset, only : cleanUp_dataset
  use timing,  only : timeTaken
  implicit none
  ! Agruments
  integer*8, intent(in) :: dateStart(1:8) ! Time and date, when MLatomF starts
  ! Local variables
  integer*8 :: dateFinish(1:8) ! Time and date, when MLatomF finishes
  
  ! Free up all allocated memory
  call cleanUp_KRR()
  call cleanUp_dataset()

  ! Termination time
  write(6,'(a)') ''
  call timeTaken(dateStart,'Wall-clock time:')
  call date_and_time(values=dateFinish)
  write(6,'(/" MLatomF terminated on",1x,2(i2.2,"."),i4.4," at ",2(i2.2,":"),i2.2/)') &
             dateFinish(3),dateFinish(2),dateFinish(1),dateFinish(5),dateFinish(6),dateFinish(7)
            !  Day           Month         Year          Hour          Minute       Second

end subroutine finishMLatomF
!==============================================================================

