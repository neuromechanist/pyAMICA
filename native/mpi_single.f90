! Single-rank MPI shim for AMICA (epic #165, phase 1).
!
! amica15.f90 is an MPI + OpenMP + LAPACK program, but pamica always runs it as
! ONE process. Every MPI collective it uses is trivial for a single rank
! (broadcast is a no-op, reduce/gather is a copy, barrier is a no-op), so instead
! of linking a real MPI runtime (Open MPI / MS-MPI), we provide a tiny stub. This
! removes the last runtime dependency after sccn/amica PR #53 removed MKL and AMD
! LibM, so the built binary is self-contained and portable across all release
! targets (macOS, Linux x64/arm64, Windows x64/arm64).
!
! This module supplies ONLY the named constants the source imports via `use mpi`.
! The MPI *subroutines* are deliberately NOT declared here, so `call MPI_BCAST(...)`
! etc. resolve as external procedures (implicit interface) and are provided by
! mpi_single.c. That is what lets the generically-typed BCAST/REDUCE/GATHER calls
! compile under `-fallow-argument-mismatch` without per-type interfaces.
!
! Datatype constants are set to their size in bytes on purpose: the C stub's
! single-rank REDUCE/ALLREDUCE/GATHER copy is `memcpy(recv, send, count*datatype)`.
module mpi
   implicit none
   ! Communicator handle: an opaque integer; COMM_SPLIT returns it unchanged.
   integer, parameter :: MPI_COMM_WORLD = 0
   ! Datatypes -- value == element size in bytes (see note above).
   integer, parameter :: MPI_CHARACTER = 1
   integer, parameter :: MPI_INTEGER = 4
   integer, parameter :: MPI_LOGICAL = 4
   integer, parameter :: MPI_DOUBLE_PRECISION = 8
   ! Reduction ops: unused by a single-rank copy, but must exist as names.
   integer, parameter :: MPI_SUM = 1
   integer, parameter :: MPI_MAX = 2
   integer, parameter :: MPI_MIN = 3
end module mpi
