#!/usr/bin/env python3
"""Patch the build copy of amica15 for a portable gfortran build (epic #165).

Applies sccn/amica PR #53's portable random_seed fix: the tracked source passes a
fixed size-2 PUT array (only ifort accepts that); gfortran needs an array of its
own SIZE. Idempotent and verified -- if the anchor lines are absent (source
drift), it fails loudly rather than building a subtly wrong binary.

Only the build copy is patched; the tracked amica15.f90 stays the read-only
parity reference.
"""

import sys

OLD_SEED = "call random_seed(PUT = c1 * (myrank+1) * (seed+myrank+1))"
NEW_SEED = """\
! Portable RNG seeding (sccn/amica PR #53): gfortran's random_seed(PUT=...)
! requires an array of the compiler-defined size (query with SIZE=); the original
! passed a fixed size-2 array, which only compiles with ifort. Fill the full-size
! array from the clock, rank, and fixed seed so per-rank streams still differ.
call random_seed(size = nseed)
allocate(seedvec(nseed))
do jj = 1, nseed
   seedvec(jj) = c1 + (jj-1) + (myrank+1)*(seed(mod(jj-1,2)+1) + myrank + 1)
end do
call random_seed(PUT = seedvec)
deallocate(seedvec)"""

HDR_ANCHOR = "integer :: ii, jj, kk, c0, c1, c2"
HDR_ADD = HDR_ANCHOR + "\nINTEGER :: nseed\nINTEGER, ALLOCATABLE :: seedvec(:)"


def patch(path, old, new, marker, label):
    """Replace `old` with `new` in `path`. `marker` is a token unique to the
    patched form; if already present, this is a no-op (idempotent)."""
    with open(path) as f:
        text = f.read()
    if marker in text:
        return  # already patched
    if old not in text:
        sys.exit(f"ERROR: {label} anchor not found in {path}; source may have drifted")
    with open(path, "w") as f:
        f.write(text.replace(old, new, 1))


def main():
    amica15, header = sys.argv[1], sys.argv[2]
    patch(header, HDR_ANCHOR, HDR_ADD, "INTEGER :: nseed", "header seedvec decl")
    patch(amica15, OLD_SEED, NEW_SEED, "call random_seed(size = nseed)", "random_seed")
    print("patched: portable random_seed + seedvec declarations")


if __name__ == "__main__":
    main()
