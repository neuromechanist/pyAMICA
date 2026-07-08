/* Portable vector-math shim for building amica15 without a vendor math library
 * (epic #84, phase #85).
 *
 * amica15.f90's non-MKL branch calls AMD LibM's vectorized exp/log
 * (`vrda_exp`/`vrda_log`); its MKL branch uses Intel VML (`vdExp`/`vdLn`).
 * Neither vendor library is present when building with plain gfortran + LAPACK
 * (e.g. on Apple Silicon, or a generic Linux box), so the link fails on
 * `vrda_exp_`/`vrda_log_`.
 *
 * These are simple element-wise ops, so this shim provides them as loops over
 * standard libm exp/log. gfortran calls them by reference with a trailing
 * underscore (`call vrda_exp(n, x, y)` -> `vrda_exp_(int* n, double* x,
 * double* y)`), which is the ABI matched here. Scalar libm exp/log are
 * correctly-rounded/IEEE-accurate (vendor SIMD transcendental libraries such as
 * AMD LibM typically trade some accuracy for throughput); the cost here is speed,
 * not accuracy -- a vendor SIMD math library could compute the exp/log-heavy
 * E-step somewhat faster (documented as a caveat in README.md).
 */
#include <math.h>

void vrda_exp_(const int *n, const double *x, double *y) {
    int i, m = *n;
    for (i = 0; i < m; i++) y[i] = exp(x[i]);
}

void vrda_log_(const int *n, const double *x, double *y) {
    int i, m = *n;
    for (i = 0; i < m; i++) y[i] = log(x[i]);
}
