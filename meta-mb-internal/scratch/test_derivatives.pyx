# distutils: language = c++
# distutils: sources = test_derivatives.cpp

from test_derivatives cimport compute_derivatives
cdef void get_derivatives(mjModel* m, mjData* dmain, mjtNum* deriv):
    compute_derivatives(m, dmain, deriv)
