// Workaround for glibc 2.41+ / CUDA 12.9 incompatibility
// glibc 2.41 added sinpi/cospi with noexcept(true), conflicting with CUDA's declarations
#pragma once

#ifdef __CUDACC__
// Prevent glibc from declaring sinpi/cospi/sinpif/cospif
#define __GLIBC_USE_IEC_60559_BFP_EXT 0
#define __GLIBC_USE_IEC_60559_FUNCS_EXT 0
#endif

