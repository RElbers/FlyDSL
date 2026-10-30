// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_CAPI_LLVM_H
#define FLYDSL_CAPI_LLVM_H

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Set an LLVM cl::opt value at runtime and return the previous value, read as the given type,
/// through oldValue unless it is null. Returns 0 on success, 1 if the option was not found, 2 on
/// type mismatch, 3 if LLVM rejects the value (the option is then left as it was).
MLIR_CAPI_EXPORTED int flydslSetLLVMOptionBool(const char *name, bool value, bool *oldValue);
MLIR_CAPI_EXPORTED int flydslSetLLVMOptionInt(const char *name, int value, int *oldValue);
MLIR_CAPI_EXPORTED int flydslSetLLVMOptionStr(const char *name, const char *value, char **oldValue);
MLIR_CAPI_EXPORTED void flydslFreeLLVMOptionStr(char *str);

/// Number of times an LLVM cl::opt has been set since it was last reset (0 if never).
/// Returns 0 on success, 1 if the option was not found.
MLIR_CAPI_EXPORTED int flydslGetLLVMOptionOccurrences(const char *name, int *occurrences);

/// Reset an LLVM cl::opt to its default value and clear its occurrence count, so options that
/// only take effect when set (getNumOccurrences() > 0) or that default to unset go back to that.
/// Returns 0 on success, 1 if the option was not found.
MLIR_CAPI_EXPORTED int flydslResetLLVMOption(const char *name);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_CAPI_LLVM_H
