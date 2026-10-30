// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// LLVM cl::opt runtime control.
// Linked into libFlyPythonCAPI.so to share the cl::opt registry with MLIR.
// Uses addOccurrence() so getNumOccurrences()-guarded options work correctly.

#include "LlvmConfig/llvm.h"

#include "llvm/Support/CommandLine.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace llvm;

namespace {

/// Find a registered cl::Option by name.  Returns nullptr if not found.
cl::Option *findOption(const std::string &name) {
  auto options = cl::getRegisteredOptions();
  auto it = options.find(name);
  return it == options.end() ? nullptr : it->second;
}

/// Check whether \p opt is a bool-valued option (ValueOptional / ValueDisallowed)
/// vs. a value-required option (int / string / enum).
/// LLVM is built without RTTI, so dynamic_cast is unavailable; we use the
/// ValueExpected flag as a proxy for the underlying C++ type.
bool isBoolOption(cl::Option *opt) { return opt->getValueExpectedFlag() != cl::ValueRequired; }

/// Set \p opt to \p value as if it were given on the command line. Returns 0, or 3 if LLVM rejects
/// the value; the option then keeps its value, and one that was unset is unset again.
int addOccurrence(cl::Option *opt, const char *name, StringRef value) {
  unsigned occurrences = opt->getNumOccurrences();
  if (!opt->addOccurrence(1, name, value))
    return 0;
  if (occurrences == 0)
    opt->reset();
  return 3;
}

} // namespace

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" {

__attribute__((visibility("default"))) int flydslSetLLVMOptionBool(const char *name, bool value,
                                                                   bool *oldValue) {
  cl::Option *opt = findOption(name);
  if (!opt)
    return 1;
  if (!isBoolOption(opt))
    return 2; // type mismatch: option is not bool
  auto *typed = static_cast<cl::opt<bool> *>(opt);
  if (oldValue)
    *oldValue = typed->getValue();
  return addOccurrence(opt, name, value ? "true" : "false");
}

__attribute__((visibility("default"))) int flydslSetLLVMOptionInt(const char *name, int value,
                                                                  int *oldValue) {
  cl::Option *opt = findOption(name);
  if (!opt)
    return 1;
  if (isBoolOption(opt))
    return 2; // type mismatch: option is bool, not int
  auto *typed = static_cast<cl::opt<int> *>(opt);
  if (oldValue)
    *oldValue = typed->getValue();
  return addOccurrence(opt, name, std::to_string(value));
}

__attribute__((visibility("default"))) int
flydslSetLLVMOptionStr(const char *name, const char *value, char **oldValue) {
  cl::Option *opt = findOption(name);
  if (!opt)
    return 1;
  if (isBoolOption(opt))
    return 2; // type mismatch: option is bool, not string
  auto *typed = static_cast<cl::opt<std::string> *>(opt);
  if (oldValue) {
    std::string prev = typed->getValue();
    *oldValue = static_cast<char *>(std::malloc(prev.size() + 1));
    std::memcpy(*oldValue, prev.c_str(), prev.size() + 1);
  }
  return addOccurrence(opt, name, value);
}

__attribute__((visibility("default"))) void flydslFreeLLVMOptionStr(char *str) { std::free(str); }

__attribute__((visibility("default"))) int flydslGetLLVMOptionOccurrences(const char *name,
                                                                          int *occurrences) {
  cl::Option *opt = findOption(name);
  if (!opt)
    return 1;
  *occurrences = opt->getNumOccurrences();
  return 0;
}

__attribute__((visibility("default"))) int flydslResetLLVMOption(const char *name) {
  cl::Option *opt = findOption(name);
  if (!opt)
    return 1;
  opt->reset();
  return 0;
}

} // extern "C"
