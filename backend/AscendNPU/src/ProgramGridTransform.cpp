/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "ProgramGridTransform.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <array>
#include <optional>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

constexpr llvm::StringLiteral kVersion = "version";
constexpr llvm::StringLiteral kExtentSource = "extent_source";
constexpr llvm::StringLiteral kHiddenExtentAxes = "hidden_extent_axes";
constexpr llvm::StringLiteral kHiddenArgumentOrder = "hidden_argument_order";
constexpr llvm::StringLiteral kHiddenArgumentTypes = "hidden_argument_types";
constexpr llvm::StringLiteral kTransforms = "transforms";
constexpr llvm::StringLiteral kOrder = "order";
constexpr llvm::StringLiteral kKind = "kind";
constexpr llvm::StringLiteral kAxis = "axis";
constexpr llvm::StringLiteral kFactor = "factor";
constexpr llvm::StringLiteral kPersistentCoverage = "persistent_coverage";
constexpr llvm::StringLiteral kGridStrideAbiVerified =
    "grid_stride_abi_verified";
constexpr llvm::StringLiteral kCeilDiv = "ceil_div";
constexpr llvm::StringLiteral kRuntimeOriginalGrid = "runtime_original_grid";
constexpr llvm::StringLiteral kOriginalGridX = "originalGridX";
constexpr llvm::StringLiteral kOriginalGridY = "originalGridY";
constexpr llvm::StringLiteral kI32 = "i32";

bool hasExactKeys(DictionaryAttr dictionary, ArrayRef<llvm::StringRef> keys) {
  if (dictionary.size() != keys.size())
    return false;
  return llvm::all_of(keys,
                      [&](llvm::StringRef key) { return dictionary.get(key); });
}

std::optional<int64_t> getInteger(DictionaryAttr dictionary,
                                  llvm::StringRef name) {
  auto value = dyn_cast_or_null<IntegerAttr>(dictionary.get(name));
  if (!value)
    return std::nullopt;
  return value.getInt();
}

std::optional<bool> getBoolean(DictionaryAttr dictionary,
                               llvm::StringRef name) {
  auto value = dyn_cast_or_null<BoolAttr>(dictionary.get(name));
  if (!value)
    return std::nullopt;
  return value.getValue();
}

bool hasExactIntegerArray(Attribute attribute, ArrayRef<int64_t> expected) {
  auto values = dyn_cast_or_null<ArrayAttr>(attribute);
  if (!values || values.size() != expected.size())
    return false;
  for (auto [value, wanted] : llvm::zip(values, expected)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer || integer.getInt() != wanted)
      return false;
  }
  return true;
}

bool hasExactStringArray(Attribute attribute,
                         ArrayRef<llvm::StringRef> expected) {
  auto values = dyn_cast_or_null<ArrayAttr>(attribute);
  if (!values || values.size() != expected.size())
    return false;
  for (auto [value, wanted] : llvm::zip(values, expected)) {
    auto string = dyn_cast<StringAttr>(value);
    if (!string || string.getValue() != wanted)
      return false;
  }
  return true;
}

bool isIAT16(const ProgramGridTransform &transform) {
  return transform.axis == 1 && transform.factor == 16 &&
         !transform.persistentCoverage && !transform.gridStrideAbiVerified;
}

bool isPTSM4(const ProgramGridTransform &transform) {
  return transform.axis == 0 && transform.factor == 4 &&
         transform.persistentCoverage && transform.gridStrideAbiVerified;
}

bool isPTSM64(const ProgramGridTransform &transform) {
  return transform.axis == 0 && transform.factor == 64 &&
         transform.persistentCoverage && transform.gridStrideAbiVerified;
}

bool isDotRowTransform(const ProgramGridTransform &transform) {
  return transform.axis == 0 && !transform.persistentCoverage &&
         !transform.gridStrideAbiVerified &&
         (transform.factor == 2 || transform.factor == 4 ||
          transform.factor == 8 || transform.factor == 16 ||
          transform.factor == 32);
}

bool isSupportedSequence(ArrayRef<ProgramGridTransform> transforms) {
  if (transforms.size() == 1)
    return isIAT16(transforms.front()) || isPTSM64(transforms.front()) ||
           isDotRowTransform(transforms.front());
  return transforms.size() == 2 && isIAT16(transforms[0]) &&
         isPTSM4(transforms[1]);
}

} // namespace

FailureOr<ProgramGridTransformContract>
mlir::triton::cfg::parseProgramGridTransformContract(Attribute attribute) {
  auto contract = dyn_cast_or_null<DictionaryAttr>(attribute);
  if (!contract)
    return failure();
  if (!hasExactKeys(contract,
                    {kVersion, kExtentSource, kHiddenExtentAxes,
                     kHiddenArgumentOrder, kHiddenArgumentTypes, kTransforms}))
    return failure();

  std::optional<int64_t> version = getInteger(contract, kVersion);
  auto extentSource = dyn_cast_or_null<StringAttr>(contract.get(kExtentSource));
  constexpr std::array<int64_t, 2> kAxes = {0, 1};
  constexpr std::array<llvm::StringRef, 2> kArgumentOrder = {kOriginalGridX,
                                                             kOriginalGridY};
  constexpr std::array<llvm::StringRef, 2> kArgumentTypes = {kI32, kI32};
  if (!version || *version != kProgramGridTransformsVersion || !extentSource ||
      extentSource.getValue() != kRuntimeOriginalGrid ||
      !hasExactIntegerArray(contract.get(kHiddenExtentAxes), kAxes) ||
      !hasExactStringArray(contract.get(kHiddenArgumentOrder),
                           kArgumentOrder) ||
      !hasExactStringArray(contract.get(kHiddenArgumentTypes), kArgumentTypes))
    return failure();

  auto transforms = dyn_cast_or_null<ArrayAttr>(contract.get(kTransforms));
  if (!transforms || transforms.empty() || transforms.size() > 2)
    return failure();

  ProgramGridTransformContract parsed;
  parsed.version = *version;
  bool seenAxis[2] = {false, false};
  for (auto [expectedOrder, rawTransform] : llvm::enumerate(transforms)) {
    auto transform = dyn_cast<DictionaryAttr>(rawTransform);
    if (!transform ||
        !hasExactKeys(transform, {kOrder, kKind, kAxis, kFactor,
                                  kPersistentCoverage, kGridStrideAbiVerified}))
      return failure();
    std::optional<int64_t> order = getInteger(transform, kOrder);
    auto kind = dyn_cast_or_null<StringAttr>(transform.get(kKind));
    std::optional<int64_t> axis = getInteger(transform, kAxis);
    std::optional<int64_t> factor = getInteger(transform, kFactor);
    std::optional<bool> persistent = getBoolean(transform, kPersistentCoverage);
    std::optional<bool> gridStride =
        getBoolean(transform, kGridStrideAbiVerified);
    if (!order || !kind || !axis || !factor || !persistent || !gridStride ||
        *order != static_cast<int64_t>(expectedOrder) ||
        kind.getValue() != kCeilDiv || *axis < 0 || *axis > 1 || *factor < 2 ||
        *persistent != *gridStride || seenAxis[*axis])
      return failure();
    seenAxis[*axis] = true;
    parsed.transforms.push_back(ProgramGridTransform{
        static_cast<int32_t>(*order), static_cast<int32_t>(*axis), *factor,
        *persistent, *gridStride});
  }
  if (!isSupportedSequence(parsed.transforms))
    return failure();
  return parsed;
}
