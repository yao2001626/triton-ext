/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
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
#include "mlir/IR/OpDefinition.h"
#include "pybind11/pybind11.h"
#include <optional>

namespace py = pybind11;
using namespace mlir;

namespace pybind11::detail {
template <> struct type_caster<OpState> : type_caster_base<OpState> {
  struct View : OpState {
    explicit View(Operation *op) : OpState(op) {}
  };
  std::optional<View> view;

  bool load(handle src, bool) {
    // Upstream registers OpState module-locally; Operation casts cross modules.
    if (!py::hasattr(src, "get_operation"))
      return false;
    view.emplace(src.attr("get_operation")().cast<Operation *>());
    value = &*view;
    return true;
  }
};
} // namespace pybind11::detail

PYBIND11_MODULE(_ascend_ir, m) {
  m.def("get_int_attr", [](OpState &op, std::string &name) -> py::object {
    auto ret = op->getAttrOfType<IntegerAttr>(name);
    if (!ret) {
      return py::none();
    }
    return py::cast(ret.getInt());
  });
  m.def("get_program_grid_transforms", [](OpState &op) -> py::object {
    Attribute attribute =
        op->getAttr(mlir::triton::cfg::kProgramGridTransformsAttr);
    if (!attribute)
      return py::none();
    FailureOr<mlir::triton::cfg::ProgramGridTransformContract> contract =
        mlir::triton::cfg::parseProgramGridTransformContract(attribute);
    if (failed(contract))
      throw std::runtime_error("invalid hacc.program_grid_transforms contract");

    py::dict result;
    result["version"] = contract->version;
    result["extent_source"] = "runtime_original_grid";
    py::list hiddenExtentAxes;
    hiddenExtentAxes.append(0);
    hiddenExtentAxes.append(1);
    result["hidden_extent_axes"] = std::move(hiddenExtentAxes);
    py::list hiddenArgumentOrder;
    hiddenArgumentOrder.append("originalGridX");
    hiddenArgumentOrder.append("originalGridY");
    result["hidden_argument_order"] = std::move(hiddenArgumentOrder);
    py::list hiddenArgumentTypes;
    hiddenArgumentTypes.append("i32");
    hiddenArgumentTypes.append("i32");
    result["hidden_argument_types"] = std::move(hiddenArgumentTypes);
    py::list transforms;
    for (const mlir::triton::cfg::ProgramGridTransform &transform :
         contract->transforms) {
      py::dict item;
      item["order"] = transform.order;
      item["kind"] = "ceil_div";
      item["axis"] = transform.axis;
      item["factor"] = transform.factor;
      item["persistent_coverage"] = transform.persistentCoverage;
      item["grid_stride_abi_verified"] = transform.gridStrideAbiVerified;
      transforms.append(std::move(item));
    }
    result["transforms"] = std::move(transforms);
    return std::move(result);
  });
  m.def("remove_attr",
        [](OpState &op, std::string &name) -> void { op->removeAttr(name); });
}
