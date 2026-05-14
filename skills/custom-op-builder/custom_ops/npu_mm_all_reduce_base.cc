#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> OutputShape(const ms::Tensor &x1, const ms::Tensor &x2) {
  auto shape = x1.shape();
  auto x2_shape = x2.shape();
  if (shape.size() < 2 || x2_shape.size() < 2) {
    return shape;
  }
  shape.back() = x2_shape.back();
  return shape;
}

ms::TypeId OutputDType(const ms::Tensor &x1, const std::optional<ms::Tensor> &dequant_scale_opt,
                       const std::optional<int64_t> &y_dtype_opt) {
  if (!dequant_scale_opt.has_value()) {
    return x1.data_type();
  }
  if (!y_dtype_opt.has_value()) {
    return dequant_scale_opt.value().data_type() == ms::TypeId::kNumberTypeBFloat16
               ? ms::TypeId::kNumberTypeBFloat16
               : ms::TypeId::kNumberTypeFloat16;
  }
  switch (y_dtype_opt.value()) {
    case 5:
      return ms::TypeId::kNumberTypeFloat16;
    case 6:
      return ms::TypeId::kNumberTypeFloat32;
    case 27:
      return ms::TypeId::kNumberTypeBFloat16;
    default:
      return ms::TypeId::kNumberTypeFloat16;
  }
}

bool IsIntegralType(ms::TypeId dtype) {
  return dtype == ms::TypeId::kNumberTypeInt8 || dtype == ms::TypeId::kNumberTypeUInt8 ||
         dtype == ms::TypeId::kNumberTypeInt16 || dtype == ms::TypeId::kNumberTypeInt32 ||
         dtype == ms::TypeId::kNumberTypeInt64;
}
}  // namespace

ms::Tensor npu_mm_all_reduce_base(
    const ms::Tensor &x1, const ms::Tensor &x2, const std::string &hcom,
    const std::optional<std::string> &reduce_op_opt = std::nullopt,
    const std::optional<ms::Tensor> &bias_opt = std::nullopt,
    const std::optional<ms::Tensor> &antiquant_scale_opt = std::nullopt,
    const std::optional<ms::Tensor> &antiquant_offset_opt = std::nullopt,
    const std::optional<ms::Tensor> &x3_opt = std::nullopt,
    const std::optional<ms::Tensor> &dequant_scale_opt = std::nullopt,
    const std::optional<ms::Tensor> &pertoken_scale_opt = std::nullopt,
    const std::optional<ms::Tensor> &comm_quant_scale_1_opt = std::nullopt,
    const std::optional<ms::Tensor> &comm_quant_scale_2_opt = std::nullopt,
    int64_t antiquant_group_size = 0, int64_t comm_turn = 0,
    const std::optional<std::vector<int64_t>> &group_sizes_opt = std::nullopt,
    const std::optional<int64_t> &y_dtype_opt = std::nullopt,
    const std::optional<int64_t> &x1_dtype_opt = std::nullopt,
    const std::optional<int64_t> &x2_dtype_opt = std::nullopt,
    const std::optional<int64_t> &dequant_scale_dtype_opt = std::nullopt,
    const std::optional<int64_t> &pertoken_scale_dtype_opt = std::nullopt, int64_t comm_quant_mode = 0) {
  (void)antiquant_group_size;
  (void)group_sizes_opt;
  (void)y_dtype_opt;
  (void)x1_dtype_opt;
  (void)x2_dtype_opt;
  (void)dequant_scale_dtype_opt;
  (void)pertoken_scale_dtype_opt;
  (void)comm_quant_mode;

  auto bias = bias_opt.value_or(ms::Tensor());
  auto x3 = x3_opt.value_or(ms::Tensor());
  auto antiquant_scale = antiquant_scale_opt.value_or(ms::Tensor());
  auto antiquant_offset = antiquant_offset_opt.value_or(ms::Tensor());
  auto dequant_scale = dequant_scale_opt.value_or(ms::Tensor());
  auto pertoken_scale = pertoken_scale_opt.value_or(ms::Tensor());
  auto comm_quant_scale_1 = comm_quant_scale_1_opt.value_or(ms::Tensor());
  auto comm_quant_scale_2 = comm_quant_scale_2_opt.value_or(ms::Tensor());
  auto reduce_op = reduce_op_opt.value_or("sum");
  constexpr int64_t kStopOnFailure = 1;
  auto out = ms::Tensor(OutputDType(x1, dequant_scale_opt, y_dtype_opt), OutputShape(x1, x2));

  const bool x1_integral = IsIntegralType(x1.data_type());
  const bool x2_integral = IsIntegralType(x2.data_type());
  if (!x1_integral && !x2_integral) {
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(x3_opt.has_value() ? "MatmulAllReduceV2"
                                                                                   : "MatmulAllReduce");
    if (x3_opt.has_value()) {
      runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulAllReduceV2, x1, x2, bias_opt, x3, hcom, reduce_op, comm_turn,
                                              kStopOnFailure, out));
      runner->Run({x1, x2, bias, x3}, {out});
    } else {
      runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulAllReduce, x1, x2, bias_opt, hcom, reduce_op, comm_turn,
                                              kStopOnFailure, out));
      runner->Run({x1, x2, bias}, {out});
    }
    return out;
  }

  if (x1_integral && x2_integral) {
    if (!dequant_scale_opt.has_value()) {
      throw std::runtime_error("npu_mm_all_reduce_base: int8 quantized path requires dequant_scale");
    }
    if (comm_quant_scale_1_opt.has_value() != comm_quant_scale_2_opt.has_value()) {
      throw std::runtime_error("npu_mm_all_reduce_base: comm_quant_scale_1 and comm_quant_scale_2 must be both set");
    }
    if (comm_quant_scale_1_opt.has_value()) {
      auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulAllReduceV3");
      runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulAllReduceV3, x1, x2, bias_opt, x3, dequant_scale,
                                              pertoken_scale, comm_quant_scale_1, comm_quant_scale_2, hcom, reduce_op,
                                              comm_turn, kStopOnFailure, out));
      runner->Run({x1, x2, bias, x3, dequant_scale, pertoken_scale, comm_quant_scale_1, comm_quant_scale_2}, {out});
      return out;
    }
    if (pertoken_scale_opt.has_value()) {
      auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulAllReduceV2");
      runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulAllReduceV2, x1, x2, bias_opt, x3, dequant_scale,
                                              pertoken_scale, hcom, reduce_op, comm_turn, kStopOnFailure, out));
      runner->Run({x1, x2, bias, x3, dequant_scale, pertoken_scale}, {out});
      return out;
    }
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulAllReduce");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulAllReduce, x1, x2, bias_opt, x3, dequant_scale, hcom,
                                            reduce_op, comm_turn, kStopOnFailure, out));
    runner->Run({x1, x2, bias, x3, dequant_scale}, {out});
    return out;
  }

  if (!x1_integral && x2_integral) {
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("WeightQuantMatmulAllReduce");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnWeightQuantMatmulAllReduce, x1, x2, bias_opt, antiquant_scale,
                                            antiquant_offset, x3, hcom, reduce_op, comm_turn, kStopOnFailure,
                                            antiquant_group_size, out));
    runner->Run({x1, x2, bias, antiquant_scale, antiquant_offset, x3}, {out});
    return out;
  }

  throw std::runtime_error("npu_mm_all_reduce_base: unsupported x1/x2 dtype combination");
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_mm_all_reduce_base", PYBOOST_CALLER(1, custom::npu_mm_all_reduce_base));
}
}  // namespace custom
