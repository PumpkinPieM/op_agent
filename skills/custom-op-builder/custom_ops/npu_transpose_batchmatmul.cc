#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kRank3 = 3;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    MS_LOG(EXCEPTION) << message;
  }
}

const std::vector<int64_t> &DefaultPermX() {
  static const std::vector<int64_t> value{0, 1, 2};
  return value;
}

const std::vector<int64_t> &DefaultPermY() {
  static const std::vector<int64_t> value{1, 0, 2};
  return value;
}

std::vector<int64_t> OutputShape(const ms::Tensor &x, const ms::Tensor &weight,
                                 const std::optional<ms::Tensor> &scale_opt,
                                 const std::vector<int64_t> &perm_x1, const std::vector<int64_t> &perm_x2,
                                 int64_t batch_split_factor) {
  const auto &x_shape = x.shape();
  const auto &weight_shape = weight.shape();
  Check(x_shape.size() == kRank3, "npu_transpose_batchmatmul: input must be rank 3.");
  Check(weight_shape.size() == kRank3, "npu_transpose_batchmatmul: weight must be rank 3.");
  Check(perm_x1.size() == kRank3 && perm_x2.size() == kRank3, "npu_transpose_batchmatmul: perm lists must have size 3.");
  Check(batch_split_factor > 0, "npu_transpose_batchmatmul: batch_split_factor must be positive.");

  const int64_t m_dim = x_shape[perm_x1[1]];
  const int64_t batch_dim = x_shape[perm_x1[0]];
  const int64_t n_dim = weight_shape[perm_x2[2]];
  if (scale_opt.has_value()) {
    return {m_dim, 1, batch_dim * n_dim};
  }
  if (batch_split_factor > 1) {
    Check((batch_dim * n_dim) % batch_split_factor == 0,
          "npu_transpose_batchmatmul: batch_dim * n_dim must be divisible by batch_split_factor.");
    return {batch_split_factor, m_dim, batch_dim * n_dim / batch_split_factor};
  }
  return {m_dim, batch_dim, n_dim};
}
}  // namespace

std::vector<ms::Tensor> npu_transpose_batchmatmul(
    const ms::Tensor &x, const ms::Tensor &weight, const std::optional<ms::Tensor> &bias_opt,
    const std::optional<ms::Tensor> &scale_opt, const std::optional<std::vector<int64_t>> &perm_x1_opt,
    const std::optional<std::vector<int64_t>> &perm_x2_opt, const std::optional<std::vector<int64_t>> &perm_y_opt,
    const std::optional<int64_t> &batch_split_factor_opt) {
  auto bias = bias_opt.value_or(ms::Tensor());
  auto scale = scale_opt.value_or(ms::Tensor());
  auto perm_x1_value = perm_x1_opt.value_or(DefaultPermX());
  auto perm_x2_value = perm_x2_opt.value_or(DefaultPermX());
  auto perm_y_value = perm_y_opt.value_or(DefaultPermY());
  auto perm_x1 = std::make_pair(perm_x1_value, true);
  auto perm_x2 = std::make_pair(perm_x2_value, true);
  auto perm_y = std::make_pair(perm_y_value, true);
  int64_t batch_split_factor_i64 = batch_split_factor_opt.value_or(1);
  int32_t batch_split_factor = static_cast<int32_t>(batch_split_factor_i64);
  int8_t cube_math_type = 0;
  auto out_dtype = scale_opt.has_value() ? ms::TypeId::kNumberTypeInt8 : x.data_type();
  auto out = ms::Tensor(out_dtype, OutputShape(x, weight, scale_opt, perm_x1_value, perm_x2_value, batch_split_factor_i64));

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("TransposeBatchMatMul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnTransposeBatchMatMul, x, weight, bias_opt, scale_opt, perm_x1, perm_x2,
                                          perm_y, cube_math_type, batch_split_factor, out));
  runner->Run({x, weight, bias, scale}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_transpose_batchmatmul", PYBOOST_CALLER(1, custom::npu_transpose_batchmatmul));
}
}  // namespace custom
