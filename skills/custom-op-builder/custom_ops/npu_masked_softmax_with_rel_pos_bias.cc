#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> ReduceLastDim(const ms::Tensor &x) { auto s = x.shape(); if (!s.empty()) s.pop_back(); return s; }
std::vector<int64_t> LastHalfShape(const ms::Tensor &x) { auto s = x.shape(); if (!s.empty()) s.back() /= 2; return s; }
int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace

ms::Tensor npu_masked_softmax_with_rel_pos_bias(const ms::Tensor &x, const std::optional<ms::Tensor> &atten_mask_opt,
    const ms::Tensor &relative_pos_bias, double scale_value = 1.0, int64_t inner_precision_mode = 0) {
  auto atten_mask = atten_mask_opt.value_or(ms::Tensor()); auto out = ms::Tensor(x.data_type(), x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MaskedSoftmaxWithRelPosBias");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMaskedSoftmaxWithRelPosBias, x, atten_mask_opt, relative_pos_bias, scale_value, inner_precision_mode, out));
  runner->Run({x, atten_mask, relative_pos_bias}, {out}); return out;
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_masked_softmax_with_rel_pos_bias", PYBOOST_CALLER(1, custom::npu_masked_softmax_with_rel_pos_bias)); }

}  // namespace custom
