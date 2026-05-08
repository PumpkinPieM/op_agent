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

std::vector<ms::Tensor> npu_grouped_dynamic_mx_quant(const ms::Tensor &x, const ms::Tensor &group_index,
    const std::string &round_mode = "rint", int64_t dst_type = 23, int64_t blocksize = 32) {
  auto y = ms::Tensor(ms::TypeId::kNumberTypeUInt8, x.shape()); auto ss = x.shape(); ss.push_back(2); ss[0] = ss[0] / blocksize / 2 + group_index.shape()[0];
  auto scale = ms::Tensor(ms::TypeId::kNumberTypeUInt8, ss); auto round = const_cast<char *>(round_mode.c_str());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedDynamicMxQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedDynamicMxQuant, x, group_index, round, dst_type, blocksize, y, scale));
  runner->Run({x, group_index}, {y, scale}); return {y, scale};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_grouped_dynamic_mx_quant", PYBOOST_CALLER(2, custom::npu_grouped_dynamic_mx_quant)); }

}  // namespace custom
