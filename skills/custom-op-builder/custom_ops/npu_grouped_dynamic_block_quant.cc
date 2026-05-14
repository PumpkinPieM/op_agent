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

std::vector<ms::Tensor> npu_grouped_dynamic_block_quant(const ms::Tensor &x, const ms::Tensor &group_list,
    double min_scale = 0.0, const std::string &round_mode = "rint", int64_t dst_type = 291, int64_t row_block_size = 1,
    int64_t col_block_size = 128, int64_t group_list_type = 0) {
  auto y = ms::Tensor(ms::TypeId::kNumberTypeInt8, x.shape()); auto ss = x.shape(); if (ss.size()==2) { ss[0] = ss[0] / row_block_size + group_list.shape()[0]; ss[1] = CeilDiv(ss[1], col_block_size); }
  auto scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ss); auto round = const_cast<char *>(round_mode.c_str());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedDynamicBlockQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedDynamicBlockQuant, x, group_list, min_scale, round, dst_type, row_block_size, col_block_size, group_list_type, y, scale));
  runner->Run({x, group_list}, {y, scale}); return {y, scale};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_grouped_dynamic_block_quant", PYBOOST_CALLER(2, custom::npu_grouped_dynamic_block_quant)); }

}  // namespace custom
