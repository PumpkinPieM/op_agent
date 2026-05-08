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

ms::Tensor npu_sim_exponential_(const ms::Tensor &self, double lambd = 1.0) {
  auto out = ms::Tensor(self.data_type(), self.shape()); int64_t count = 1; for (auto dim : self.shape()) count *= dim;
  constexpr int64_t seed = 1; constexpr int64_t offset = 0; auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("SimThreadExponential");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnSimThreadExponential, out, count, lambd, seed, offset));
  runner->Run({self}, {out}); return out;
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_sim_exponential_", PYBOOST_CALLER(1, custom::npu_sim_exponential_)); }

}  // namespace custom
