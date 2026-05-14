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

ms::Tensor npu_scatter_pa_cache(const ms::Tensor &key, const ms::Tensor &slot_mapping,
    const std::optional<ms::Tensor> &compress_lens_opt = std::nullopt,
    const std::optional<ms::Tensor> &compress_seq_offsets_opt = std::nullopt,
    const std::optional<ms::Tensor> &seq_lens_opt = std::nullopt, const ms::Tensor &key_cache = ms::Tensor()) {
  auto out = ms::Tensor(key_cache.data_type(), key_cache.shape()); auto compress_lens = compress_lens_opt.value_or(ms::Tensor());
  auto compress_seq_offsets = compress_seq_offsets_opt.value_or(ms::Tensor()); auto seq_lens = seq_lens_opt.value_or(ms::Tensor()); auto mode = const_cast<char *>("Norm");
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ScatterPaCache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnScatterPaCache, key, out, slot_mapping, compress_lens_opt, compress_seq_offsets_opt, seq_lens_opt, mode));
  runner->Run({key, key_cache, slot_mapping, compress_lens, compress_seq_offsets, seq_lens}, {out}); return out;
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_scatter_pa_cache", PYBOOST_CALLER(1, custom::npu_scatter_pa_cache)); }

}  // namespace custom
