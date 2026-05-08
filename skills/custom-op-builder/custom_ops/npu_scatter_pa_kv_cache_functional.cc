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

std::vector<ms::Tensor> npu_scatter_pa_kv_cache_functional(const ms::Tensor &key, const ms::Tensor &value,
    const ms::Tensor &key_cache, const ms::Tensor &value_cache, const ms::Tensor &slot_mapping,
    const std::optional<ms::Tensor> &compress_lens_opt = std::nullopt,
    const std::optional<ms::Tensor> &compress_seq_offsets_opt = std::nullopt,
    const std::optional<ms::Tensor> &seq_lens_opt = std::nullopt) {
  auto key_out = ms::Tensor(key_cache.data_type(), key_cache.shape()); auto value_out = ms::Tensor(value_cache.data_type(), value_cache.shape());
  auto compress_lens = compress_lens_opt.value_or(ms::Tensor()); auto compress_seq_offsets = compress_seq_offsets_opt.value_or(ms::Tensor()); auto seq_lens = seq_lens_opt.value_or(ms::Tensor());
  auto cache_mode = const_cast<char *>("PA_NZ"); auto scatter_mode = const_cast<char *>("None"); auto strides = std::make_pair(std::vector<int64_t>{1,1}, true); auto offsets = std::make_pair(std::vector<int64_t>{0,0}, true);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ScatterPaKvCache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnScatterPaKvCache, key, key_out, slot_mapping, value, value_out, compress_lens_opt, compress_seq_offsets_opt, seq_lens_opt, cache_mode, scatter_mode, strides, offsets));
  runner->Run({key, value, key_cache, value_cache, slot_mapping, compress_lens, compress_seq_offsets, seq_lens}, {key_out, value_out}); return {key_out, value_out};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_scatter_pa_kv_cache_functional", PYBOOST_CALLER(2, custom::npu_scatter_pa_kv_cache_functional)); }

}  // namespace custom
