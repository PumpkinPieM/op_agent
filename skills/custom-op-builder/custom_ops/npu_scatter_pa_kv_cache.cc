#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_scatter_pa_kv_cache(
    const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &key_cache, const ms::Tensor &value_cache,
    const ms::Tensor &slot_mapping, const std::optional<ms::Tensor> &compress_lens_opt = std::nullopt,
    const std::optional<ms::Tensor> &compress_seq_offsets_opt = std::nullopt,
    const std::optional<ms::Tensor> &seq_lens_opt = std::nullopt,
    const std::optional<std::string> &cache_mode_opt = std::nullopt) {
  auto compress_lens = compress_lens_opt.value_or(ms::Tensor());
  auto compress_seq_offsets = compress_seq_offsets_opt.value_or(ms::Tensor());
  auto seq_lens = seq_lens_opt.value_or(ms::Tensor());
  auto cache_mode = cache_mode_opt.value_or("PA_NZ");
  std::string scatter_mode = "None";
  auto strides = std::make_pair(std::vector<int64_t>{1, 1}, true);
  auto offsets = std::make_pair(std::vector<int64_t>{0, 0}, true);

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ScatterPaKvCache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnScatterPaKvCache, key, key_cache, slot_mapping, value, value_cache,
                                          compress_lens_opt, compress_seq_offsets_opt, seq_lens_opt, cache_mode,
                                          scatter_mode, strides, offsets));
  runner->Run({key, value, key_cache, value_cache, slot_mapping, compress_lens, compress_seq_offsets, seq_lens},
              {key_cache, value_cache});
  return {key_cache, value_cache};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_scatter_pa_kv_cache", PYBOOST_CALLER(2, custom::npu_scatter_pa_kv_cache));
}
}  // namespace custom
