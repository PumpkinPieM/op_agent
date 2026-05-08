#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_gather_pa_kv_cache(const ms::Tensor &key_cache, const ms::Tensor &value_cache, const ms::Tensor &block_tables, const ms::Tensor &seq_lens, const ms::Tensor &key, const ms::Tensor &value, const std::optional<ms::Tensor> &seq_offset_opt = std::nullopt, bool is_seq_lens_cumsum = false) {
  auto seq_offset = seq_offset_opt.value_or(ms::Tensor());
  auto key_out = ms::Tensor(key.data_type(), key.shape());
  auto value_out = ms::Tensor(value.data_type(), value.shape());
  const char *cache_mode = "Norm";
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GatherPaKvCache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGatherPaKvCache, key_cache, value_cache, block_tables, seq_lens,
                                          key_out, value_out, seq_offset_opt, cache_mode, is_seq_lens_cumsum));
  runner->Run({key_cache, value_cache, block_tables, seq_lens, key, value, seq_offset}, {key_out, value_out});
  return {key_out, value_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_gather_pa_kv_cache", PYBOOST_CALLER(2, custom::npu_gather_pa_kv_cache));
}
}  // namespace custom
