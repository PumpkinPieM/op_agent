#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_mrope(const ms::Tensor &positions, const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos_sin_cache, int64_t head_size, const std::optional<std::vector<int64_t>> &mrope_section_opt = std::nullopt, const std::optional<std::string> &rotary_mode_opt = std::nullopt, const std::optional<std::string> &cache_mode_opt = std::nullopt) {
  auto mrope_section = std::make_pair(mrope_section_opt.value_or(std::vector<int64_t>{}), true);
  auto rotary_mode = rotary_mode_opt.value_or("half");
  bool is_neox_style = rotary_mode == "half";
  auto query_out = ms::Tensor(query.data_type(), query.shape());
  auto key_out = ms::Tensor(key.data_type(), key.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("RopeWithSinCosCacheV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnRopeWithSinCosCache, positions, query, key, cos_sin_cache,
                                          mrope_section, head_size, is_neox_style, query_out, key_out));
  runner->Run({positions, query, key, cos_sin_cache}, {query_out, key_out});
  return {query_out, key_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_mrope", PYBOOST_CALLER(2, custom::npu_mrope));
}
}  // namespace custom
