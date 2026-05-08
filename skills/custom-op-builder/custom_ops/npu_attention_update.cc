#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_attention_update(const ms::Tensor &attention_out, const ms::Tensor &value, const ms::Tensor &attention_in, const ms::Tensor &kv_index, int64_t num_heads) {
  auto out = ms::Tensor(attention_out.data_type(), attention_out.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AttentionUpdate");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAttentionUpdate, attention_out, value, attention_in, kv_index, num_heads, out));
  runner->Run({attention_out, value, attention_in, kv_index}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_attention_update", PYBOOST_CALLER(1, custom::npu_attention_update));
}
}  // namespace custom
