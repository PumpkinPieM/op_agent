#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
using TensorList = std::vector<ms::Tensor>;
}  // namespace

std::vector<ms::Tensor> npu_attention_update(const TensorList &lse, const TensorList &local_out, int64_t update_type) {
  auto out = ms::Tensor(local_out[0].data_type(), local_out[0].shape());
  auto lse_out = ms::Tensor(lse[0].data_type(), lse[0].shape());
  std::optional<ms::Tensor> lse_out_opt = std::nullopt;
  if (update_type == 1) {
    lse_out_opt = lse_out;
  }

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AttentionUpdate");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAttentionUpdate, lse, local_out, update_type, out, lse_out_opt));

  TensorList inputs;
  inputs.insert(inputs.end(), lse.begin(), lse.end());
  inputs.insert(inputs.end(), local_out.begin(), local_out.end());
  runner->Run(inputs, {out, lse_out});
  return {out, lse_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_attention_update", PYBOOST_CALLER(2, custom::npu_attention_update));
}
}  // namespace custom
