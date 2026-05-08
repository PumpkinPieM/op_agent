#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_top_k_top_p_sample(
    const ms::Tensor &logits, const ms::Tensor &top_k, const ms::Tensor &top_p,
    const std::optional<ms::Tensor> &q_opt = std::nullopt, const std::optional<ms::Tensor> &min_ps_opt = std::nullopt,
    double eps = 1e-8, bool is_need_logits = false, int64_t top_k_guess = 32, int64_t ks_max = 1024,
    bool input_is_logits = true, const std::string &post_sample = "qSample") {
  auto q = q_opt.value_or(ms::Tensor());
  auto min_ps = min_ps_opt.value_or(ms::Tensor());
  auto batch = logits.shape().empty() ? 1 : logits.shape()[0];
  auto voc_size = logits.shape().size() < 2 ? 1 : logits.shape()[1];
  bool is_need_sample_result = post_sample == "multiNomial";
  auto logits_select_idx =
      ms::Tensor(ms::TypeId::kNumberTypeInt64, is_need_sample_result ? std::vector<int64_t>{batch, 1}
                                                                     : std::vector<int64_t>{batch});
  auto logits_top_kp_select = ms::Tensor(ms::TypeId::kNumberTypeFloat32, logits.shape());
  auto logits_idx = ms::Tensor(ms::TypeId::kNumberTypeInt64, logits.shape());
  auto logits_sort_masked = ms::Tensor(ms::TypeId::kNumberTypeFloat32, logits.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("TopKTopPSampleV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnTopKTopPSampleV2, logits, top_k, top_p, q_opt, min_ps_opt, eps,
                                          is_need_logits, top_k_guess, ks_max, input_is_logits,
                                          is_need_sample_result, logits_select_idx, logits_top_kp_select, logits_idx,
                                          logits_sort_masked));
  runner->Run({logits, top_k, top_p, q, min_ps}, {logits_select_idx, logits_top_kp_select});
  return {logits_select_idx, logits_top_kp_select};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_top_k_top_p_sample", PYBOOST_CALLER(2, custom::npu_top_k_top_p_sample));
}
}  // namespace custom
