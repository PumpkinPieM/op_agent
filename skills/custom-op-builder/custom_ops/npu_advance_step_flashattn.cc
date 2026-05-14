#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_advance_step_flashattn(const ms::Tensor &input_tokens, const ms::Tensor &sampled_token_ids, const ms::Tensor &input_positions, const ms::Tensor &seq_lens, const ms::Tensor &slot_mapping, const ms::Tensor &block_tables, int64_t num_seqs, int64_t num_queries, int64_t block_size, const std::optional<ms::Tensor> &spec_token_opt = std::nullopt, const std::optional<ms::Tensor> &accepted_num_opt = std::nullopt) {
  auto spec_token = spec_token_opt.value_or(ms::Tensor());
  auto accepted_num = accepted_num_opt.value_or(ms::Tensor());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AdvanceStepV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAdvanceStepV2, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size, spec_token_opt, accepted_num_opt));
  runner->Run({input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, spec_token, accepted_num}, {});
  return {};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_advance_step_flashattn", PYBOOST_CALLER(0, custom::npu_advance_step_flashattn));
}
}  // namespace custom
