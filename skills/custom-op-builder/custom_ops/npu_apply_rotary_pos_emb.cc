#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {

constexpr int64_t kLayoutBsndBsh = 1;
constexpr int64_t kLayoutSbnd = 2;
constexpr int64_t kLayoutBnsd = 3;
constexpr int64_t kLayoutTnd = 4;

int64_t GetLayoutValue(const std::string &layout) {
  if (layout == "BNSD") {
    return kLayoutBnsd;
  }
  if (layout == "SBND") {
    return kLayoutSbnd;
  }
  if (layout == "TND") {
    return kLayoutTnd;
  }
  return kLayoutBsndBsh;
}

}  // namespace

std::vector<ms::Tensor> npu_apply_rotary_pos_emb(const ms::Tensor &query, const ms::Tensor &key,
                                                 const ms::Tensor &cos, const ms::Tensor &sin,
                                                 const std::string &layout = "BSH",
                                                 const std::string &rotary_mode = "half") {
  auto query_out = query;
  auto key_out = key;
  auto layout_value = GetLayoutValue(layout);
  auto rotary_mode_value = const_cast<char *>(rotary_mode.c_str());

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ApplyRotaryPosEmbV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyRotaryPosEmbV2, query_out, key_out, cos, sin, layout_value,
                                          rotary_mode_value));
  runner->Run({query, key, cos, sin}, {query_out, key_out});
  return {query_out, key_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_apply_rotary_pos_emb", PYBOOST_CALLER(2, custom::npu_apply_rotary_pos_emb));
}
}  // namespace custom
