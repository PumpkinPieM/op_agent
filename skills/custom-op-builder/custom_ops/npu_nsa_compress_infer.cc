#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
ms::Tensor npu_nsa_compress_infer(const ms::Tensor &input, const ms::Tensor &weight, const ms::Tensor &slot_mapping,
                                  int64_t compress_block_size, int64_t compress_stride, int64_t page_block_size,
                                  const std::optional<ms::Tensor> &block_table_opt,
                                  const std::optional<std::vector<int64_t>> &actual_seq_len_opt,
                                  const ms::Tensor &cache) {
  auto out = ms::Tensor(cache.data_type(), cache.shape());
  auto block_table = block_table_opt.value_or(ms::Tensor());
  auto actual_seq_len = std::make_pair(actual_seq_len_opt.value_or(std::vector<int64_t>{}), true);
  std::string layout = "TND";
  char *layout_ptr = const_cast<char *>(layout.data());
  int64_t actual_seq_len_type = 1;

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("NsaCompressWithCache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnNsaCompressWithCache, input, weight, slot_mapping, actual_seq_len,
                                          block_table_opt, layout_ptr, compress_block_size, compress_stride,
                                          actual_seq_len_type, page_block_size, out));
  runner->Run({input, weight, slot_mapping, block_table, cache}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_nsa_compress_infer", PYBOOST_CALLER(1, custom::npu_nsa_compress_infer));
}
}  // namespace custom
