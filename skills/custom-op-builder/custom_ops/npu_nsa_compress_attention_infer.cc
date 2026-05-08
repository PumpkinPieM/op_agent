#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
ms::TypeId DTypeFromOptional(const std::optional<int64_t> &dtype, ms::TypeId fallback) {
  if (!dtype.has_value() || dtype.value() < 0) { return fallback; }
  switch (dtype.value()) {
    case 0: return ms::TypeId::kNumberTypeUInt8;
    case 1: return ms::TypeId::kNumberTypeInt8;
    case 2: return ms::TypeId::kNumberTypeInt16;
    case 3: return ms::TypeId::kNumberTypeInt32;
    case 4: return ms::TypeId::kNumberTypeInt64;
    case 5: return ms::TypeId::kNumberTypeFloat16;
    case 6: return ms::TypeId::kNumberTypeFloat32;
    case 27: return ms::TypeId::kNumberTypeBFloat16;
    default: return fallback;
  }
}
std::vector<int64_t> MatmulShape(const ms::Tensor &x1, const ms::Tensor &x2) {
  auto s1 = x1.shape();
  auto s2 = x2.shape();
  if (s1.size() < 2 || s2.size() < 2) { return s1; }
  std::vector<int64_t> out;
  if (s1.size() > 2) { out.insert(out.end(), s1.begin(), s1.end() - 2); }
  out.push_back(s1[s1.size() - 2]);
  out.push_back(s2.back());
  return out;
}
std::vector<int64_t> Conv2dShape(const ms::Tensor &input, const ms::Tensor &weight, const std::vector<int64_t> &strides, const std::vector<int64_t> &pads, const std::vector<int64_t> &dilations) {
  auto x=input.shape(); auto w=weight.shape(); if (x.size()!=4 || w.size()!=4) return x;
  int64_t sh=strides.empty()?1:strides[0], sw=strides.size()>1?strides[1]:sh;
  int64_t ph=pads.empty()?0:pads[0], pw=pads.size()>1?pads[1]:ph;
  int64_t dh=dilations.empty()?1:dilations[0], dw=dilations.size()>1?dilations[1]:dh;
  int64_t oh=(x[2]+2*ph-dh*(w[2]-1)-1)/sh+1; int64_t ow=(x[3]+2*pw-dw*(w[3]-1)-1)/sw+1;
  return {x[0], w[0], std::max<int64_t>(1,oh), std::max<int64_t>(1,ow)};
}
}  // namespace

std::vector<ms::Tensor> npu_nsa_compress_attention_infer(const ms::Tensor & query, const ms::Tensor & key, const ms::Tensor & value, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, int64_t compress_block_size, int64_t compress_stride, const std::optional<std::string> & layout_opt, const std::optional<ms::Tensor> & atten_mask_opt, const std::optional<ms::Tensor> & block_table_opt, const std::optional<ms::Tensor> & topk_mask_opt, const std::optional<std::vector<int64_t>> & actual_seq_qlen_opt, const std::optional<std::vector<int64_t>> & actual_cmp_seq_kvlen_opt, const std::optional<std::vector<int64_t>> & actual_sel_seq_kvlen_opt) {
  auto out=ms::Tensor(query.data_type(), query.shape()); auto topk_indices=ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{query.shape()[0], head_num, select_block_count}); auto layout=layout_opt.value_or("TND"); auto atten_mask=atten_mask_opt.value_or(ms::Tensor()); auto block_table=block_table_opt.value_or(ms::Tensor()); auto topk_mask=topk_mask_opt.value_or(ms::Tensor()); auto actual_seq_qlen=std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true); auto actual_cmp_seq_kvlen=std::make_pair(actual_cmp_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true); auto actual_sel_seq_kvlen=std::make_pair(actual_sel_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("NsaCompressAttentionInfer");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnNsaCompressAttentionInfer, query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, layout, atten_mask_opt, block_table_opt, topk_mask_opt, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen, out, topk_indices));
  runner->Run({query,key,value,atten_mask,block_table,topk_mask}, {out,topk_indices});
  return {out,topk_indices};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_nsa_compress_attention_infer", PYBOOST_CALLER(2, custom::npu_nsa_compress_attention_infer));
}
}  // namespace custom
