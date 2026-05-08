#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {

std::vector<int64_t> GetOutputShape(const std::vector<int64_t> &perm, const std::vector<int64_t> &shape,
                                    bool transpose_first) {
  if (transpose_first) {
    return shape;
  }
  std::vector<int64_t> out_shape;
  out_shape.reserve(perm.size());
  for (auto axis : perm) {
    out_shape.push_back(shape.at(static_cast<size_t>(axis)));
  }
  return out_shape;
}

}  // namespace

ms::Tensor npu_confusion_transpose(const ms::Tensor &self, const std::vector<int64_t> &perm,
                                   const std::vector<int64_t> &shape, bool transpose_first) {
  auto out = ms::Tensor(self.data_type(), GetOutputShape(perm, shape, transpose_first));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ConfusionTranspose");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnConfusionTranspose, self, perm, shape, transpose_first, out));
  runner->Run({self}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_confusion_transpose", PYBOOST_CALLER(1, custom::npu_confusion_transpose));
}
}  // namespace custom
