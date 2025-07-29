#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/extension.h>

constexpr auto kVirtD = c10::DeviceType::PrivateUse1;

namespace dbg {
void _print_prefix() { std::cout << "[C++]: "; }
void print_tensor(const at::Tensor &ten) {
  _print_prefix();
  auto tensor_data = ten.data_ptr<float>();
  int64_t num_elements = ten.numel();
  std::cout << "Device: " << ten.device() << ", Tensor values: [";
  for (int64_t i = 0; i < num_elements; ++i) {
    std::cout << tensor_data[i];
    if (i < num_elements - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}
} // namespace dbg
// print current function name
#define print_fn()                                                             \
  do {                                                                         \
    dbg::_print_prefix();                                                      \
    std::cout << __FUNCTION__ << std::endl;                                    \
  } while (0)

namespace kn {
at::Tensor empty_strided_for_virtd(c10::IntArrayRef size,
                                   c10::IntArrayRef stride,
                                   c10::optional<at::ScalarType> dtype,
                                   c10::optional<at::Layout> layout,
                                   c10::optional<at::Device> device,
                                   c10::optional<bool> pin_memory) {
  print_fn();

  // 1. 获取参数
  auto scalar_type = dtype.value();
  auto *allocator = c10::GetAllocator(kVirtD);

  // 2. 计算存储大小
  auto nbytes = at::detail::computeStorageNbytes(size, stride,
                                                 at::elementSize(scalar_type));

  // 3. 创建存储
  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), nbytes, allocator,
      /*resizable=*/true);

  // 4. 创建TensorImpl
  auto tensor_impl = c10::make_intrusive<at::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      scalarTypeToTypeMeta(scalar_type)); // 必须转换为TypeMeta

  // 5. 设置形状和步长
  tensor_impl->set_sizes_and_strides(size, stride);

  // 6. 返回Tensor
  return at::Tensor(std::move(tensor_impl));
}

at::Tensor empty_for_virtd(c10::IntArrayRef size,
                           c10::optional<at::ScalarType> dtype,
                           c10::optional<at::Layout> layout,
                           c10::optional<at::Device> device,
                           c10::optional<bool> pin_memory,
                           c10::optional<at::MemoryFormat> memory_format) {
  print_fn();
  auto *allocator = c10::GetAllocator(kVirtD);
  auto storage = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      at::detail::computeStorageNbytesContiguous(
          size, c10::elementSize(dtype.value_or(at::kFloat))),
      allocator,
      /*resizable=*/true);

  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      scalarTypeToTypeMeta(dtype.value()));
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  return tensor;
}

at::Tensor &add_out_for_virtd(const at::Tensor &self, const at::Tensor &other,
                              const at::Scalar &alpha, at::Tensor &out) {
  print_fn();
  auto self_data = self.data_ptr<float>();
  auto other_data = other.data_ptr<float>();
  auto out_data = out.data_ptr<float>();
  dbg::print_tensor(self);
  dbg::print_tensor(other);
  for (int64_t i = 0; i < self.numel(); ++i) {
    out_data[i] = self_data[i] + other_data[i] * alpha.to<float>();
  }
  dbg::print_tensor(out);
  return out;
}

at::Tensor copy_from_for_virtd(const at::Tensor &src, const at::Tensor &self,
                               bool non_blocking) {
  print_fn();
  TORCH_CHECK(!non_blocking, "Async copy not supported");
  TORCH_CHECK(self.device().type() != src.device().type(),
              "Copy between same device not supported");
  TORCH_CHECK(src.data_ptr() != self.data_ptr(),
              "Source and destination memory overlap");
  std::memcpy(self.data_ptr(), src.data_ptr(), src.nbytes());
  return self;
}
} // namespace kn

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  print_fn();
  m.impl("empty.memory_format", &kn::empty_for_virtd);
  m.impl("add.out", &kn::add_out_for_virtd);
  m.impl("_copy_from", &kn::copy_from_for_virtd);
  m.impl("empty_strided", &kn::empty_strided_for_virtd);
}

namespace disp {
/*
逻辑上的多设备管理，如virtd逻辑上有5个设备，那么定义张量时：
torch.tensor([1.0, 2.0, 3.0], device='virtd:2') # 有效
torch.tensor([1.0, 2.0, 3.0], device='virtd:10') # 无效，有效范围0-4
为了简化目前只有一个逻辑设备
*/
class VirtDDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
public:
  VirtDDeviceGuardImpl() { print_fn(); }
  c10::DeviceType type() const override { return kVirtD; }
  c10::Device exchangeDevice(c10::Device device) const override {
    return device;
  }
  c10::Device getDevice() const override { return c10::Device(kVirtD, 0); }
  void setDevice(c10::Device device) const override { print_fn(); }
  void uncheckedSetDevice(c10::Device device) const noexcept override {}
  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }
  c10::Stream exchangeStream(c10::Stream s) const { return s; }
  c10::DeviceIndex deviceCount() const noexcept { return 1; }
};

/*
PyTorch默认只有CPU/CUDA的内存管理
自定义设备（如FPGA、NPU等）有独立的内存空间管理方式，需要实现这个类
*/
class VirtDAllocator : public c10::Allocator {
public:
  static void deallocate(void *p) {
    print_fn();
    free(p);
  }
  c10::DataPtr allocate(size_t nbytes) override {
    print_fn();
    void *data = malloc(nbytes);
    return {data, data, VirtDAllocator::deallocate, kVirtD};
  }
  void copy_data(void *dest, const void *src, std::size_t count) const {
    print_fn();
    std::memcpy(dest, src, count);
  }
};

void register_custom_device() {
  print_fn();
  torch::utils::set_requires_device_init(kVirtD, false);

  c10::register_privateuse1_backend("virtd");

  static VirtDDeviceGuardImpl impl;
  c10::impl::DeviceGuardImplRegistrar(kVirtD, &impl);

  static VirtDAllocator alloc;
  SetAllocator(kVirtD, &alloc);
}
} // namespace disp

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  print_fn();
  m.def("init", disp::register_custom_device);
}