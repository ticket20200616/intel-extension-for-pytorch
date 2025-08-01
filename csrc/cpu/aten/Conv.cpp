#include "Conv.h"
#include <torch/all.h>
#include "WeightPack.h"
#include "autocast/autocast_mode.h"
#include "ideep/IDeepConversions.h"
#include "utils/utils.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(causal_conv1d_update_kernel_stub);
IPEX_DEFINE_DISPATCH(causal_conv1d_fn_kernel_stub);
std::vector<int64_t> calc_conv_output_size(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = kernel_size[0];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (kernel_size[d] - 1) + 1;
    output_size[d] =
        (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

void convolution_kernel_output(
    const at::Tensor& input,
    const ideep::tensor& mkldnn_weight,
    const ideep::tensor& mkldnn_bias,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
  TORCH_CHECK(
      (IS_CONTIGUOUS_ANY(input)) && (IS_CONTIGUOUS_ANY(output)),
      "input and output are need contiguous tensor for "
      "convolution_kernel_output");
  const ideep::tensor mkldnn_input_ = itensor_view_from_dense(input);
  ideep::tensor mkldnn_input = mkldnn_input_;
  // The following code forces the 3D input to channels last, which is a
  // temporary workaround before channels last 1D is formally supported in
  // PyTorch.
  if (mkldnn_input_.ndims() == 3 &&
      !mkldnn_input_.get_desc().is_channels_last()) {
    ideep::tensor mkldnn_input_conv1d{
        mkldnn_input_.get_desc().to_format(ideep::format_tag::nwc)};
    mkldnn_input_conv1d.feed_from(mkldnn_input_);
    mkldnn_input = mkldnn_input_conv1d;
  }
  auto output_sizes = output.sizes();

  ideep::tensor mkldnn_output = itensor_view_from_dense(output);

  if (mkldnn_bias.is_empty()) {
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        {output_sizes.begin(), output_sizes.end()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  } else {
    ideep::convolution_forward::compute(
        mkldnn_input,
        mkldnn_weight,
        mkldnn_bias,
        {output_sizes.begin(), output_sizes.end()},
        mkldnn_output,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  }
}

at::Tensor convolution_kernel(
    const at::Tensor& input,
    const ideep::tensor& mkldnn_weight,
    const ideep::tensor& mkldnn_bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr,
    at::MemoryFormat memory_format) {
  // Base convolution kernel, this base kernel will not change input's format,
  // so make sure you has make process the input's format before call this
  // function, the output wil has same format with input.
  // TODO: the input will be actively converted to channels last format
  // after the 5-D tensor supports channels last format.
  TORCH_CHECK(
      IS_CONTIGUOUS_ANY(input),
      "input is need to a contiguous tensor for convolution_kernel");
  auto kernel_size = mkldnn_weight.get_dims();
  auto input_size = input.sizes();
  std::vector<int64_t> output_sizes =
      calc_conv_output_size(input_size, kernel_size, padding, stride, dilation);

  at::Tensor output;
  if (input.dim() != 3) {
    output =
        at::empty(output_sizes, input.options().memory_format(memory_format));
  } else {
    // This a temporary workaround before channels last 1D is formally supported
    // in PyTorch. We will force to return nwc output.
    TORCH_WARN(
        "Conv1d channels last support will be deprecated in Intel® Extension for PyTorch* 2.9.");
    std::vector<int64_t> output_strides = {
        (output_sizes[1] * output_sizes[2]), 1, output_sizes[1]};
    output = at::empty_strided(output_sizes, output_strides, input.options());
  }

  convolution_kernel_output(
      input,
      mkldnn_weight,
      mkldnn_bias,
      output,
      stride,
      padding,
      dilation,
      groups,
      attr);
  return output;
}

at::Tensor convolution_forward_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context,
    c10::optional<at::IntArrayRef> kernel_size,
    c10::optional<at::IntArrayRef> padding,
    c10::optional<at::IntArrayRef> stride,
    c10::optional<at::IntArrayRef> dilation,
    c10::optional<bool> weight_channels_last) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::convolution_forward_impl\n");
#endif
  RECORD_FUNCTION(
      "torch_ipex::convolution_forward_impl", c10::ArrayRef<c10::IValue>({}));

  return reinterpret_cast<IpexConvolutionOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run(input, ideep::attr_t(torch_ipex::fpmath_mode));
}

at::Tensor convolution_backward_input(
    at::IntArrayRef input_size,
    const at::Tensor& grad_output,
    const ideep::tensor& mkldnn_weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    bool weight_use_channels_last) {
  TORCH_CHECK(
      input_size.size() == 4 || input_size.size() == 5,
      "Only support 2d or 3d convolution for convolution_backward_input");

  const ideep::tensor mkldnn_grad_output = itensor_view_from_dense(grad_output);
  bool is_channels_last_contiguous =
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d);

  auto memory_format = at::MemoryFormat::Contiguous;
  if (is_channels_last_contiguous) {
    if (input_size.size() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }

  auto grad_input =
      at::empty(input_size, grad_output.options().memory_format(memory_format));
  ideep::tensor mkldnn_grad_input;
  if (is_channels_last_contiguous) {
    mkldnn_grad_input = itensor_view_from_dense(grad_input);
  }

  ideep::convolution_backward_data::compute(
      mkldnn_grad_output,
      mkldnn_weight,
      input_size.vec(),
      mkldnn_grad_input,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups,
      ideep::attr_t(torch_ipex::fpmath_mode));

  if (is_channels_last_contiguous) {
    return grad_input;
  } else {
    return mkldnn_to_dense(new_with_itensor_mkldnn(
                               std::move(mkldnn_grad_input),
                               c10::optTypeMetaToScalarType(
                                   grad_output.options().dtype_opt()),
                               grad_output.options().device_opt()))
        .contiguous(memory_format);
  }
}

std::tuple<at::Tensor, at::Tensor> convolution_backward_weights(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const ideep::tensor::desc& packed_weight_desc,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "Only support 2d or 3d convolution for convolution_backward_weights");
  const ideep::tensor mkldnn_grad_output = itensor_view_from_dense(grad_output);
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input);

  bool is_channels_last_contiguous =
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      grad_output.is_contiguous(at::MemoryFormat::ChannelsLast3d);

  auto grad_weight = at::empty_like(weight, grad_output.options());
  at::Tensor grad_bias;
  ideep::tensor mkldnn_grad_weight, mkldnn_grad_bias;
  if (grad_output.scalar_type() == at::ScalarType::Float) {
    mkldnn_grad_weight.init(
        packed_weight_desc, grad_weight.template data_ptr<float>());
  } else if (grad_output.scalar_type() == at::ScalarType::BFloat16) {
    mkldnn_grad_weight.init(
        packed_weight_desc, grad_weight.template data_ptr<c10::BFloat16>());
  } else {
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Half,
        "Only support bfloat16, float16 and float for convolution_backward_weights");
    mkldnn_grad_weight.init(
        packed_weight_desc, grad_weight.template data_ptr<c10::Half>());
  }

  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
    mkldnn_grad_bias = itensor_view_from_dense(grad_bias);
    ideep::convolution_backward_weights::compute(
        mkldnn_input,
        mkldnn_grad_output,
        packed_weight_desc.get_dims(),
        mkldnn_grad_weight,
        mkldnn_grad_bias,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        ideep::attr_t(torch_ipex::fpmath_mode));
  } else {
    ideep::convolution_backward_weights::compute(
        mkldnn_input,
        mkldnn_grad_output,
        packed_weight_desc.get_dims(),
        mkldnn_grad_weight,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        ideep::attr_t(torch_ipex::fpmath_mode));
  }
  return std::make_tuple(grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_kernel(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& at_weight,
    const ideep::tensor& mkldnn_weight,
    const ideep::tensor& mkldnn_bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const bool weight_channels_last,
    std::array<bool, 3> output_mask) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::convolution_backward\n");
#endif
  RECORD_FUNCTION(
      "torch_ipex::convolution_backward", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "Only support 2d or 3d convolution for convolution_backward");

  bool use_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d ||
      weight_channels_last;

  auto memory_format = at::MemoryFormat::Contiguous;
  if (use_channels_last) {
    if (input.dim() == 4) {
      memory_format = at::MemoryFormat::ChannelsLast;
    } else {
      memory_format = at::MemoryFormat::ChannelsLast3d;
    }
  }
  auto grad_output_ = grad_output.contiguous(memory_format);

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = convolution_backward_input(
        input.sizes(),
        grad_output_,
        mkldnn_weight,
        padding,
        stride,
        dilation,
        groups,
        output_mask[2],
        weight_channels_last);
  }
  if (output_mask[1] || output_mask[2]) {
    auto input_ = input.contiguous(memory_format);
    std::tie(grad_weight, grad_bias) = convolution_backward_weights(
        grad_output_,
        input_,
        at_weight,
        mkldnn_weight.get_desc(),
        padding,
        stride,
        dilation,
        groups,
        output_mask[2]);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask,
    const at::Tensor& op_context,
    c10::optional<bool> weight_channels_last) {
  return reinterpret_cast<IpexConvolutionOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_backward(input, grad_output, output_mask);
}

at::Tensor IPEXConvolutionOp::_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context,
    c10::optional<at::IntArrayRef> kernel_size,
    c10::optional<at::IntArrayRef> padding,
    c10::optional<at::IntArrayRef> stride,
    c10::optional<at::IntArrayRef> dilation,
    c10::optional<bool> weight_channels_last) {
  at::AutoDispatchBelowADInplaceOrView g;
  RECORD_FUNCTION(
      "IPEXConvolutionOp::_forward", c10::ArrayRef<c10::IValue>({}));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::convolution_forward", "")
                       .typed<decltype(convolution_forward)>();
  return op.call(
      input,
      weight,
      bias_opt,
      op_context,
      kernel_size,
      padding,
      stride,
      dilation,
      weight_channels_last);
}

at::Tensor IPEXConvolutionOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context,
    c10::optional<at::IntArrayRef> kernel_size,
    c10::optional<at::IntArrayRef> padding,
    c10::optional<at::IntArrayRef> stride,
    c10::optional<at::IntArrayRef> dilation,
    c10::optional<bool> weight_channels_last) {
  RECORD_FUNCTION("IPEXConvolutionOp::forward", c10::ArrayRef<c10::IValue>({}));

  at::AutoDispatchBelowADInplaceOrView g;
  ctx->saved_data["op_context"] = op_context;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] =
      bias_opt.has_value() && bias_opt.value().requires_grad() ? true : false;
  ctx->saved_data["bias_opt"] = bias_opt;
  ctx->saved_data["weight_channels_last"] = weight_channels_last;
  ctx->save_for_backward({input, weight});

  return _forward(
      input,
      weight,
      bias_opt,
      op_context,
      kernel_size,
      padding,
      stride,
      dilation,
      weight_channels_last);
}

torch::autograd::variable_list IPEXConvolutionOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  RECORD_FUNCTION(
      "IPEXConvolutionOp::backward", c10::ArrayRef<c10::IValue>({}));

  auto op_context = ctx->saved_data["op_context"].toTensor();
  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  auto bias_opt = ctx->saved_data["bias_opt"].toOptional<at::Tensor>();
  auto weight_channels_last =
      ctx->saved_data["weight_channels_last"].toOptional<bool>();
  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor weight = saved[1];
  at::Tensor grad_input, grad_weight, grad_bias;
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_ipex::convolution_backward", "")
          .typed<decltype(convolution_backward)>();
  std::tie(grad_input, grad_weight, grad_bias) = op.call(
      input,
      weight,
      bias_opt,
      grad_outputs[0],
      output_mask,
      op_context,
      weight_channels_last);
  return {
      grad_input,
      grad_weight,
      grad_bias,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

at::Tensor convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context,
    c10::optional<at::IntArrayRef> kernel_size,
    c10::optional<at::IntArrayRef> padding,
    c10::optional<at::IntArrayRef> stride,
    c10::optional<at::IntArrayRef> dilation,
    c10::optional<bool> weight_channels_last) {
  if (at::GradMode::is_enabled()) {
    return IPEXConvolutionOp::apply(
        input,
        weight,
        bias_opt,
        op_context,
        kernel_size,
        padding,
        stride,
        dilation,
        weight_channels_last);
  }
  return IPEXConvolutionOp::_forward(
      input,
      weight,
      bias_opt,
      op_context,
      kernel_size,
      padding,
      stride,
      dilation,
      weight_channels_last);
}

/**
 * Official Python implementation: causal_conv1d_update_ref:
 * https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py#L206
 * @param hidden_states (batch, dim) or (batch, dim, seqlen)
 * @param conv_states (batch, dim, state_len), where state_len >= width - 1
 * @param conv_weights (dim, width)
 * @param conv_bias (dim,)
 * @param silu_activation If true, apply the SiLU activation function.
 * @param cache_seqlens (batch,) or None
 * @return (hidden_states, conv_states)
 */
std::tuple<at::Tensor, at::Tensor> causal_conv1d_update(
    const at::Tensor& hidden_states,
    const at::Tensor& conv_states,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation,
    const c10::optional<at::Tensor>& cache_seqlens) {
  RECORD_FUNCTION("causal_conv1d_update", c10::ArrayRef<c10::IValue>({}));
  return causal_conv1d_update_kernel_stub(
      kCPU,
      hidden_states,
      conv_states,
      conv_weights,
      conv_bias,
      silu_activation,
      cache_seqlens);
}

/**
 * Official Python implementation: causal_conv1d_ref:
 * https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py#L133
 * @param x (batch, dim, seqlen)
 * @param conv_weights (dim, width)
 * @param conv_bias (dim,)
 * @param initial_states (batch, dim, width - 1)
 * @param final_states_out (batch, dim, width - 1)
 * @param silu_activation If true, apply the SiLU activation function.
 * @return (out, final_states_out)
 * out: (batch, dim, seqlen)
 * final_states_out: (batch, dim, width - 1)
 */
std::tuple<at::Tensor, at::Tensor> causal_conv1d_fn(
    const at::Tensor& x,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    const c10::optional<at::Tensor>& initial_states,
    const c10::optional<at::Tensor>& final_states_out,
    bool silu_activation) {
  RECORD_FUNCTION("causal_conv1d_fn", c10::ArrayRef<c10::IValue>({}));
  return causal_conv1d_fn_kernel_stub(
      kCPU,
      x,
      conv_weights,
      conv_bias,
      initial_states,
      final_states_out,
      silu_activation);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

at::Tensor convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& op_context,
    c10::optional<at::IntArrayRef> kernel_size,
    c10::optional<at::IntArrayRef> padding,
    c10::optional<at::IntArrayRef> stride,
    c10::optional<at::IntArrayRef> dilation,
    c10::optional<bool> weight_channels_last) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::convolution_forward", "")
                       .typed<decltype(convolution_forward)>();
  auto target_type = get_autocast_dtype();

  // TODO: make check weight dtype should be float for training case.
  return op.call(
      cpu_cached_cast(target_type, input),
      weight,
      bias_opt,
      op_context,
      kernel_size,
      padding,
      stride,
      dilation,
      weight_channels_last);
}

} // namespace autocast
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "convolution_forward(Tensor input, Tensor weight, Tensor? bias, "
      "Tensor W_prepack, int[]? kernel_size, int[]? padding, int[]? stride, int[]? dilation, bool? weight_channels_last) -> Tensor");
  m.impl(
      "convolution_forward",
      c10::DispatchKey::Autograd,
      torch_ipex::cpu::convolution_forward);
  m.impl(
      "convolution_forward",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::convolution_forward);
  m.impl(
      "convolution_forward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::convolution_forward_impl);
  m.def(
      "causal_conv1d_update(Tensor hidden_states, Tensor conv_states, Tensor conv_weights, Tensor? conv_bias, bool silu_activation, Tensor? cache_seqlens=None) -> (Tensor, Tensor)");
  m.impl(
      "causal_conv1d_update",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::causal_conv1d_update);
  m.def(
      "causal_conv1d_fn(Tensor x, Tensor conv_weights, Tensor? conv_bias, Tensor? initial_states, Tensor? final_states_out, bool silu_activation) -> (Tensor, Tensor)");
  m.impl(
      "causal_conv1d_fn",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::causal_conv1d_fn);
  // bw
  m.def(
      "convolution_backward(Tensor input, Tensor weight, Tensor? bias, Tensor grad_output, bool[3] out_mask, "
      "Tensor W_prepack, bool? weight_channels_last) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "convolution_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::convolution_backward);
}
} // namespace
