// Streaming inference port of microsoft/fastdiar at f48bff53b9a6a0cce4539e94b6def95f609eaea4.
// Copyright (c) Microsoft Corporation and Palabra.ai. Licensed under the MIT License.

#include "network.h"
#include <stdexcept>

namespace fastdiar {
namespace {
/** Keep float32 activation operands on Metal as well as float32 accumulation. */
ggml_tensor * product(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * x) {
    auto * y = mul_mat(ctx, weights, x);
    ggml_prec_set_src(y, GGML_PREC_F32, 1);
    return y;
}
// Spatial tensors are [frequency, time, channels]; shared maps are [channels*frequency, time], channels fastest.
ggml_tensor * map(ggml_context * ctx, ggml_tensor * x) {
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    return ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2]);
}
ggml_tensor * spatial(ggml_context * ctx, ggml_tensor * x, int channels, int mels) {
    return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, x, channels, mels, x->ne[1]), 2, 0, 1, 3));
}
} // namespace
ggml_tensor * Network::conv(Graph & g, const std::string & name, ggml_tensor * x, int sf, int st, bool causal, const NetworkState & state,
                            NetworkOutput & out) const {
    auto * ctx = g.ctx();
    auto * kernel = m_.tensor(name + ".weight");
    const bool dw = name.find("conv_blocks") != std::string::npos &&
                    (name.substr(name.size() - 5) == "conv1" || name.substr(name.size() - 5) == "conv2");
    const int past = causal ? (int)kernel->ne[1] - 1 : 0;
    if (past) {
        const auto it = state.find(name);
        const size_t size = (size_t)(x->ne[0] * past * x->ne[2]);
        const std::vector<float> zeros(it == state.end() ? size : 0, 0);
        const auto & history = it == state.end() ? zeros : it->second;
        if (history.size() != size)
            throw std::invalid_argument("invalid causal history for " + name);
        x = ggml_concat(ctx, g.input(history, x->ne[0], past, x->ne[2]), x, 1);
        auto * tail = ggml_cont(ctx, ggml_view_3d(ctx, x, x->ne[0], past, x->ne[2], x->nb[1], x->nb[2], (x->ne[1] - past) * x->nb[1]));
        out.caches.emplace(name, tail);
        g.output(tail);
    }
    if (!dw && kernel->ne[0] == 1 && kernel->ne[1] == 1) {
        // A pointwise convolution is a matrix product; im2col would duplicate its input on every short step.
        const int64_t f = x->ne[0], time = x->ne[1], in = x->ne[2], out_channels = kernel->ne[3];
        auto * input = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        input = ggml_reshape_2d(ctx, input, in, f * time);
        auto * y = product(ctx, ggml_reshape_2d(ctx, kernel, in, out_channels), input);
        y = ggml_add(ctx, y, m_.tensor(name + ".bias"));
        return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, y, out_channels, f, time), 2, 0, 1, 3));
    }
    ggml_tensor * y;
    if (dw) {
        y = ggml_conv_2d_dw_direct(ctx, kernel, ggml_cont(ctx, x), sf, st, causal ? 1 : 0, 0, 1, 1);
    } else {
        y = ggml_conv_2d_direct(ctx, kernel, ggml_cont(ctx, x), sf, st, causal ? 1 : 0, 0, 1, 1);
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, m_.tensor(name + ".bias"), 1, 1, y->ne[2]));
    }
    return y;
}
ggml_tensor * Network::point(ggml_context * ctx, const std::string & p, ggml_tensor * x) const {
    auto * y = product(ctx, m_.tensor(p + ".weight"), x);
    return ggml_add(ctx, y, m_.tensor(p + ".bias"));
}
ggml_tensor * Network::layer(ggml_context * ctx, const std::string & p, ggml_tensor * x) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, m_.f32("fastdiar.layer_eps")), m_.tensor(p + ".weight")), m_.tensor(p + ".bias"));
}
ggml_tensor * Network::batch(ggml_context * ctx, const std::string & p, ggml_tensor * x, float eps, bool image) const {
    auto * variance = ggml_scale_bias(ctx, m_.tensor(p + ".running_var"), 1.0f, eps);
    auto * scale = ggml_div(ctx, m_.tensor(p + ".weight"), ggml_sqrt(ctx, variance));
    auto * shift = ggml_sub(ctx, m_.tensor(p + ".bias"), ggml_mul(ctx, m_.tensor(p + ".running_mean"), scale));
    if (image) {
        scale = ggml_reshape_3d(ctx, scale, 1, 1, x->ne[2]);
        shift = ggml_reshape_3d(ctx, shift, 1, 1, x->ne[2]);
    }
    return ggml_add(ctx, ggml_mul(ctx, x, scale), shift);
}
ggml_tensor * Network::aggregate(ggml_context * ctx, const std::string & name, const std::vector<ggml_tensor *> & maps) const {
    auto * weights = ggml_soft_max(ctx, m_.tensor(name + ".w"));
    ggml_tensor * sum = nullptr;
    for (size_t i = 0; i < maps.size(); i++) {
        auto * row = ggml_cont(ctx, ggml_view_2d(ctx, weights, 1, weights->ne[1], weights->nb[1], i * sizeof(float)));
        auto * term = ggml_mul(ctx, maps[i], ggml_reshape_1d(ctx, row, weights->ne[1]));
        sum = sum ? ggml_add(ctx, sum, term) : term;
    }
    return sum;
}
ggml_tensor * Network::stage(Graph & g, int i, const std::vector<ggml_tensor *> & maps, const NetworkState & state,
                             NetworkOutput & out) const {
    auto * ctx = g.ctx();
    if (i < 0 || i >= m_.count("fastdiar.stage_count") || maps.size() != (size_t)i + 1)
        throw std::invalid_argument("invalid FASTDIAR stage inputs");
    const std::string prefix = "fastdiar.stage." + std::to_string(i) + ".";
    const int c = m_.size(prefix + "channels"), f = m_.size(prefix + "mels");
    const int64_t frames = maps.front()->ne[1];
    if (frames <= 0 || frames % m_.size(prefix + "time_stride"))
        throw std::invalid_argument("FASTDIAR stage frame count is not divisible by its stride");
    for (auto * map : maps)
        if (map->ne[0] != c * f || map->ne[1] != frames)
            throw std::invalid_argument("inconsistent FASTDIAR shared map shapes");
    const std::string meta = "fastdiar.stage." + std::to_string(i) + ".", b = "backbone.stages." + std::to_string(i);
    const int sf = m_.size(meta + "freq_stride"), st = m_.size(meta + "time_stride");
    auto * x = spatial(ctx, aggregate(ctx, b + ".aggregate", maps), c, f);
    x = conv(g, b + ".downsample", x, sf, st, false, state, out);
    for (int j = 0; j < m_.count(meta + "blocks"); j++) {
        const std::string block = b + ".conv_blocks." + std::to_string(j) + ".conv_block";
        auto * residual = x;
        for (int k = 1; k <= 2; k++) {
            const std::string n = std::to_string(k);
            x = conv(g, block + ".conv" + n, x, 1, 1, true, state, out);
            x = conv(g, block + ".conv" + n + "pw", x, 1, 1, false, state, out);
            if (k == 1)
                x = ggml_relu(ctx, x);
            x = batch(ctx, block + ".bn" + n, x, m_.f32(meta + "bn_eps"), true);
        }
        x = ggml_relu(ctx, ggml_add(ctx, x, residual));
    }
    if (m_.u32(meta + "project"))
        x = batch(ctx, b + ".project.1", conv(g, b + ".project.0", x, 1, 1, false, state, out), m_.f32(meta + "project_eps"), true);
    x = map(ctx, x);
    auto * residual = x;
    x = layer(ctx, b + ".context.red_dim_conv.1", point(ctx, b + ".context.red_dim_conv.0", x));
    x = ggml_gelu_erf(ctx, layer(ctx, b + ".context.tcm.1", point(ctx, b + ".context.tcm.0", x)));
    x = ggml_add(ctx, point(ctx, b + ".context.exp_dim_conv", point(ctx, b + ".context.tcm.3", x)), residual);
    // Nearest upsampling duplicates each column st times, before mixing with the shared maps.
    if (st > 1) {
        x = ggml_reshape_3d(ctx, x, x->ne[0], 1, x->ne[1]);
        x = ggml_repeat(ctx, x, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, x->ne[0], st, x->ne[2]));
        x = ggml_reshape_2d(ctx, x, x->ne[0], frames);
    }
    return x;
}
NetworkOutput Network::forward(Graph & g, const std::vector<float> & features, int64_t frames, const NetworkState & state,
                               bool keep) const {
    const int mel = m_.size("fastdiar.mels"), channels = m_.size("fastdiar.channels"), stride = m_.size("fastdiar.time_stride");
    if (frames <= 0 || frames % stride || features.size() != (size_t)(mel * frames))
        throw std::invalid_argument("log-mel frames must be positive and divisible by the encoder stride");
    auto * ctx = g.ctx();
    NetworkOutput out;
    const auto save = [&](const std::string & name, ggml_tensor * t) {
        if (keep) {
            out.stages.emplace(name, t);
            g.output(t);
        }
        return t;
    };
    auto * x = conv(g, "backbone.stem.0", g.input(features, mel, frames, 1), 1, 1, true, state, out);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    x = layer(ctx, "backbone.stem.1", x);
    x = save("stem", ggml_reshape_2d(ctx, x, channels * mel, frames));
    std::vector<ggml_tensor *> maps{x};
    int c = channels, f = mel;
    for (int i = 0; i < m_.count("fastdiar.stage_count"); i++) {
        x = stage(g, i, maps, state, out);
        maps.push_back(save("stage" + std::to_string(i), x));
        c *= m_.size("fastdiar.stage." + std::to_string(i) + ".freq_stride");
        f /= m_.size("fastdiar.stage." + std::to_string(i) + ".freq_stride");
    }
    x = aggregate(ctx, "backbone.fin_wght1d", maps);
    if (!m_.boolean("fastdiar.time_upsampling"))
        x = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], frames / stride, x->nb[1] * stride, 0));
    x = spatial(ctx, x, c, f);
    if (m_.u32("fastdiar.head_channels"))
        x = conv(g, "backbone.head", x, 1, 1, false, state, out);
    // The final projection flattens channel then frequency, unlike the shared maps' frequency then channel.
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    x = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2]);
    x = save("backbone", x);
    x = point(ctx, "linear", batch(ctx, "bn", x, m_.f32("fastdiar.bn_eps"), false));
    out.embeddings = ggml_l2_norm(ctx, x, m_.f32("fastdiar.normalize_eps"));
    g.output(out.embeddings);
    return out;
}
} // namespace fastdiar
