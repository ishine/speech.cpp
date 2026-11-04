// Streaming inference port of microsoft/fastdiar at f48bff53b9a6a0cce4539e94b6def95f609eaea4.
// Copyright (c) Microsoft Corporation and Palabra.ai. Licensed under the MIT License.

#include "frontend.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastdiar {
namespace {
std::vector<float> read(const ModelFile & m, const char * name) {
    auto * t = m.tensor(name);
    std::vector<float> out((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
}
float bounded(float x, float low, float high) {
    return std::max(low, std::min(high, x));
}
} // namespace
Frontend::Frontend(const ModelFile & m)
    : mels_(m.size("fastdiar.mels")), hop_(m.size("fastdiar.hop")), window_(m.size("fastdiar.window")), bins_(m.size("fastdiar.bins")),
      norm_eps_(m.f32("fastdiar.norm_eps")), stft_eps_(m.f32("fastdiar.stft_eps")), cms_eps_(m.f32("fastdiar.cms_eps")),
      variance_floor_(m.f32("fastdiar.variance_floor")), filter_(read(m, "spec.torchfbank.1.flipped_filter")),
      real_(read(m, "spec.torchfbank.2.real_kernel_pt")), imag_(read(m, "spec.torchfbank.2.image_kernel_pt")),
      mel_(read(m, "spec.torchfbank.2.melbanks_pt")) {
    reset();
}
void Frontend::reset() {
    samples_ = frames_ = 0;
    sum_ = sumsq_ = last_ = 0;
    history_.assign(window_ - hop_, 0);
    cms_.assign(mels_, 0);
}
std::vector<float> Frontend::compute(const float * samples, size_t n) {
    if (!samples || !n || n % (size_t)hop_)
        throw std::invalid_argument("frontend audio must contain complete nonempty hops");
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(samples[i]))
            throw std::invalid_argument("audio contains a non-finite sample");
    std::vector<float> emphasized = history_;
    emphasized.reserve(history_.size() + n);
    double prefix = 0, prefixsq = 0;
    float total = sum_, totalsq = sumsq_;
    for (size_t i = 0; i < n; i++) {
        prefix += samples[i];
        const float square = samples[i] * samples[i];
        prefixsq += square;
        // PyTorch's CPU cumsum accumulates each block in double, then adds its previous float32 state.
        total = sum_ + (float)prefix;
        totalsq = sumsq_ + (float)prefixsq;
        const float count = (float)(samples_ + i + 1), mean = total / count;
        const float variance = std::max(totalsq / count - mean * mean, variance_floor_);
        const float normalized = (samples[i] - mean) / (std::sqrt(variance) + norm_eps_);
        emphasized.push_back(last_ * filter_[0] + normalized * filter_[1]);
        last_ = normalized;
    }
    sum_ = total;
    sumsq_ = totalsq;
    samples_ += n;
    history_.assign(emphasized.end() - (window_ - hop_), emphasized.end());
    const size_t count = n / hop_;
    std::vector<float> output(count * mels_), power(bins_);
    std::vector<double> cumulative(mels_, 0);
    for (size_t frame = 0; frame < count; frame++) {
        const float * x = emphasized.data() + frame * hop_;
        for (int b = 0; b < bins_; b++) {
            double r = 0, im = 0;
            for (int j = 0; j < window_; j++) {
                r += (double)x[j] * real_[b * window_ + j];
                im += (double)x[j] * imag_[b * window_ + j];
            }
            const float rf = (float)r, imf = (float)im;
            power[b] = bounded(rf * rf + imf * imf, stft_eps_, 1.0f / stft_eps_);
        }
        for (int mel = 0; mel < mels_; mel++) {
            double sum = 0;
            for (int b = 0; b < bins_; b++)
                sum += (double)power[b] * mel_[mel * bins_ + b];
            const float logarithm = std::log(std::max(bounded((float)sum, stft_eps_, 1.0f / stft_eps_), cms_eps_));
            cumulative[mel] += logarithm;
            const float total = cms_[mel] + (float)cumulative[mel];
            output[frame * mels_ + mel] = logarithm - total / (float)(frames_ + frame + 1);
        }
    }
    for (int mel = 0; mel < mels_; mel++)
        cms_[mel] += (float)cumulative[mel];
    frames_ += count;
    return output;
}
} // namespace fastdiar
