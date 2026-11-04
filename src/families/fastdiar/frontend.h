#pragma once
#include <cstddef>
#include "model-file.h"
#include <cstdint>
#include <vector>
namespace fastdiar {
/** Causal normalization, pre-emphasis, fixed-kernel STFT, mel projection and cumulative mean subtraction on the CPU. */
class Frontend {
  public:
    explicit Frontend(const ModelFile & model);
    /** Resets every running statistic and sample history for another recording. */
    void reset();
    /** Takes complete hops at 16 kHz and returns [mels, frames], mels fastest. */
    std::vector<float> compute(const float * samples, size_t n);

  private:
    int mels_, hop_, window_, bins_;
    float norm_eps_, stft_eps_, cms_eps_, variance_floor_;
    uint64_t samples_ = 0, frames_ = 0;
    float sum_ = 0, sumsq_ = 0, last_ = 0;
    std::vector<float> filter_, real_, imag_, mel_, history_, cms_;
};
} // namespace fastdiar
