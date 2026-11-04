#pragma once
#include <cstdint>
#include <optional>
#include <vector>
namespace fastdiar {
/** Confirmed speech in samples at 16 kHz: [start, end). */
struct Interval {
    int64_t start, end;
};
/**
 * The causal hysteresis of microsoft/fastdiar's StreamingVAD over Silero's probability of each 512-sample frame, with
 * its defaults: threshold 0.5, offset threshold 0.35, 250 ms of least speech and silence and 30 ms of padding.
 */
class Hysteresis {
  public:
    /** Takes the next frame's probability and appends the speech it confirms. */
    void add(double prob, std::vector<Interval> & out);
    /** Closes a confirmed run at the end of the stream. */
    void finish(std::vector<Interval> & out);
    /** The sample before which speech and silence can no longer be revised. */
    int64_t decided() const;

  private:
    void emit(int64_t safe_frame, bool closing, std::vector<Interval> & out);
    int64_t frame_ = 0, run_start_ = 0, emitted_ = 0;
    std::optional<int64_t> silence_;
    bool triggered_ = false, confirmed_ = false;
};
} // namespace fastdiar
