#include "vad.h"
#include <algorithm>
namespace fastdiar {
namespace {
constexpr int64_t kFrame = 512, kPad = 480, kMinSpeech = 8, kMinSilence = 8;
constexpr double kThreshold = 0.5, kOffThreshold = 0.5 - 0.15;
} // namespace
void Hysteresis::add(double prob, std::vector<Interval> & out) {
    const int64_t i = frame_;
    if (prob >= kThreshold) {
        silence_.reset();
        if (!triggered_) {
            triggered_ = true;
            run_start_ = i;
            confirmed_ = false;
        }
    } else if (triggered_ && prob < kOffThreshold) {
        if (!silence_)
            silence_ = i;
        if (i - *silence_ >= kMinSilence) {
            if (confirmed_)
                emit(*silence_, true, out);
            triggered_ = confirmed_ = false;
            silence_.reset();
        }
    }
    if (triggered_ && !confirmed_ && i - run_start_ + 1 >= kMinSpeech)
        confirmed_ = true;
    if (triggered_ && confirmed_)
        emit(silence_ ? *silence_ : i + 1, false, out);
    frame_ = i + 1;
}
void Hysteresis::finish(std::vector<Interval> & out) {
    if (triggered_ && confirmed_)
        emit(frame_, true, out);
    triggered_ = confirmed_ = false;
    silence_.reset();
}
int64_t Hysteresis::decided() const {
    if (triggered_)
        return confirmed_ ? emitted_ : std::max(emitted_, run_start_ * kFrame - kPad);
    return std::max(emitted_, frame_ * kFrame - kPad);
}
void Hysteresis::emit(int64_t safe_frame, bool closing, std::vector<Interval> & out) {
    const int64_t start = std::max({run_start_ * kFrame - kPad, emitted_, (int64_t)0});
    const int64_t end = safe_frame * kFrame + (closing ? kPad : 0);
    if (end <= start)
        return;
    emitted_ = end;
    out.push_back({start, end});
}
} // namespace fastdiar
