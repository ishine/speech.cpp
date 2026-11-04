#include "labels.h"
#include <cmath>
namespace fastdiar {
Labels::Labels(size_t dimension, int frame_samples)
    : dimension_(dimension), frame_samples_(frame_samples), shift_((double)frame_samples / 16000), clustering_(dimension) {}
void Labels::add_speech(const std::vector<Interval> & intervals) {
    for (const Interval & i : intervals) {
        // std::nearbyint rounds halves to even, as Python's round() does.
        const size_t first = (size_t)std::nearbyint((double)i.start / frame_samples_), last = (size_t)std::nearbyint((double)i.end / frame_samples_);
        if (last > first)
            speech_.push_back({first, last});
    }
}
void Labels::add_embeddings(const std::vector<float> & rows) {
    for (size_t at = 0; at + dimension_ <= rows.size(); at += dimension_)
        waiting_.emplace_back(rows.begin() + at, rows.begin() + at + dimension_);
}
bool Labels::is_speech(size_t frame) {
    while (head_ < speech_.size() && speech_[head_].second <= frame)
        head_++;
    return head_ < speech_.size() && speech_[head_].first <= frame;
}
void Labels::drain(int64_t decided) {
    while (!waiting_.empty() && (int64_t)(next_ + 1) * frame_samples_ <= decided) {
        const std::vector<float> row = std::move(waiting_.front());
        waiting_.pop_front();
        const size_t frame = next_++;
        emit(clustering_.push(row.data(), is_speech(frame)));
    }
}
void Labels::finish() {
    emit(clustering_.end());
    close_turn();
}
void Labels::close_turn() {
    if (open_)
        turns_.push_back(*open_);
    open_.reset();
}
void Labels::emit(const std::vector<FrameLabel> & labels) {
    for (const FrameLabel & label : labels) {
        emitted_ = label.frame + 1;
        if (label.cluster < 0) {
            close_turn();
            continue;
        }
        const int speaker = speaker_of_.emplace(label.cluster, (int)speaker_of_.size()).first->second;
        const double start = (double)label.frame * shift_, end = start + shift_;
        if (open_ && open_->speaker == speaker && label.frame == open_frame_ + 1) {
            open_->end = end;
        } else {
            close_turn();
            open_ = Turn{start, end, speaker};
        }
        open_frame_ = label.frame;
    }
}
} // namespace fastdiar
