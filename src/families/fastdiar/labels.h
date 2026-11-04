#pragma once
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <utility>
#include <vector>
#include "clustering.h"
#include "vad.h"
namespace fastdiar {
/** A run of one speaker in seconds; speakers number from 0 in the order they first speak. */
struct Turn {
    double start, end;
    int speaker;
};
/** What microsoft/fastdiar's StreamingDiarizer does between its VAD, encoder and clustering, for post_process=False. */
class Labels {
  public:
    Labels(size_t dimension, int frame_samples);
    void add_speech(const std::vector<Interval> & intervals);
    /** Takes rows of [frames, dimension]. */
    void add_embeddings(const std::vector<float> & rows);
    /** Clusters each buffered frame wholly before the sample `decided`. */
    void drain(int64_t decided);
    /** Ends the stream, once every frame has been added. */
    void finish();
    /** Gives the turn under way as far as it is final; the next frame of its speaker starts another. */
    void close_turn();
    const std::vector<Turn> & turns() const { return turns_; }
    /** The time before which every turn has been given, once close_turn() is called. */
    double settled() const { return (double)emitted_ * shift_; }

  private:
    bool is_speech(size_t frame);
    void emit(const std::vector<FrameLabel> & labels);
    size_t dimension_;
    int frame_samples_;
    double shift_;
    OnlineClustering clustering_;
    std::deque<std::vector<float>> waiting_;
    size_t next_ = 0, emitted_ = 0, head_ = 0;
    std::vector<std::pair<size_t, size_t>> speech_;
    std::map<int, int> speaker_of_;
    std::optional<Turn> open_;
    size_t open_frame_ = 0;
    std::vector<Turn> turns_;
};
} // namespace fastdiar
