#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "engine.h"
#include "fastdiar/encoder.h"
#include "fastdiar/labels.h"
#include "fastdiar/layout.h"
#include "fastdiar/vad.h"
#include "ggml-cpu.h"
#include "silero-vad/detector.h"

namespace {

/** The encoder step of a live stream, 320 ms of 80 ms frames, which sets how often results come and never their value. */
constexpr int kStepFrames = 4;

class FastdiarStream : public DiarizationStream {
public:
    /**
     * With `pieces`, each push gives what is final of the turn under way, so that settled() advances through a long
     * turn. ponytail: a stream's turns come in pieces and speaking() gives nothing, since the C API walks a model's fixed
     * number of speakers; a call that lists the speakers under way would give whole turns.
     */
    FastdiarStream(const std::string & path, ggml_backend_t backend, silero_vad::Detector & detector, bool pieces)
        : encoder_(path, backend, kStepFrames), chunks_(detector), labels_(encoder_.dimension(), encoder_.frame_samples()), pieces_(pieces) {}

    void push(const float * samples, size_t n) override {
        probs_.clear();
        chunks_.push(samples, n, probs_);
        speech_.clear();
        for (const float p : probs_) vad_.add(p, speech_);
        labels_.add_speech(speech_);
        rows_.clear();
        encoder_.push(samples, n, rows_);
        labels_.add_embeddings(rows_);
        labels_.drain(vad_.decided());
        if (pieces_) labels_.close_turn();
        collect();
    }

    void end() override {
        // Silero's last partial chunk stays unscored, as in StreamingVAD.flush().
        speech_.clear();
        vad_.finish(speech_);
        labels_.add_speech(speech_);
        rows_.clear();
        encoder_.end(rows_);
        labels_.add_embeddings(rows_);
        labels_.drain(std::numeric_limits<int64_t>::max());
        labels_.finish();
        ended_ = true;
        collect();
    }

    const std::vector<SpeakerSegment> & segments() const override { return given_; }
    double settled() const override { return ended_ ? std::numeric_limits<double>::infinity() : labels_.settled(); }
    std::optional<double> speaking(int) const override { return std::nullopt; }

private:
    void collect() {
        const std::vector<fastdiar::Turn> & turns = labels_.turns();
        for (size_t i = given_.size(); i < turns.size(); i++) given_.push_back({turns[i].start, turns[i].end, turns[i].speaker});
    }

    fastdiar::Encoder encoder_;
    silero_vad::ChunkStream chunks_;
    fastdiar::Hysteresis vad_;
    fastdiar::Labels labels_;
    bool pieces_, ended_ = false;
    std::vector<float> probs_, rows_;
    std::vector<fastdiar::Interval> speech_;
    std::vector<SpeakerSegment> given_;
};

class FastdiarEngine : public Engine {
public:
    FastdiarEngine(const std::string & path, ggml_backend_t backend)
        : path_(path), backend_(backend), cpu_(ggml_backend_cpu_init(), ggml_backend_free),
          // The official runs its VAD on the CPU, and Silero's graphs are not for a GPU.
          detector_(path, cpu_.get(), fastdiar::layout, fastdiar::is_vad_tensor) {}

    std::vector<SpeakerSegment> diarize(const std::vector<float> & samples, const RequestValues &, Run & run) override {
        FastdiarStream stream(path_, backend_, detector_, false);
        const size_t piece = 16000;
        for (size_t at = 0; at < samples.size(); at += piece) {
            stream.push(samples.data() + at, std::min(piece, samples.size() - at));
            run.progress((double) std::min(at + piece, samples.size()) / (double) samples.size());
            if (run.stopped()) return {};
        }
        stream.end();
        return stream.segments();
    }

    /** Each stream holds an encoder of its own, whose convolutions keep the history of its audio. */
    std::unique_ptr<DiarizationStream> start_diarization(const RequestValues &) override {
        return std::make_unique<FastdiarStream>(path_, backend_, detector_, true);
    }

    void warm_up() override {
        FastdiarStream stream(path_, backend_, detector_, false);
        const std::vector<float> silence(16000, 0.0f);
        stream.push(silence.data(), silence.size());
        stream.end();
    }

private:
    std::string path_;
    ggml_backend_t backend_;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu_;
    silero_vad::Detector detector_;
};

}  // namespace

/** FASTDIAR opens clusters as it hears speakers, so it has no most speakers, which its information gives as 0, and takes no options. */
FamilyInfo describe_fastdiar(const std::shared_ptr<const ModelFile> &) {
    return {};
}

std::unique_ptr<Engine> load_fastdiar(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<FastdiarEngine>(path, backend);
}
