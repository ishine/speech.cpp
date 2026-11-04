#pragma once
#include <cstddef>
#include "frontend.h"
#include "network.h"
#include <memory>
#include <vector>
namespace fastdiar {
/** Streaming normalized speaker embeddings from mono 16 kHz audio; the frontend runs on CPU and the network on backend. */
class Encoder {
  public:
    Encoder(const std::string & path, ggml_backend_t backend, int step_frames = 4);
    /** Starts another recording without reloading weights. */
    void reset();
    /** Appends complete output frames to embeddings, row-major [frames, embedding_dim]. */
    void push(const float * samples, size_t n, std::vector<float> & embeddings);
    /** Emits remaining whole encoder frames; an incomplete final frame is not padded, as in StreamingEncoder.flush(). */
    void end(std::vector<float> & embeddings);
    int dimension() const { return model_.size("fastdiar.embedding_dim"); }
    int frame_samples() const {
        return model_.size("fastdiar.hop") * (model_.boolean("fastdiar.time_upsampling") ? 1 : model_.size("fastdiar.time_stride"));
    }
    /** The validated GGUF model and its network are available to stage checks. */
    const ModelFile & model() const { return model_; }

  private:
    void process(size_t n, std::vector<float> & embeddings);
    ModelFile model_;
    Frontend frontend_;
    Network network_;
    ggml_backend_t backend_;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_;
    NetworkState state_;
    std::vector<float> pending_;
    size_t block_;
    bool ended_ = false;
};
} // namespace fastdiar
