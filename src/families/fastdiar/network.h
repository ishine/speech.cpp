#pragma once
#include "graph.h"
#include "model-file.h"
#include <map>
#include <string>
#include <vector>

namespace fastdiar {
/** Per-convolution causal history, held separately for each recording. */
using NetworkState = std::map<std::string, std::vector<float>>;
/** A graph's embeddings and optional intermediate stages, in channels-first ggml order. */
struct NetworkOutput {
    ggml_tensor * embeddings = nullptr;
    std::map<std::string, ggml_tensor *> stages, caches;
};
/** The streaming ReDimNet2 backbone and per-frame projection; no VAD or clustering. */
class Network {
  public:
    explicit Network(const ModelFile & model) : m_(model) {}
    /** Takes log-mel [mels, frames], whose frame count is a multiple of time_stride, and the previous histories. */
    NetworkOutput forward(Graph & graph, const std::vector<float> & features, int64_t frames, const NetworkState & state,
                          bool keep_stages = false) const;
    /** One stage from the official shared-map inputs, for independent stage checks. */
    ggml_tensor * stage(Graph & graph, int index, const std::vector<ggml_tensor *> & maps, const NetworkState & state,
                        NetworkOutput & output) const;

  private:
    ggml_tensor * conv(Graph &, const std::string &, ggml_tensor *, int, int, bool, const NetworkState &, NetworkOutput &) const;
    ggml_tensor * point(ggml_context *, const std::string &, ggml_tensor *) const;
    ggml_tensor * batch(ggml_context *, const std::string &, ggml_tensor *, float, bool) const;
    ggml_tensor * layer(ggml_context *, const std::string &, ggml_tensor *) const;
    ggml_tensor * aggregate(ggml_context *, const std::string &, const std::vector<ggml_tensor *> &) const;
    const ModelFile & m_;
};
} // namespace fastdiar
