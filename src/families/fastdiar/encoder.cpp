#include "encoder.h"
#include "error.h"
#include "layout.h"
#include <cmath>
#include <stdexcept>
namespace fastdiar {
Encoder::Encoder(const std::string & path, ggml_backend_t backend, int step)
    : model_(path, backend, layout, [](const std::string & name) { return !is_vad_tensor(name); }), frontend_(model_), network_(model_), backend_(backend),
      allocator_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {
    if (step < 1 || step > 4096)
        throw std::invalid_argument("encoder step must be between 1 and 4096 frames");
    if (!allocator_)
        throw Error(Fault::OutOfMemory, "cannot allocate FASTDIAR graph allocator");
    block_ = (size_t)step * model_.size("fastdiar.hop") * model_.size("fastdiar.time_stride");
}
void Encoder::reset() {
    frontend_.reset();
    state_.clear();
    pending_.clear();
    ended_ = false;
}
void Encoder::process(size_t n, std::vector<float> & embeddings) {
    if (!n)
        return;
    auto features = frontend_.compute(pending_.data(), n);
    Graph graph;
    auto output = network_.forward(graph, features, (int64_t)n / model_.size("fastdiar.hop"), state_);
    graph.compute(backend_, allocator_.get());
    for (const auto & entry : output.caches)
        state_[entry.first] = Graph::read(entry.second);
    auto rows = Graph::read(output.embeddings);
    embeddings.insert(embeddings.end(), rows.begin(), rows.end());
    pending_.erase(pending_.begin(), pending_.begin() + n);
}
void Encoder::push(const float * samples, size_t n, std::vector<float> & embeddings) {
    if (ended_)
        throw std::logic_error("encoder has ended; reset it for another recording");
    if (!n)
        return;
    if (!samples)
        throw std::invalid_argument("audio samples are null");
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(samples[i]))
            throw std::invalid_argument("audio contains a non-finite sample");
    pending_.insert(pending_.end(), samples, samples + n);
    // Keep graph sizes bounded; a large caller push is consumed with the same configured step as small pushes.
    try {
        while (pending_.size() >= block_)
            process(block_, embeddings);
    } catch (...) {
        ended_ = true;
        throw;
    }
}
void Encoder::end(std::vector<float> & embeddings) {
    if (ended_)
        throw std::logic_error("encoder has ended; reset it for another recording");
    const size_t quantum = (size_t)model_.size("fastdiar.hop") * model_.size("fastdiar.time_stride");
    ended_ = true;
    process(pending_.size() / quantum * quantum, embeddings);
    pending_.clear();
}
} // namespace fastdiar
