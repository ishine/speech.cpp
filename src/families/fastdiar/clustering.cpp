// Adapted from microsoft/fastdiar, commit f48bff53b9a6a0cce4539e94b6def95f609eaea4.
// Copyright (c) Microsoft Corporation. Licensed under the MIT License.

#include "clustering.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace fastdiar {
namespace {

float dot(const std::vector<float> & a, const std::vector<float> & b) {
    float sum = 0;
    for (size_t d = 0; d < a.size(); d++) sum += a[d] * b[d];
    return sum;
}

void normalize(std::vector<float> & v) {
    const float norm = std::sqrt(dot(v, v));
    for (float & x : v) x /= norm;
}

}  // namespace

void OnlineClustering::Median::add(const std::vector<float> & v) {
    if (lower.empty()) { lower.resize(v.size()); upper.resize(v.size()); }
    for (size_t d = 0; d < v.size(); d++) {
        auto & lo = lower[d];
        auto & hi = upper[d];
        if (lo.empty() || v[d] <= lo.front()) {
            lo.push_back(v[d]);
            std::push_heap(lo.begin(), lo.end());
        } else {
            hi.push_back(v[d]);
            std::push_heap(hi.begin(), hi.end(), std::greater<float>());
        }
        if (lo.size() > hi.size() + 1) {
            std::pop_heap(lo.begin(), lo.end());
            hi.push_back(lo.back()); lo.pop_back();
            std::push_heap(hi.begin(), hi.end(), std::greater<float>());
        } else if (hi.size() > lo.size()) {
            std::pop_heap(hi.begin(), hi.end(), std::greater<float>());
            lo.push_back(hi.back()); hi.pop_back();
            std::push_heap(lo.begin(), lo.end());
        }
    }
    count++;
    unit.clear();
}

const std::vector<float> & OnlineClustering::Median::value() {
    if (!unit.empty()) return unit;
    unit.resize(lower.size());
    for (size_t d = 0; d < unit.size(); d++) {
        unit[d] = count % 2 ? lower[d].front() : (lower[d].front() + upper[d].front()) * 0.5f;
    }
    normalize(unit);
    return unit;
}

OnlineClustering::OnlineClustering(size_t dimension, ClusteringOptions options)
    : dimension_(dimension), options_(options) {
    if (!dimension || options.delay_frames < 0 || options.max_delay_frames < 0 ||
        !std::isfinite(options.clust_th) || !std::isfinite(options.merge_th) || !std::isfinite(options.confidence)) {
        throw std::invalid_argument("invalid clustering dimension, frame delay or threshold");
    }
}

int OnlineClustering::create(size_t anchor) {
    clusters_.emplace_back();
    clusters_.back().anchors.push_back(anchor);
    return (int) clusters_.size() - 1;
}

int OnlineClustering::nearest(size_t frame, float * similarity) {
    float best = -std::numeric_limits<float>::infinity();
    int found = -1;
    for (size_t c = 0; c < clusters_.size(); c++) {
        Cluster & cluster = clusters_[c];
        if (!cluster.active) continue;
        while (cluster.median.count < cluster.anchors.size()) {
            cluster.median.add(embeddings_[cluster.anchors[cluster.median.count]]);
        }
        const float score = dot(embeddings_[frame], cluster.median.value());
        if (found < 0 || (std::isnan(score) && !std::isnan(best)) || score > best) { best = score; found = (int) c; }
    }
    if (similarity) *similarity = best;
    return found;
}

std::vector<size_t> OnlineClustering::take_ready(size_t frame) {
    std::vector<size_t> ready, waiting;
    for (size_t p : pending_) {
        if (frame >= (size_t) options_.max_delay_frames && p <= frame - options_.max_delay_frames) ready.push_back(p);
        else waiting.push_back(p);
    }
    pending_ = std::move(waiting);
    return ready;
}

bool OnlineClustering::label(int c, const std::vector<size_t> & frames) {
    bool fresh = false;
    for (size_t f : frames) {
        labels_[f] = vad_[f] ? c : -1;
        if (!vad_[f]) continue;
        if (!clusters_[c].speech_seen) {
            clusters_[c].speech_seen = true;
            speech_order_.push_back(c);
            fresh = true;
        }
        clusters_[c].speech.push_back(f);
    }
    return fresh;
}

void OnlineClustering::assign(size_t frame) {
    if (frame < (size_t) options_.delay_frames || !vad_[frame] ||
        dot(embeddings_[frame], embeddings_[frame - options_.delay_frames]) < options_.confidence) {
        pending_.push_back(frame);
        return;
    }
    float score;
    int c = nearest(frame, &score);
    const bool new_cluster = c < 0 || !(score >= options_.clust_th);
    auto ready = take_ready(frame);
    pending_.push_back(frame);
    if (new_cluster) c = create(frame);
    else clusters_[c].anchors.push_back(frame);
    if (label(c, ready) && options_.online_merge) merge();
}

void OnlineClustering::force(size_t frame) {
    const auto ready = take_ready(frame);
    bool fresh = false;
    for (size_t f : ready) {
        if (!vad_[f]) { labels_[f] = -1; continue; }
        int c = nearest(f);
        if (c < 0) c = create(f);
        fresh = label(c, {f}) || fresh;
    }
    if (fresh && options_.online_merge) merge();
}

std::vector<float> OnlineClustering::speech_median(int c) const {
    const auto & frames = clusters_[c].speech;
    std::vector<float> median(dimension_), values(frames.size());
    for (size_t d = 0; d < dimension_; d++) {
        for (size_t i = 0; i < frames.size(); i++) values[i] = embeddings_[frames[i]][d];
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        median[d] = values[mid];
        if (values.size() % 2 == 0) median[d] = (median[d] + *std::max_element(values.begin(), values.begin() + mid)) * 0.5f;
    }
    normalize(median);
    return median;
}

void OnlineClustering::merge() {
    std::vector<int> ids;
    std::vector<std::vector<float>> medians;
    for (int c : speech_order_) {
        if (clusters_[c].speech.empty()) continue;
        ids.push_back(c);
        medians.push_back(speech_median(c));
    }
    std::vector<size_t> component(ids.size());
    std::iota(component.begin(), component.end(), 0);
    const auto root = [&](size_t i) { while (component[i] != i) i = component[i]; return i; };
    for (size_t i = 0; i < ids.size(); i++) {
        for (size_t j = 0; j < i; j++) {
            if (dot(medians[i], medians[j]) > options_.merge_th) component[root(i)] = root(j);
        }
    }
    std::map<size_t, std::vector<int>> groups;
    for (size_t i = 0; i < ids.size(); i++) groups[root(i)].push_back(ids[i]);
    for (const auto & [unused, group] : groups) {
        const int target = *std::min_element(group.begin(), group.end());
        for (int source : group) {
            if (source == target) continue;
            Cluster & to = clusters_[target];
            Cluster & from = clusters_[source];
            for (size_t f : from.speech) labels_[f] = target;
            to.speech.insert(to.speech.end(), from.speech.begin(), from.speech.end());
            to.anchors.insert(to.anchors.end(), from.anchors.begin(), from.anchors.end());
            from.speech.clear(); from.anchors.clear(); from.median = Median{};
            from.active = false;
        }
    }
}

std::vector<FrameLabel> OnlineClustering::emit(long long last) {
    std::vector<FrameLabel> out;
    while ((long long) next_ <= last && next_ < labels_.size() && labels_[next_] != -2) {
        out.push_back({next_, labels_[next_]}); next_++;
    }
    return out;
}

std::vector<FrameLabel> OnlineClustering::push(const float * embedding, bool speech) {
    if (ended_) throw std::logic_error("clustering has ended; start a new stream");
    if (!embedding) throw std::invalid_argument("embedding is null");
    std::vector<float> row(embedding, embedding + dimension_);
    for (float x : row) if (!std::isfinite(x)) throw std::invalid_argument("embedding contains a non-finite value");
    const float norm = dot(row, row);
    if (!(norm > 0) || std::abs(norm - 1.0f) > 0.001f) throw std::invalid_argument("embedding is not normalized");
    embeddings_.push_back(std::move(row)); vad_.push_back(speech); labels_.push_back(-2);
    const size_t i = embeddings_.size() - 1;
    assign(i); force(i);
    return emit((long long) i - options_.max_delay_frames);
}

std::vector<FrameLabel> OnlineClustering::end() {
    if (ended_) throw std::logic_error("clustering has ended; start a new stream");
    ended_ = true;
    if (!clusters_.empty()) {
        for (size_t f : pending_) {
            if (!vad_[f]) { labels_[f] = -1; continue; }
            label(nearest(f), {f});
        }
    }
    pending_.clear();
    return emit((long long) embeddings_.size() - 1);
}

}  // namespace fastdiar
