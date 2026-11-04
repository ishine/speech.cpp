#pragma once

#include <cstddef>
#include <vector>

namespace fastdiar {

/** Settings of FASTDIAR's streaming OnlineClustering, without its offline postprocessing pass. */
struct ClusteringOptions {
    int delay_frames = 10;
    int max_delay_frames = 12;
    double clust_th = 0.4, merge_th = 0.8, confidence = 0.8;
    bool online_merge = true;
};

/** A finalized frame and its internal cluster ID, or -1 for silence. */
struct FrameLabel {
    size_t frame;
    int cluster;
};

/**
 * Confidence-gated online clustering of normalized float32 frame embeddings, as microsoft/fastdiar's
 * OnlineClustering with post_process=False. It retains the reference's full embedding and median history.
 * Cluster IDs are internal and are not the arrival-ordered speaker numbers of StreamingDiarizer.
 */
class OnlineClustering {
public:
    explicit OnlineClustering(size_t dimension, ClusteringOptions options = {});
    /** Takes one finite normalized embedding and its VAD decision, returning newly finalized frames. */
    std::vector<FrameLabel> push(const float * embedding, bool speech);
    /** Finalizes the remaining frames once; a later push or end is an error. */
    std::vector<FrameLabel> end();

private:
    struct Median {
        std::vector<std::vector<float>> lower, upper;
        size_t count = 0;
        std::vector<float> unit;
        void add(const std::vector<float> & embedding);
        const std::vector<float> & value();
    };
    struct Cluster {
        bool active = true, speech_seen = false;
        std::vector<size_t> anchors, speech;
        Median median;
    };
    std::vector<size_t> take_ready(size_t frame);
    int nearest(size_t frame, float * similarity = nullptr);
    int create(size_t anchor);
    void assign(size_t frame);
    void force(size_t frame);
    bool label(int cluster, const std::vector<size_t> & frames);
    void merge();
    std::vector<float> speech_median(int cluster) const;
    std::vector<FrameLabel> emit(long long last);

    size_t dimension_, next_ = 0;
    ClusteringOptions options_;
    bool ended_ = false;
    std::vector<std::vector<float>> embeddings_;
    std::vector<bool> vad_;
    std::vector<int> labels_;
    std::vector<Cluster> clusters_;
    std::vector<int> speech_order_;
    std::vector<size_t> pending_;
};

}  // namespace fastdiar
