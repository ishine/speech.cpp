// Checks cluster IDs and their emission time exactly against the official streaming clustering dumps.
// The inputs are normalized frame embeddings, independent of an encoder or a GPU.
// usage: fastdiar-clustering-check <reference case dir> [...]

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastdiar/clustering.h"
#include "npy.h"

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: fastdiar-clustering-check <reference case dir> [...]\n"); return 2; }
    try {
        for (int a = 1; a < argc; a++) {
            const std::string root = argv[a];
            const Npy embeddings = read_npy(root + "/embeddings.npy"), vad = read_npy(root + "/vad.npy"),
                      expected = read_npy(root + "/expected.npy"), options = read_npy(root + "/options.npy");
            if (embeddings.shape.size() != 2 || embeddings.shape[1] < 1 || vad.u8.size() != (size_t) embeddings.shape[0] ||
                embeddings.f32.size() != (size_t) embeddings.size() || expected.shape.size() != 2 || expected.shape[1] != 3 ||
                expected.i32.size() != (size_t) expected.size() || options.f64.size() != 6) {
                throw std::runtime_error(root + ": invalid dump shapes or dtypes");
            }
            fastdiar::ClusteringOptions settings;
            settings.delay_frames = (int) options.f64[0]; settings.max_delay_frames = (int) options.f64[1];
            settings.clust_th = options.f64[2]; settings.merge_th = options.f64[3]; settings.confidence = options.f64[4];
            settings.online_merge = options.f64[5] != 0;
            fastdiar::OnlineClustering clustering((size_t) embeddings.shape[1], settings);
            size_t seen = 0;
            const auto compare = [&](size_t step, const std::vector<fastdiar::FrameLabel> & output) {
                for (const auto & label : output) {
                    if (seen >= expected.i32.size() / 3 || expected.i32[seen * 3] != (int) step ||
                        expected.i32[seen * 3 + 1] != (int) label.frame || expected.i32[seen * 3 + 2] != label.cluster) {
                        throw std::runtime_error(root + ": label or emission time differs at step " + std::to_string(step) +
                                                 ", frame " + std::to_string(label.frame) + ", cluster " + std::to_string(label.cluster));
                    }
                    seen++;
                }
            };
            for (size_t i = 0; i < vad.u8.size(); i++) compare(i, clustering.push(&embeddings.f32[i * embeddings.shape[1]], vad.u8[i] != 0));
            compare(vad.u8.size(), clustering.end());
            if (seen != expected.i32.size() / 3) throw std::runtime_error(root + ": missing finalized labels");
            bool rejected = false;
            try { clustering.end(); } catch (const std::logic_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error(root + ": accepts an ended stream");
            std::printf("%s: %zu labels, IDs and emission steps match exactly\n", root.c_str(), seen);
        }
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
