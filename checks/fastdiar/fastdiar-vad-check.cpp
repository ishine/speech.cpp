// Checks the VAD's hysteresis exactly against the official StreamingVAD's trace of probabilities.
// usage: fastdiar-vad-check <reference case dir> [...]
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastdiar/vad.h"
#include "npy.h"

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: fastdiar-vad-check <reference case dir> [...]\n"); return 2; }
    try {
        for (int a = 1; a < argc; a++) {
            const std::string root = argv[a];
            const Npy probs = read_npy(root + "/probs.npy"), decided = read_npy(root + "/decided.npy"), expected = read_npy(root + "/intervals.npy");
            fastdiar::Hysteresis vad;
            size_t seen = 0;
            const auto compare = [&](size_t step, const std::vector<fastdiar::Interval> & out) {
                for (const fastdiar::Interval & i : out) {
                    if (seen >= expected.f64.size() / 3 || expected.f64[seen * 3] != (double) step || expected.f64[seen * 3 + 1] != (double) i.start ||
                        expected.f64[seen * 3 + 2] != (double) i.end) {
                        throw std::runtime_error(root + ": interval differs at frame " + std::to_string(step));
                    }
                    seen++;
                }
            };
            std::vector<fastdiar::Interval> out;
            for (size_t i = 0; i < probs.f64.size(); i++) {
                out.clear();
                vad.add(probs.f64[i], out);
                compare(i, out);
                if ((double) vad.decided() != decided.f64[i]) throw std::runtime_error(root + ": decided differs at frame " + std::to_string(i));
            }
            out.clear();
            vad.finish(out);
            compare(probs.f64.size(), out);
            if (seen != expected.f64.size() / 3) throw std::runtime_error(root + ": missing intervals");
            std::printf("%s: %zu frames and %zu intervals match exactly\n", root.c_str(), probs.f64.size(), seen);
        }
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    return 0;
}
