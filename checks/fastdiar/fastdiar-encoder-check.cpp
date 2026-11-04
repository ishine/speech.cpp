// Compares the native streaming encoder's frontend, network stages and causal histories with official dumps.
// Each network step starts from the reference mel input and previous histories; whole-waveform runs check accumulated state.
// usage: fastdiar-encoder-check <model.gguf> <case dir> [cpu|gpu]
#include "backend.h"
#include "compare.h"
#include "fastdiar/clustering.h"
#include "fastdiar/encoder.h"
#include "fastdiar/layout.h"
#include "npy.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;
void require_close(const std::string & name, const std::vector<float> & got, const std::vector<float> & expected, double bound) {
    if (got.size() != expected.size())
        throw std::runtime_error(name + ": different output size");
    for (float x : got)
        if (!std::isfinite(x))
            throw std::runtime_error(name + ": non-finite output");
    const Diff d = compare(got, expected);
    print_diff(name, d);
    if (!(d.snr_db >= bound))
        throw std::runtime_error(name + ": SNR is below " + std::to_string(bound) + " dB");
}
int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: fastdiar-encoder-check <model.gguf> <case dir> [cpu|gpu]\n");
        return 2;
    }
    try {
        const std::string device = argc == 4 ? argv[3] : "cpu", root = argv[2];
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(device), 1), ggml_backend_free);
        fastdiar::Encoder encoder(argv[1], backend.get());
        fastdiar::Frontend frontend(encoder.model());
        fastdiar::Network network(encoder.model());
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        if (device != "cpu" && device != "gpu")
            throw std::invalid_argument("device must be cpu or gpu");
        // Both backends request F32 operands; the checks keep the same strict floor.
        const double bound = 80;
        for (int i = 0;; i++) {
            const std::string step = root + "/step" + std::to_string(i);
            if (!fs::exists(fs::u8path(step)))
                break;
            const auto audio = read_npy(step + "/audio.npy").f32;
            const auto features = read_npy(step + "/features.npy");
            require_close("frontend step" + std::to_string(i), frontend.compute(audio.data(), audio.size()), features.f32, 80);
            fastdiar::NetworkState state;
            for (const auto & entry : fs::directory_iterator(fs::u8path(step))) {
                const std::string name = entry.path().filename().u8string();
                if (name.rfind("input-", 0) == 0 && entry.path().extension() == ".npy")
                    state[name.substr(6, name.size() - 10)] = read_npy(entry.path().u8string()).f32;
            }
            for (int stage = 0; stage < encoder.model().count("fastdiar.stage_count"); stage++) {
                Graph independent;
                std::vector<ggml_tensor *> maps;
                for (int j = 0; j <= stage; j++) {
                    const auto input = read_npy(step + "/stage" + std::to_string(stage) + "-input" + std::to_string(j) + ".npy");
                    maps.push_back(independent.input(input.f32, input.shape.at(1), input.shape.at(0)));
                }
                fastdiar::NetworkOutput saved;
                auto * result = network.stage(independent, stage, maps, state, saved);
                independent.output(result);
                independent.compute(backend.get(), alloc.get());
                require_close("independent stage" + std::to_string(stage), Graph::read(result),
                              read_npy(step + "/stage" + std::to_string(stage) + ".npy").f32, bound);
            }
            Graph graph;
            auto output = network.forward(graph, features.f32, features.shape.at(0), state, true);
            graph.compute(backend.get(), alloc.get());
            for (const auto & [name, t] : output.stages)
                require_close(name + " step" + std::to_string(i), Graph::read(t), read_npy(step + "/" + name + ".npy").f32, bound);
            require_close("embeddings step" + std::to_string(i), Graph::read(output.embeddings), read_npy(step + "/embeddings.npy").f32,
                          bound);
            for (const auto & [name, t] : output.caches)
                require_close("cache " + name, Graph::read(t), read_npy(step + "/output-" + name + ".npy").f32, bound);
        }
        const auto audio = read_npy(root + "/audio.npy").f32, expected = read_npy(root + "/expected.npy").f32;
        std::vector<size_t> chunks{5120};
        if (audio.size() < 100000)
            chunks.insert(chunks.end(), {137, 1, std::max(size_t(1), audio.size())});
        if (audio.size() < 100000) {
            for (int step : {1, 25}) {
                fastdiar::Encoder alternate(argv[1], backend.get(), step);
                std::vector<float> embeddings;
                alternate.push(audio.data(), audio.size(), embeddings);
                alternate.end(embeddings);
                require_close("network step=" + std::to_string(step), embeddings, expected, bound);
            }
        }
        for (size_t chunk : chunks) {
            encoder.reset();
            std::vector<float> embeddings;
            for (size_t at = 0; at < audio.size(); at += chunk) {
                encoder.push(audio.data() + at, std::min(chunk, audio.size() - at), embeddings);
                if (audio.size() > 160000 && at % 512000 == 0) {
                    std::printf("waveform: %.1f / %.1f seconds\n", at / 16000.0, audio.size() / 16000.0);
                    std::fflush(stdout);
                }
            }
            encoder.end(embeddings);
            require_close("waveform chunk=" + std::to_string(chunk), embeddings, expected, bound);
            if (fs::exists(fs::u8path(root + "/vad.npy"))) {
                const auto vad = read_npy(root + "/vad.npy").u8;
                const auto gold = read_npy(root + "/labels.npy").i32;
                if (vad.size() * encoder.dimension() != embeddings.size())
                    throw std::runtime_error("VAD length differs from embedding frames");
                fastdiar::OnlineClustering clustering(encoder.dimension());
                std::vector<int32_t> labels;
                const auto collect = [&](size_t step, const std::vector<fastdiar::FrameLabel> & output) {
                    for (const auto & label : output) {
                        labels.push_back((int32_t)step);
                        labels.push_back((int32_t)label.frame);
                        labels.push_back(label.cluster);
                    }
                };
                for (size_t i = 0; i < vad.size(); i++)
                    collect(i, clustering.push(embeddings.data() + i * encoder.dimension(), vad[i] != 0));
                collect(vad.size(), clustering.end());
                if (labels != gold)
                    throw std::runtime_error("native waveform embeddings change the reference cluster labels or emission time");
                std::printf("waveform + clustering: %zu labels and emission steps match exactly\n", gold.size() / 3);
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
