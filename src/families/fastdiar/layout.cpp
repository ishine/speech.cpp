#include "layout.h"
#include "error.h"
#include "silero-vad/layout.h"
#include <cmath>
#include <numeric>

namespace fastdiar {
namespace {
const Storage f32 = float32_storage();
void require(bool good, const ModelFile & m, const std::string & what) {
    if (!good)
        throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}
void norm(std::vector<TensorSpec> & t, const std::string & p, int64_t channels, bool batch) {
    add_block(t, p, {{".weight", {channels}, f32}, {".bias", {channels}, f32}});
    if (batch)
        add_block(t, p, {{".running_mean", {channels}, f32}, {".running_var", {channels}, f32}});
}
void conv(std::vector<TensorSpec> & t, const std::string & p, int64_t in, int64_t out, int64_t f, int64_t time, bool bias = true) {
    t.push_back({p + ".weight", {f, time, in, out}, f32});
    if (bias)
        t.push_back({p + ".bias", {out}, f32});
}
void linear(std::vector<TensorSpec> & t, const std::string & p, int64_t in, int64_t out) {
    add_block(t, p, {{".weight", {in, out}, f32}, {".bias", {out}, f32}});
}
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "diarization", nullptr);
    const int mel = m.size("fastdiar.mels"), channels = m.size("fastdiar.channels"), stages = m.count("fastdiar.stage_count");
    const int hop = m.size("fastdiar.hop"), window = m.size("fastdiar.window"), bins = m.size("fastdiar.bins");
    const int stride = m.size("fastdiar.time_stride"), embedding = m.size("fastdiar.embedding_dim");
    require(m.u32("speech.sample_rate") == 16000 && mel <= 512 && channels <= 1024 && stages <= 16 && (int64_t)mel * channels <= 16384 &&
                hop <= window && window <= 4096 && bins <= 4096 && stride <= 256 && embedding <= 4096,
            m, "encoder dimensions are outside the supported range");
    for (const char * key : {"norm_eps", "stft_eps", "cms_eps", "variance_floor", "normalize_eps", "layer_eps", "bn_eps"}) {
        const float v = m.f32(std::string("fastdiar.") + key);
        require(std::isfinite(v) && v > 0 && v < 1, m, std::string("invalid ") + key);
    }
    m.boolean("fastdiar.time_upsampling");
    std::vector<TensorSpec> t;
    conv(t, "backbone.stem.0", 1, channels, 3, 3);
    norm(t, "backbone.stem.1", channels, false);
    int c = channels, f = mel, ts = 1;
    const int cf = c * f;
    for (int i = 0; i < stages; i++) {
        const std::string p = "fastdiar.stage." + std::to_string(i) + ".", b = "backbone.stages." + std::to_string(i);
        const int mid = m.size(p + "mid_channels"), fs = m.size(p + "freq_stride"), time = m.size(p + "time_stride");
        const int groups = m.size(p + "groups"), blocks = m.count(p + "blocks"), hidden = m.size(p + "bottleneck");
        const uint32_t project = m.u32(p + "project");
        require(m.size(p + "channels") == c && m.size(p + "mels") == f && f % fs == 0 && time % ts == 0 && time <= stride &&
                    groups == std::gcd(c, mid) && mid <= 4096 && blocks <= 64 && hidden <= cf && project <= 1 && (project || mid == c * fs),
                m, "stage dimensions do not agree");
        const float eps = m.f32(p + "bn_eps");
        require(std::isfinite(eps) && eps > 0 && eps < 1, m, "invalid residual BatchNorm epsilon");
        t.push_back({b + ".aggregate.w", {i + 1, cf}, f32});
        conv(t, b + ".downsample", c, mid, fs, time);
        for (int j = 0; j < blocks; j++) {
            const std::string block = b + ".conv_blocks." + std::to_string(j) + ".conv_block";
            for (int k = 1; k <= 2; k++) {
                const std::string n = std::to_string(k);
                conv(t, block + ".conv" + n, 1, mid, 3, 3, false);
                conv(t, block + ".conv" + n + "pw", mid, mid, 1, 1);
                norm(t, block + ".bn" + n, mid, true);
            }
        }
        if (project) {
            require(m.f32(p + "project_eps") > 0 && m.f32(p + "project_eps") < 1, m, "invalid projection BatchNorm epsilon");
            conv(t, b + ".project.0", mid, c * fs, 1, 1);
            norm(t, b + ".project.1", c * fs, true);
        }
        linear(t, b + ".context.red_dim_conv.0", cf, hidden);
        norm(t, b + ".context.red_dim_conv.1", hidden, false);
        linear(t, b + ".context.tcm.0", hidden, hidden * 2);
        norm(t, b + ".context.tcm.1", hidden * 2, false);
        linear(t, b + ".context.tcm.3", hidden * 2, hidden);
        linear(t, b + ".context.exp_dim_conv", hidden, cf);
        c *= fs;
        f /= fs;
        ts = time;
    }
    require(ts == stride, m, "time_stride does not match the last stage");
    t.push_back({"backbone.fin_wght1d.w", {stages + 1, cf}, f32});
    const uint32_t head = m.u32("fastdiar.head_channels");
    require(head <= 4096, m, "head_channels is too large");
    if (head)
        conv(t, "backbone.head", c, head, 1, 1);
    const int64_t dim = (head ? head : c) * f;
    norm(t, "bn", dim, true);
    linear(t, "linear", dim, embedding);
    add_block(t, "spec.torchfbank",
              {{".1.flipped_filter", {2}, f32},
               {".2.real_kernel_pt", {window, bins}, f32},
               {".2.image_kernel_pt", {window, bins}, f32},
               {".2.melbanks_pt", {bins, mel}, f32}});
    for (TensorSpec & spec : silero_vad::network_tensors(m))
        t.push_back(std::move(spec));
    return t;
}
} // namespace
const Layout layout = {"fastdiar", 2, "convert it with reference/fastdiar/convert.py", tensors};
bool is_vad_tensor(const std::string & name) {
    for (const char * prefix : {"stft.", "encoder.", "lstm.", "decoder."})
        if (name.rfind(prefix, 0) == 0)
            return true;
    return false;
}
} // namespace fastdiar
