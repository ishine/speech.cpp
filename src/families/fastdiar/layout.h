#pragma once
#include "model-file.h"
#include <string>
namespace fastdiar {
/** Layout 2: the streaming ReDimNet2 encoder, frontend kernels included, in float32, and Silero VAD's keys and tensors. */
extern const Layout layout;
/** Whether a tensor of the file is Silero VAD's. */
bool is_vad_tensor(const std::string & name);
} // namespace fastdiar
