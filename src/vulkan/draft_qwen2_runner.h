#ifndef DRAFT_QWEN2_RUNNER_H
#define DRAFT_QWEN2_RUNNER_H

#include <stdint.h>

class DraftQwen2Runner {
public:
    DraftQwen2Runner();
    ~DraftQwen2Runner();

    bool init(const char *model_path);
    void reset_kv_cache();
    int forward_token_gpu_argmax(int token_id, int pos);
    void cleanup();

private:
    void *m_impl = nullptr;
    void *m_file_base = nullptr;
    void *m_h_file = nullptr;
    void *m_h_map = nullptr;
};

#endif // DRAFT_QWEN2_RUNNER_H
