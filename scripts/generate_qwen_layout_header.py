import sys
from parse_qwen_tensors import parse_gguf

def generate_header(gguf_path, out_header_path):
    kv, tensors, data_start = parse_gguf(gguf_path)
    
    hidden_dim = kv['qwen2.embedding_length']
    intermediate_dim = kv['qwen2.feed_forward_length']
    num_layers = kv['qwen2.block_count']
    num_heads = kv['qwen2.attention.head_count']
    num_kv_heads = kv['qwen2.attention.head_count_kv']
    head_dim = hidden_dim // num_heads
    vocab_size = tensors['output.weight']['dims'][1]
    rope_theta = float(kv.get('qwen2.rope.freq_base', 1000000.0))
    rms_eps = float(kv.get('qwen2.attention.layer_norm_rms_epsilon', 1e-6))

    code = []
    model_name = "7B" if num_layers == 28 else "0.5B"
    guard_name = f"QWEN2_{model_name.replace('.', '_')}_LAYOUT_H"
    code.append(f"/* Auto-generated Qwen2.5-Coder-{model_name} Layout Header */")
    code.append(f"#ifndef {guard_name}")
    code.append(f"#define {guard_name}")
    code.append("")
    code.append("#include <stdint.h>")
    code.append("")
    code.append(f"#define QWEN2_DATA_START_OFFSET {data_start}ULL")
    code.append(f"#define QWEN2_VOCAB_SIZE {vocab_size}")
    code.append(f"#define QWEN2_HIDDEN_SIZE {hidden_dim}")
    code.append(f"#define QWEN2_INTERMEDIATE_SIZE {intermediate_dim}")
    code.append(f"#define QWEN2_NUM_LAYERS {num_layers}")
    code.append(f"#define QWEN2_NUM_HEADS {num_heads}")
    code.append(f"#define QWEN2_NUM_KV_HEADS {num_kv_heads}")
    code.append(f"#define QWEN2_HEAD_DIM {head_dim}")
    code.append(f"#define QWEN2_RMS_NORM_EPS {rms_eps}f")
    code.append(f"#define QWEN2_ROPE_THETA {rope_theta}f")
    code.append("")
    code.append("struct Qwen2LayerOffsets {")
    code.append("    uint64_t input_layernorm;     // attn_norm.weight")
    code.append("    uint64_t q_proj;              // attn_q.weight")
    code.append("    uint64_t q_bias;              // attn_q.bias")
    code.append("    uint64_t k_proj;              // attn_k.weight")
    code.append("    uint64_t k_bias;              // attn_k.bias")
    code.append("    uint64_t v_proj;              // attn_v.weight")
    code.append("    uint64_t v_bias;              // attn_v.bias")
    code.append("    uint64_t o_proj;              // attn_output.weight")
    code.append("    uint64_t post_attn_layernorm; // ffn_norm.weight")
    code.append("    uint64_t gate_proj;           // ffn_gate.weight")
    code.append("    uint64_t up_proj;             // ffn_up.weight")
    code.append("    uint64_t down_proj;           // ffn_down.weight")
    code.append("};")
    code.append("")
    code.append("struct Qwen2Offsets {")
    code.append("    uint64_t embed_tokens;        // token_embd.weight")
    code.append("    uint64_t final_norm;          // output_norm.weight")
    code.append("    uint64_t lm_head;             // output.weight")
    code.append(f"    Qwen2LayerOffsets layers[{num_layers}];")
    code.append("};")
    code.append("")
    code.append("static const Qwen2Offsets g_qwen2_offsets = {")
    
    embed_off = tensors['token_embd.weight']['offset']
    final_norm_off = tensors['output_norm.weight']['offset']
    lm_head_off = tensors['output.weight']['offset']
    
    code.append(f"    .embed_tokens = {embed_off}ULL,")
    code.append(f"    .final_norm   = {final_norm_off}ULL,")
    code.append(f"    .lm_head      = {lm_head_off}ULL,")
    code.append("    .layers = {")
    
    for l in range(num_layers):
        in_ln   = tensors[f'blk.{l}.attn_norm.weight']['offset']
        q_p     = tensors[f'blk.{l}.attn_q.weight']['offset']
        q_b     = tensors[f'blk.{l}.attn_q.bias']['offset']
        k_p     = tensors[f'blk.{l}.attn_k.weight']['offset']
        k_b     = tensors[f'blk.{l}.attn_k.bias']['offset']
        v_p     = tensors[f'blk.{l}.attn_v.weight']['offset']
        v_b     = tensors[f'blk.{l}.attn_v.bias']['offset']
        o_p     = tensors[f'blk.{l}.attn_output.weight']['offset']
        post_ln = tensors[f'blk.{l}.ffn_norm.weight']['offset']
        gate_p  = tensors[f'blk.{l}.ffn_gate.weight']['offset']
        up_p    = tensors[f'blk.{l}.ffn_up.weight']['offset']
        down_p  = tensors[f'blk.{l}.ffn_down.weight']['offset']
        
        code.append(f"        /* Layer {l} */")
        code.append("        {")
        code.append(f"            .input_layernorm     = {in_ln}ULL,")
        code.append(f"            .q_proj              = {q_p}ULL,")
        code.append(f"            .q_bias              = {q_b}ULL,")
        code.append(f"            .k_proj              = {k_p}ULL,")
        code.append(f"            .k_bias              = {k_b}ULL,")
        code.append(f"            .v_proj              = {v_p}ULL,")
        code.append(f"            .v_bias              = {v_b}ULL,")
        code.append(f"            .o_proj              = {o_p}ULL,")
        code.append(f"            .post_attn_layernorm = {post_ln}ULL,")
        code.append(f"            .gate_proj           = {gate_p}ULL,")
        code.append(f"            .up_proj             = {up_p}ULL,")
        code.append(f"            .down_proj           = {down_p}ULL")
        code.append("        },")
        
    code.append("    }")
    code.append("};")
    code.append("")
    code.append("#endif /* QWEN2_0_5B_LAYOUT_H */")
    
    with open(out_header_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(code))
    print(f"Generated layout header: {out_header_path}")

if __name__ == "__main__":
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else r"H:\ai model\qwen2.5-coder-0.5b-instruct-q8_0.gguf"
    out_path = sys.argv[2] if len(sys.argv) > 2 else r"H:\asurrini\src\vulkan\qwen2_0_5b_layout.h"
    generate_header(gguf_path, out_path)
