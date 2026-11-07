/* Auto-generated Qwen2.5-Coder-7B Layout Header */
#ifndef QWEN2_7B_LAYOUT_H
#define QWEN2_7B_LAYOUT_H

#include <stdint.h>

#define QWEN2_DATA_START_OFFSET 5953536ULL
#define QWEN2_VOCAB_SIZE 152064
#define QWEN2_HIDDEN_SIZE 3584
#define QWEN2_INTERMEDIATE_SIZE 18944
#define QWEN2_NUM_LAYERS 28
#define QWEN2_NUM_HEADS 28
#define QWEN2_NUM_KV_HEADS 4
#define QWEN2_HEAD_DIM 128
#define QWEN2_RMS_NORM_EPS 9.999999974752427e-07f
#define QWEN2_ROPE_THETA 1000000.0f

struct Qwen2LayerOffsets {
    uint64_t input_layernorm;     // attn_norm.weight
    uint64_t q_proj;              // attn_q.weight
    uint64_t q_bias;              // attn_q.bias
    uint64_t k_proj;              // attn_k.weight
    uint64_t k_bias;              // attn_k.bias
    uint64_t v_proj;              // attn_v.weight
    uint64_t v_bias;              // attn_v.bias
    uint64_t o_proj;              // attn_output.weight
    uint64_t post_attn_layernorm; // ffn_norm.weight
    uint64_t gate_proj;           // ffn_gate.weight
    uint64_t up_proj;             // ffn_up.weight
    uint64_t down_proj;           // ffn_down.weight
};

struct Qwen2Offsets {
    uint64_t embed_tokens;        // token_embd.weight
    uint64_t final_norm;          // output_norm.weight
    uint64_t lm_head;             // output.weight
    Qwen2LayerOffsets layers[28];
};

static const Qwen2Offsets g_qwen2_offsets = {
    .embed_tokens = 0ULL,
    .final_norm   = 4425422848ULL,
    .lm_head      = 3284457472ULL,
    .layers = {
        /* Layer 0 */
        {
            .input_layernorm     = 306561024ULL,
            .q_proj              = 429436928ULL,
            .q_bias              = 429422592ULL,
            .k_proj              = 421165056ULL,
            .k_bias              = 421163008ULL,
            .v_proj              = 436664320ULL,
            .v_bias              = 436662272ULL,
            .o_proj              = 422197248ULL,
            .post_attn_layernorm = 421148672ULL,
            .gate_proj           = 344766464ULL,
            .up_proj             = 382957568ULL,
            .down_proj           = 306575360ULL
        },
        /* Layer 1 */
        {
            .input_layernorm     = 437696512ULL,
            .q_proj              = 560572416ULL,
            .q_bias              = 560558080ULL,
            .k_proj              = 552300544ULL,
            .k_bias              = 552298496ULL,
            .v_proj              = 567799808ULL,
            .v_bias              = 567797760ULL,
            .o_proj              = 553332736ULL,
            .post_attn_layernorm = 552284160ULL,
            .gate_proj           = 475901952ULL,
            .up_proj             = 514093056ULL,
            .down_proj           = 437710848ULL
        },
        /* Layer 2 */
        {
            .input_layernorm     = 568832000ULL,
            .q_proj              = 691707904ULL,
            .q_bias              = 691693568ULL,
            .k_proj              = 683436032ULL,
            .k_bias              = 683433984ULL,
            .v_proj              = 698935296ULL,
            .v_bias              = 698933248ULL,
            .o_proj              = 684468224ULL,
            .post_attn_layernorm = 683419648ULL,
            .gate_proj           = 607037440ULL,
            .up_proj             = 645228544ULL,
            .down_proj           = 568846336ULL
        },
        /* Layer 3 */
        {
            .input_layernorm     = 699967488ULL,
            .q_proj              = 822843392ULL,
            .q_bias              = 822829056ULL,
            .k_proj              = 814571520ULL,
            .k_bias              = 814569472ULL,
            .v_proj              = 830070784ULL,
            .v_bias              = 830068736ULL,
            .o_proj              = 815603712ULL,
            .post_attn_layernorm = 814555136ULL,
            .gate_proj           = 738172928ULL,
            .up_proj             = 776364032ULL,
            .down_proj           = 699981824ULL
        },
        /* Layer 4 */
        {
            .input_layernorm     = 831102976ULL,
            .q_proj              = 953978880ULL,
            .q_bias              = 953964544ULL,
            .k_proj              = 945707008ULL,
            .k_bias              = 945704960ULL,
            .v_proj              = 961206272ULL,
            .v_bias              = 961204224ULL,
            .o_proj              = 946739200ULL,
            .post_attn_layernorm = 945690624ULL,
            .gate_proj           = 869308416ULL,
            .up_proj             = 907499520ULL,
            .down_proj           = 831117312ULL
        },
        /* Layer 5 */
        {
            .input_layernorm     = 962238464ULL,
            .q_proj              = 1085114368ULL,
            .q_bias              = 1085100032ULL,
            .k_proj              = 1076842496ULL,
            .k_bias              = 1076840448ULL,
            .v_proj              = 1092341760ULL,
            .v_bias              = 1092339712ULL,
            .o_proj              = 1077874688ULL,
            .post_attn_layernorm = 1076826112ULL,
            .gate_proj           = 1000443904ULL,
            .up_proj             = 1038635008ULL,
            .down_proj           = 962252800ULL
        },
        /* Layer 6 */
        {
            .input_layernorm     = 1689174016ULL,
            .q_proj              = 1101647872ULL,
            .q_bias              = 1101633536ULL,
            .k_proj              = 1093376000ULL,
            .k_bias              = 1093373952ULL,
            .v_proj              = 1108875264ULL,
            .v_bias              = 1108873216ULL,
            .o_proj              = 1094408192ULL,
            .post_attn_layernorm = 1803761664ULL,
            .gate_proj           = 1727379456ULL,
            .up_proj             = 1765570560ULL,
            .down_proj           = 1689188352ULL
        },
        /* Layer 7 */
        {
            .input_layernorm     = 1803776000ULL,
            .q_proj              = 1926651904ULL,
            .q_bias              = 1926637568ULL,
            .k_proj              = 1918380032ULL,
            .k_bias              = 1918377984ULL,
            .v_proj              = 1933879296ULL,
            .v_bias              = 1933877248ULL,
            .o_proj              = 1919412224ULL,
            .post_attn_layernorm = 1918363648ULL,
            .gate_proj           = 1841981440ULL,
            .up_proj             = 1880172544ULL,
            .down_proj           = 1803790336ULL
        },
        /* Layer 8 */
        {
            .input_layernorm     = 1934911488ULL,
            .q_proj              = 2057787392ULL,
            .q_bias              = 2057773056ULL,
            .k_proj              = 2049515520ULL,
            .k_bias              = 2049513472ULL,
            .v_proj              = 2065014784ULL,
            .v_bias              = 2065012736ULL,
            .o_proj              = 2050547712ULL,
            .post_attn_layernorm = 2049499136ULL,
            .gate_proj           = 1973116928ULL,
            .up_proj             = 2011308032ULL,
            .down_proj           = 1934925824ULL
        },
        /* Layer 9 */
        {
            .input_layernorm     = 2066046976ULL,
            .q_proj              = 2188922880ULL,
            .q_bias              = 2188908544ULL,
            .k_proj              = 2180651008ULL,
            .k_bias              = 2180648960ULL,
            .v_proj              = 2196150272ULL,
            .v_bias              = 2196148224ULL,
            .o_proj              = 2181683200ULL,
            .post_attn_layernorm = 2180634624ULL,
            .gate_proj           = 2104252416ULL,
            .up_proj             = 2142443520ULL,
            .down_proj           = 2066061312ULL
        },
        /* Layer 10 */
        {
            .input_layernorm     = 1109907456ULL,
            .q_proj              = 1232783360ULL,
            .q_bias              = 1232769024ULL,
            .k_proj              = 1224511488ULL,
            .k_bias              = 1224509440ULL,
            .v_proj              = 1240010752ULL,
            .v_bias              = 1240008704ULL,
            .o_proj              = 1225543680ULL,
            .post_attn_layernorm = 1224495104ULL,
            .gate_proj           = 1148112896ULL,
            .up_proj             = 1186304000ULL,
            .down_proj           = 1109921792ULL
        },
        /* Layer 11 */
        {
            .input_layernorm     = 1241042944ULL,
            .q_proj              = 1363918848ULL,
            .q_bias              = 1363904512ULL,
            .k_proj              = 1355646976ULL,
            .k_bias              = 1355644928ULL,
            .v_proj              = 1371146240ULL,
            .v_bias              = 1371144192ULL,
            .o_proj              = 1356679168ULL,
            .post_attn_layernorm = 1355630592ULL,
            .gate_proj           = 1279248384ULL,
            .up_proj             = 1317439488ULL,
            .down_proj           = 1241057280ULL
        },
        /* Layer 12 */
        {
            .input_layernorm     = 1372178432ULL,
            .q_proj              = 1495054336ULL,
            .q_bias              = 1495040000ULL,
            .k_proj              = 1486782464ULL,
            .k_bias              = 1486780416ULL,
            .v_proj              = 1502281728ULL,
            .v_bias              = 1502279680ULL,
            .o_proj              = 1487814656ULL,
            .post_attn_layernorm = 1486766080ULL,
            .gate_proj           = 1410383872ULL,
            .up_proj             = 1448574976ULL,
            .down_proj           = 1372192768ULL
        },
        /* Layer 13 */
        {
            .input_layernorm     = 1503313920ULL,
            .q_proj              = 1626189824ULL,
            .q_bias              = 1626175488ULL,
            .k_proj              = 1617917952ULL,
            .k_bias              = 1617915904ULL,
            .v_proj              = 1633417216ULL,
            .v_bias              = 1633415168ULL,
            .o_proj              = 1618950144ULL,
            .post_attn_layernorm = 1617901568ULL,
            .gate_proj           = 1541519360ULL,
            .up_proj             = 1579710464ULL,
            .down_proj           = 1503328256ULL
        },
        /* Layer 14 */
        {
            .input_layernorm     = 2197182464ULL,
            .q_proj              = 1680914432ULL,
            .q_bias              = 1680900096ULL,
            .k_proj              = 1672642560ULL,
            .k_bias              = 1672640512ULL,
            .v_proj              = 1688141824ULL,
            .v_bias              = 1688139776ULL,
            .o_proj              = 1673674752ULL,
            .post_attn_layernorm = 2273579008ULL,
            .gate_proj           = 1634449408ULL,
            .up_proj             = 2235387904ULL,
            .down_proj           = 2197196800ULL
        },
        /* Layer 15 */
        {
            .input_layernorm     = 2273593344ULL,
            .q_proj              = 2396469248ULL,
            .q_bias              = 2396454912ULL,
            .k_proj              = 2388197376ULL,
            .k_bias              = 2388195328ULL,
            .v_proj              = 2403696640ULL,
            .v_bias              = 2403694592ULL,
            .o_proj              = 2389229568ULL,
            .post_attn_layernorm = 2388180992ULL,
            .gate_proj           = 2311798784ULL,
            .up_proj             = 2349989888ULL,
            .down_proj           = 2273607680ULL
        },
        /* Layer 16 */
        {
            .input_layernorm     = 2404728832ULL,
            .q_proj              = 2527604736ULL,
            .q_bias              = 2527590400ULL,
            .k_proj              = 2519332864ULL,
            .k_bias              = 2519330816ULL,
            .v_proj              = 2534832128ULL,
            .v_bias              = 2534830080ULL,
            .o_proj              = 2520365056ULL,
            .post_attn_layernorm = 2519316480ULL,
            .gate_proj           = 2442934272ULL,
            .up_proj             = 2481125376ULL,
            .down_proj           = 2404743168ULL
        },
        /* Layer 17 */
        {
            .input_layernorm     = 2535864320ULL,
            .q_proj              = 2658740224ULL,
            .q_bias              = 2658725888ULL,
            .k_proj              = 2650468352ULL,
            .k_bias              = 2650466304ULL,
            .v_proj              = 2665967616ULL,
            .v_bias              = 2665965568ULL,
            .o_proj              = 2651500544ULL,
            .post_attn_layernorm = 2650451968ULL,
            .gate_proj           = 2574069760ULL,
            .up_proj             = 2612260864ULL,
            .down_proj           = 2535878656ULL
        },
        /* Layer 18 */
        {
            .input_layernorm     = 2666999808ULL,
            .q_proj              = 2789875712ULL,
            .q_bias              = 2789861376ULL,
            .k_proj              = 2781603840ULL,
            .k_bias              = 2781601792ULL,
            .v_proj              = 2797103104ULL,
            .v_bias              = 2797101056ULL,
            .o_proj              = 2782636032ULL,
            .post_attn_layernorm = 2781587456ULL,
            .gate_proj           = 2705205248ULL,
            .up_proj             = 2743396352ULL,
            .down_proj           = 2667014144ULL
        },
        /* Layer 19 */
        {
            .input_layernorm     = 2798135296ULL,
            .q_proj              = 2921011200ULL,
            .q_bias              = 2920996864ULL,
            .k_proj              = 2912739328ULL,
            .k_bias              = 2912737280ULL,
            .v_proj              = 2928238592ULL,
            .v_bias              = 2928236544ULL,
            .o_proj              = 2913771520ULL,
            .post_attn_layernorm = 2912722944ULL,
            .gate_proj           = 2836340736ULL,
            .up_proj             = 2874531840ULL,
            .down_proj           = 2798149632ULL
        },
        /* Layer 20 */
        {
            .input_layernorm     = 2929270784ULL,
            .q_proj              = 3052146688ULL,
            .q_bias              = 3052132352ULL,
            .k_proj              = 3043874816ULL,
            .k_bias              = 3043872768ULL,
            .v_proj              = 3059374080ULL,
            .v_bias              = 3059372032ULL,
            .o_proj              = 3044907008ULL,
            .post_attn_layernorm = 3043858432ULL,
            .gate_proj           = 2967476224ULL,
            .up_proj             = 3005667328ULL,
            .down_proj           = 2929285120ULL
        },
        /* Layer 21 */
        {
            .input_layernorm     = 3060406272ULL,
            .q_proj              = 3183282176ULL,
            .q_bias              = 3183267840ULL,
            .k_proj              = 3175010304ULL,
            .k_bias              = 3175008256ULL,
            .v_proj              = 3190509568ULL,
            .v_bias              = 3190507520ULL,
            .o_proj              = 3176042496ULL,
            .post_attn_layernorm = 3174993920ULL,
            .gate_proj           = 3098611712ULL,
            .up_proj             = 3136802816ULL,
            .down_proj           = 3060420608ULL
        },
        /* Layer 22 */
        {
            .input_layernorm     = 3731525632ULL,
            .q_proj              = 3276197888ULL,
            .q_bias              = 3276183552ULL,
            .k_proj              = 3267926016ULL,
            .k_bias              = 3267923968ULL,
            .v_proj              = 3283425280ULL,
            .v_bias              = 3283423232ULL,
            .o_proj              = 3268958208ULL,
            .post_attn_layernorm = 3769731072ULL,
            .gate_proj           = 3191541760ULL,
            .up_proj             = 3229732864ULL,
            .down_proj           = 3731539968ULL
        },
        /* Layer 23 */
        {
            .input_layernorm     = 3769745408ULL,
            .q_proj              = 3892621312ULL,
            .q_bias              = 3892606976ULL,
            .k_proj              = 3884349440ULL,
            .k_bias              = 3884347392ULL,
            .v_proj              = 3899848704ULL,
            .v_bias              = 3899846656ULL,
            .o_proj              = 3885381632ULL,
            .post_attn_layernorm = 3884333056ULL,
            .gate_proj           = 3807950848ULL,
            .up_proj             = 3846141952ULL,
            .down_proj           = 3769759744ULL
        },
        /* Layer 24 */
        {
            .input_layernorm     = 3900880896ULL,
            .q_proj              = 4023756800ULL,
            .q_bias              = 4023742464ULL,
            .k_proj              = 4015484928ULL,
            .k_bias              = 4015482880ULL,
            .v_proj              = 4030984192ULL,
            .v_bias              = 4030982144ULL,
            .o_proj              = 4016517120ULL,
            .post_attn_layernorm = 4015468544ULL,
            .gate_proj           = 3939086336ULL,
            .up_proj             = 3977277440ULL,
            .down_proj           = 3900895232ULL
        },
        /* Layer 25 */
        {
            .input_layernorm     = 4032016384ULL,
            .q_proj              = 4154892288ULL,
            .q_bias              = 4154877952ULL,
            .k_proj              = 4146620416ULL,
            .k_bias              = 4146618368ULL,
            .v_proj              = 4162119680ULL,
            .v_bias              = 4162117632ULL,
            .o_proj              = 4147652608ULL,
            .post_attn_layernorm = 4146604032ULL,
            .gate_proj           = 4070221824ULL,
            .up_proj             = 4108412928ULL,
            .down_proj           = 4032030720ULL
        },
        /* Layer 26 */
        {
            .input_layernorm     = 4163151872ULL,
            .q_proj              = 4286027776ULL,
            .q_bias              = 4286013440ULL,
            .k_proj              = 4277755904ULL,
            .k_bias              = 4277753856ULL,
            .v_proj              = 4293255168ULL,
            .v_bias              = 4293253120ULL,
            .o_proj              = 4278788096ULL,
            .post_attn_layernorm = 4277739520ULL,
            .gate_proj           = 4201357312ULL,
            .up_proj             = 4239548416ULL,
            .down_proj           = 4163166208ULL
        },
        /* Layer 27 */
        {
            .input_layernorm     = 4294287360ULL,
            .q_proj              = 4417163264ULL,
            .q_bias              = 4417148928ULL,
            .k_proj              = 4408891392ULL,
            .k_bias              = 4408889344ULL,
            .v_proj              = 4424390656ULL,
            .v_bias              = 4424388608ULL,
            .o_proj              = 4409923584ULL,
            .post_attn_layernorm = 4408875008ULL,
            .gate_proj           = 4332492800ULL,
            .up_proj             = 4370683904ULL,
            .down_proj           = 4294301696ULL
        },
    }
};

#endif /* QWEN2_0_5B_LAYOUT_H */