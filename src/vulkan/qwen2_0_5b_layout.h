/* Auto-generated Qwen2.5-Coder-0.5B Layout Header */
#ifndef QWEN2_0_5B_LAYOUT_H
#define QWEN2_0_5B_LAYOUT_H

#include <stdint.h>

#define QWEN2_DATA_START_OFFSET 5947776ULL
#define QWEN2_VOCAB_SIZE 151936
#define QWEN2_HIDDEN_SIZE 896
#define QWEN2_INTERMEDIATE_SIZE 4864
#define QWEN2_NUM_LAYERS 24
#define QWEN2_NUM_HEADS 14
#define QWEN2_NUM_KV_HEADS 2
#define QWEN2_HEAD_DIM 64
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
    Qwen2LayerOffsets layers[24];
};

static const Qwen2Offsets g_qwen2_offsets = {
    .embed_tokens = 144643072ULL,
    .final_norm   = 669759488ULL,
    .lm_head      = 0ULL,
    .layers = {
        /* Layer 0 */
        {
            .input_layernorm     = 289286144ULL,
            .q_proj              = 304163840ULL,
            .q_bias              = 304160256ULL,
            .k_proj              = 303185408ULL,
            .k_bias              = 303184896ULL,
            .v_proj              = 305017344ULL,
            .v_bias              = 305016832ULL,
            .o_proj              = 303307264ULL,
            .post_attn_layernorm = 303181312ULL,
            .gate_proj           = 293920256ULL,
            .up_proj             = 298550784ULL,
            .down_proj           = 289289728ULL
        },
        /* Layer 1 */
        {
            .input_layernorm     = 305139200ULL,
            .q_proj              = 320016896ULL,
            .q_bias              = 320013312ULL,
            .k_proj              = 319038464ULL,
            .k_bias              = 319037952ULL,
            .v_proj              = 320870400ULL,
            .v_bias              = 320869888ULL,
            .o_proj              = 319160320ULL,
            .post_attn_layernorm = 319034368ULL,
            .gate_proj           = 309773312ULL,
            .up_proj             = 314403840ULL,
            .down_proj           = 305142784ULL
        },
        /* Layer 2 */
        {
            .input_layernorm     = 479522816ULL,
            .q_proj              = 494400512ULL,
            .q_bias              = 494396928ULL,
            .k_proj              = 493422080ULL,
            .k_bias              = 493421568ULL,
            .v_proj              = 495254016ULL,
            .v_bias              = 495253504ULL,
            .o_proj              = 493543936ULL,
            .post_attn_layernorm = 493417984ULL,
            .gate_proj           = 484156928ULL,
            .up_proj             = 488787456ULL,
            .down_proj           = 479526400ULL
        },
        /* Layer 3 */
        {
            .input_layernorm     = 558788096ULL,
            .q_proj              = 573665792ULL,
            .q_bias              = 573662208ULL,
            .k_proj              = 572687360ULL,
            .k_bias              = 572686848ULL,
            .v_proj              = 574519296ULL,
            .v_bias              = 574518784ULL,
            .o_proj              = 572809216ULL,
            .post_attn_layernorm = 572683264ULL,
            .gate_proj           = 563422208ULL,
            .up_proj             = 568052736ULL,
            .down_proj           = 558791680ULL
        },
        /* Layer 4 */
        {
            .input_layernorm     = 574641152ULL,
            .q_proj              = 589518848ULL,
            .q_bias              = 589515264ULL,
            .k_proj              = 588540416ULL,
            .k_bias              = 588539904ULL,
            .v_proj              = 590372352ULL,
            .v_bias              = 590371840ULL,
            .o_proj              = 588662272ULL,
            .post_attn_layernorm = 588536320ULL,
            .gate_proj           = 579275264ULL,
            .up_proj             = 583905792ULL,
            .down_proj           = 574644736ULL
        },
        /* Layer 5 */
        {
            .input_layernorm     = 590494208ULL,
            .q_proj              = 605371904ULL,
            .q_bias              = 605368320ULL,
            .k_proj              = 604393472ULL,
            .k_bias              = 604392960ULL,
            .v_proj              = 606225408ULL,
            .v_bias              = 606224896ULL,
            .o_proj              = 604515328ULL,
            .post_attn_layernorm = 604389376ULL,
            .gate_proj           = 595128320ULL,
            .up_proj             = 599758848ULL,
            .down_proj           = 590497792ULL
        },
        /* Layer 6 */
        {
            .input_layernorm     = 606347264ULL,
            .q_proj              = 621224960ULL,
            .q_bias              = 621221376ULL,
            .k_proj              = 620246528ULL,
            .k_bias              = 620246016ULL,
            .v_proj              = 622078464ULL,
            .v_bias              = 622077952ULL,
            .o_proj              = 620368384ULL,
            .post_attn_layernorm = 620242432ULL,
            .gate_proj           = 610981376ULL,
            .up_proj             = 615611904ULL,
            .down_proj           = 606350848ULL
        },
        /* Layer 7 */
        {
            .input_layernorm     = 622200320ULL,
            .q_proj              = 637078016ULL,
            .q_bias              = 637074432ULL,
            .k_proj              = 636099584ULL,
            .k_bias              = 636099072ULL,
            .v_proj              = 637931520ULL,
            .v_bias              = 637931008ULL,
            .o_proj              = 636221440ULL,
            .post_attn_layernorm = 636095488ULL,
            .gate_proj           = 626834432ULL,
            .up_proj             = 631464960ULL,
            .down_proj           = 622203904ULL
        },
        /* Layer 8 */
        {
            .input_layernorm     = 638053376ULL,
            .q_proj              = 652931072ULL,
            .q_bias              = 652927488ULL,
            .k_proj              = 651952640ULL,
            .k_bias              = 651952128ULL,
            .v_proj              = 653784576ULL,
            .v_bias              = 653784064ULL,
            .o_proj              = 652074496ULL,
            .post_attn_layernorm = 651948544ULL,
            .gate_proj           = 642687488ULL,
            .up_proj             = 647318016ULL,
            .down_proj           = 638056960ULL
        },
        /* Layer 9 */
        {
            .input_layernorm     = 653906432ULL,
            .q_proj              = 668784128ULL,
            .q_bias              = 668780544ULL,
            .k_proj              = 667805696ULL,
            .k_bias              = 667805184ULL,
            .v_proj              = 669637632ULL,
            .v_bias              = 669637120ULL,
            .o_proj              = 667927552ULL,
            .post_attn_layernorm = 667801600ULL,
            .gate_proj           = 658540544ULL,
            .up_proj             = 663171072ULL,
            .down_proj           = 653910016ULL
        },
        /* Layer 10 */
        {
            .input_layernorm     = 320992256ULL,
            .q_proj              = 335869952ULL,
            .q_bias              = 335866368ULL,
            .k_proj              = 334891520ULL,
            .k_bias              = 334891008ULL,
            .v_proj              = 336723456ULL,
            .v_bias              = 336722944ULL,
            .o_proj              = 335013376ULL,
            .post_attn_layernorm = 334887424ULL,
            .gate_proj           = 325626368ULL,
            .up_proj             = 330256896ULL,
            .down_proj           = 320995840ULL
        },
        /* Layer 11 */
        {
            .input_layernorm     = 336845312ULL,
            .q_proj              = 351723008ULL,
            .q_bias              = 351719424ULL,
            .k_proj              = 350744576ULL,
            .k_bias              = 350744064ULL,
            .v_proj              = 352576512ULL,
            .v_bias              = 352576000ULL,
            .o_proj              = 350866432ULL,
            .post_attn_layernorm = 350740480ULL,
            .gate_proj           = 341479424ULL,
            .up_proj             = 346109952ULL,
            .down_proj           = 336848896ULL
        },
        /* Layer 12 */
        {
            .input_layernorm     = 352698368ULL,
            .q_proj              = 367576064ULL,
            .q_bias              = 367572480ULL,
            .k_proj              = 366597632ULL,
            .k_bias              = 366597120ULL,
            .v_proj              = 368429568ULL,
            .v_bias              = 368429056ULL,
            .o_proj              = 366719488ULL,
            .post_attn_layernorm = 366593536ULL,
            .gate_proj           = 357332480ULL,
            .up_proj             = 361963008ULL,
            .down_proj           = 352701952ULL
        },
        /* Layer 13 */
        {
            .input_layernorm     = 368551424ULL,
            .q_proj              = 383429120ULL,
            .q_bias              = 383425536ULL,
            .k_proj              = 382450688ULL,
            .k_bias              = 382450176ULL,
            .v_proj              = 384282624ULL,
            .v_bias              = 384282112ULL,
            .o_proj              = 382572544ULL,
            .post_attn_layernorm = 382446592ULL,
            .gate_proj           = 373185536ULL,
            .up_proj             = 377816064ULL,
            .down_proj           = 368555008ULL
        },
        /* Layer 14 */
        {
            .input_layernorm     = 384404480ULL,
            .q_proj              = 399282176ULL,
            .q_bias              = 399278592ULL,
            .k_proj              = 398303744ULL,
            .k_bias              = 398303232ULL,
            .v_proj              = 400135680ULL,
            .v_bias              = 400135168ULL,
            .o_proj              = 398425600ULL,
            .post_attn_layernorm = 398299648ULL,
            .gate_proj           = 389038592ULL,
            .up_proj             = 393669120ULL,
            .down_proj           = 384408064ULL
        },
        /* Layer 15 */
        {
            .input_layernorm     = 400257536ULL,
            .q_proj              = 415135232ULL,
            .q_bias              = 415131648ULL,
            .k_proj              = 414156800ULL,
            .k_bias              = 414156288ULL,
            .v_proj              = 415988736ULL,
            .v_bias              = 415988224ULL,
            .o_proj              = 414278656ULL,
            .post_attn_layernorm = 414152704ULL,
            .gate_proj           = 404891648ULL,
            .up_proj             = 409522176ULL,
            .down_proj           = 400261120ULL
        },
        /* Layer 16 */
        {
            .input_layernorm     = 416110592ULL,
            .q_proj              = 430988288ULL,
            .q_bias              = 430984704ULL,
            .k_proj              = 430009856ULL,
            .k_bias              = 430009344ULL,
            .v_proj              = 431841792ULL,
            .v_bias              = 431841280ULL,
            .o_proj              = 430131712ULL,
            .post_attn_layernorm = 430005760ULL,
            .gate_proj           = 420744704ULL,
            .up_proj             = 425375232ULL,
            .down_proj           = 416114176ULL
        },
        /* Layer 17 */
        {
            .input_layernorm     = 431963648ULL,
            .q_proj              = 446841344ULL,
            .q_bias              = 446837760ULL,
            .k_proj              = 445862912ULL,
            .k_bias              = 445862400ULL,
            .v_proj              = 447694848ULL,
            .v_bias              = 447694336ULL,
            .o_proj              = 445984768ULL,
            .post_attn_layernorm = 445858816ULL,
            .gate_proj           = 436597760ULL,
            .up_proj             = 441228288ULL,
            .down_proj           = 431967232ULL
        },
        /* Layer 18 */
        {
            .input_layernorm     = 447816704ULL,
            .q_proj              = 462694400ULL,
            .q_bias              = 462690816ULL,
            .k_proj              = 461715968ULL,
            .k_bias              = 461715456ULL,
            .v_proj              = 463547904ULL,
            .v_bias              = 463547392ULL,
            .o_proj              = 461837824ULL,
            .post_attn_layernorm = 461711872ULL,
            .gate_proj           = 452450816ULL,
            .up_proj             = 457081344ULL,
            .down_proj           = 447820288ULL
        },
        /* Layer 19 */
        {
            .input_layernorm     = 463669760ULL,
            .q_proj              = 478547456ULL,
            .q_bias              = 478543872ULL,
            .k_proj              = 477569024ULL,
            .k_bias              = 477568512ULL,
            .v_proj              = 479400960ULL,
            .v_bias              = 479400448ULL,
            .o_proj              = 477690880ULL,
            .post_attn_layernorm = 477564928ULL,
            .gate_proj           = 468303872ULL,
            .up_proj             = 472934400ULL,
            .down_proj           = 463673344ULL
        },
        /* Layer 20 */
        {
            .input_layernorm     = 495375872ULL,
            .q_proj              = 510253568ULL,
            .q_bias              = 510249984ULL,
            .k_proj              = 509275136ULL,
            .k_bias              = 509274624ULL,
            .v_proj              = 511107072ULL,
            .v_bias              = 511106560ULL,
            .o_proj              = 509396992ULL,
            .post_attn_layernorm = 509271040ULL,
            .gate_proj           = 500009984ULL,
            .up_proj             = 504640512ULL,
            .down_proj           = 495379456ULL
        },
        /* Layer 21 */
        {
            .input_layernorm     = 511228928ULL,
            .q_proj              = 526106624ULL,
            .q_bias              = 526103040ULL,
            .k_proj              = 525128192ULL,
            .k_bias              = 525127680ULL,
            .v_proj              = 526960128ULL,
            .v_bias              = 526959616ULL,
            .o_proj              = 525250048ULL,
            .post_attn_layernorm = 525124096ULL,
            .gate_proj           = 515863040ULL,
            .up_proj             = 520493568ULL,
            .down_proj           = 511232512ULL
        },
        /* Layer 22 */
        {
            .input_layernorm     = 527081984ULL,
            .q_proj              = 541959680ULL,
            .q_bias              = 541956096ULL,
            .k_proj              = 540981248ULL,
            .k_bias              = 540980736ULL,
            .v_proj              = 542813184ULL,
            .v_bias              = 542812672ULL,
            .o_proj              = 541103104ULL,
            .post_attn_layernorm = 540977152ULL,
            .gate_proj           = 531716096ULL,
            .up_proj             = 536346624ULL,
            .down_proj           = 527085568ULL
        },
        /* Layer 23 */
        {
            .input_layernorm     = 542935040ULL,
            .q_proj              = 557812736ULL,
            .q_bias              = 557809152ULL,
            .k_proj              = 556834304ULL,
            .k_bias              = 556833792ULL,
            .v_proj              = 558666240ULL,
            .v_bias              = 558665728ULL,
            .o_proj              = 556956160ULL,
            .post_attn_layernorm = 556830208ULL,
            .gate_proj           = 547569152ULL,
            .up_proj             = 552199680ULL,
            .down_proj           = 542938624ULL
        },
    }
};

#endif /* QWEN2_0_5B_LAYOUT_H */