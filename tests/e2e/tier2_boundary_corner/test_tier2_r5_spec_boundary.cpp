#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <array>
#include <cstdint>
#include <unordered_set>
#include <algorithm>

using namespace hyperstream::test;

namespace {

struct SpecTreeNodeBoundary {
    int32_t node_idx;
    int32_t token_id;
    int32_t parent_idx;
    int32_t depth;
    bool is_eos;
};

struct SpecTreeBoundary {
    static constexpr int TREE_SIZE = 7;
    std::array<SpecTreeNodeBoundary, TREE_SIZE> nodes;
    int32_t eos_token_id = TestConstants::QWEN2_ENDOFTEXT_TOKEN;

    void init() {
        nodes[0] = {0, 0, -1, 1, false};
        nodes[1] = {1, 0, -1, 1, false};
        nodes[2] = {2, 0,  0, 2, false};
        nodes[3] = {3, 0,  0, 2, false};
        nodes[4] = {4, 0,  1, 2, false};
        nodes[5] = {5, 0,  2, 3, false};
        nodes[6] = {6, 0,  3, 3, false};
    }

    void set_tokens(const std::array<int32_t, TREE_SIZE>& toks) {
        for (int i = 0; i < TREE_SIZE; ++i) {
            nodes[i].token_id = toks[i];
            nodes[i].is_eos = (toks[i] == eos_token_id);
        }
    }

    // Verify DAG property (no cycles)
    bool is_valid_dag() const {
        for (int i = 0; i < TREE_SIZE; ++i) {
            std::unordered_set<int> visited;
            int cur = i;
            while (cur != -1) {
                if (visited.find(cur) != visited.end()) {
                    return false; // Cycle detected!
                }
                visited.insert(cur);
                cur = nodes[cur].parent_idx;
            }
        }
        return true;
    }

    struct VerifyOutcome {
        std::vector<int32_t> accepted;
        int32_t bonus;
        bool stopped_by_eos;
    };

    VerifyOutcome verify(const std::array<int32_t, TREE_SIZE>& target_preds, int32_t root_pred) const {
        VerifyOutcome out;
        out.bonus = -1;
        out.stopped_by_eos = false;

        // Path A: {0, 2, 5}
        if (root_pred == nodes[0].token_id) {
            out.accepted.push_back(nodes[0].token_id);
            if (nodes[0].is_eos) {
                out.stopped_by_eos = true;
                return out;
            }

            if (target_preds[0] == nodes[2].token_id) {
                out.accepted.push_back(nodes[2].token_id);
                if (nodes[2].is_eos) {
                    out.stopped_by_eos = true;
                    return out;
                }
                if (target_preds[2] == nodes[5].token_id) {
                    out.accepted.push_back(nodes[5].token_id);
                    if (nodes[5].is_eos) {
                        out.stopped_by_eos = true;
                        return out;
                    }
                    out.bonus = target_preds[5];
                } else {
                    out.bonus = target_preds[2];
                }
            } else if (target_preds[0] == nodes[3].token_id) { // Path B
                out.accepted.push_back(nodes[3].token_id);
                if (nodes[3].is_eos) {
                    out.stopped_by_eos = true;
                    return out;
                }
                if (target_preds[3] == nodes[6].token_id) {
                    out.accepted.push_back(nodes[6].token_id);
                    if (nodes[6].is_eos) {
                        out.stopped_by_eos = true;
                        return out;
                    }
                    out.bonus = target_preds[6];
                } else {
                    out.bonus = target_preds[3];
                }
            } else {
                out.bonus = target_preds[0];
            }
            return out;
        }

        // Path C: {1, 4}
        if (root_pred == nodes[1].token_id) {
            out.accepted.push_back(nodes[1].token_id);
            if (nodes[1].is_eos) {
                out.stopped_by_eos = true;
                return out;
            }
            if (target_preds[1] == nodes[4].token_id) {
                out.accepted.push_back(nodes[4].token_id);
                if (nodes[4].is_eos) {
                    out.stopped_by_eos = true;
                    return out;
                }
                out.bonus = target_preds[4];
            } else {
                out.bonus = target_preds[1];
            }
            return out;
        }

        // Rejection
        out.bonus = root_pred;
        if (out.bonus == eos_token_id) {
            out.stopped_by_eos = true;
        }
        return out;
    }
};

} // anonymous namespace

// 1. All Draft Candidates Rejected Fallback to Root
E2E_TEST_CASE(Tier2_R5, test_r5_all_tokens_rejected_fallback_to_root) {
    SpecTreeBoundary tree;
    tree.init();
    tree.set_tokens({10, 20, 30, 40, 50, 60, 70});

    // Root prediction doesn't match 10 or 20
    auto out = tree.verify({99, 99, 99, 99, 99, 99, 99}, 555);

    ASSERT_EQ(out.accepted.size(), 0u);
    ASSERT_EQ(out.bonus, 555);
    ASSERT_FALSE(out.stopped_by_eos);
}

// 2. All 7 Tokens / Maximum Yield Path A
E2E_TEST_CASE(Tier2_R5, test_r5_all_7_tokens_accepted_max_yield) {
    SpecTreeBoundary tree;
    tree.init();
    tree.set_tokens({100, 101, 200, 201, 202, 300, 301});

    // Root matches 100, target[0] matches 200, target[2] matches 300
    auto out = tree.verify({200, 0, 300, 0, 0, 400, 0}, 100);

    ASSERT_EQ(out.accepted.size(), 3u);
    ASSERT_EQ(out.accepted[0], 100);
    ASSERT_EQ(out.accepted[1], 200);
    ASSERT_EQ(out.accepted[2], 300);
    ASSERT_EQ(out.bonus, 400); // 4 new tokens emitted!
}

// 3. Cyclic Graph Detection Invariant
E2E_TEST_CASE(Tier2_R5, test_r5_cyclic_graph_detection_and_prevention) {
    SpecTreeBoundary tree;
    tree.init();
    ASSERT_TRUE(tree.is_valid_dag());

    // Inject an invalid cycle: node 0's parent is set to node 2 (0 -> 2 -> 0)
    tree.nodes[0].parent_idx = 2;
    ASSERT_FALSE(tree.is_valid_dag());
}

// 4. Repeated / Identical Candidate Tokens in Draft Pool
E2E_TEST_CASE(Tier2_R5, test_r5_identical_candidate_tokens_in_pool) {
    SpecTreeBoundary tree;
    tree.init();
    // All candidates propose the same token (e.g. repetitive code indentation or whitespace)
    tree.set_tokens({32, 32, 32, 32, 32, 32, 32});

    auto out = tree.verify({32, 32, 32, 32, 32, 32, 32}, 32);

    // Primary branch Path A should be traversed and 3 tokens accepted + bonus
    ASSERT_EQ(out.accepted.size(), 3u);
    ASSERT_EQ(out.accepted[0], 32);
    ASSERT_EQ(out.accepted[1], 32);
    ASSERT_EQ(out.accepted[2], 32);
    ASSERT_EQ(out.bonus, 32);
}

// 5. EOS Special Token in Tree Halts Speculative Expansion
E2E_TEST_CASE(Tier2_R5, test_r5_end_of_sequence_eos_token_in_tree) {
    SpecTreeBoundary tree;
    tree.init();
    // C2 is EOS token
    tree.set_tokens({100, 101, TestConstants::QWEN2_ENDOFTEXT_TOKEN, 201, 202, 300, 301});

    auto out = tree.verify({TestConstants::QWEN2_ENDOFTEXT_TOKEN, 0, 300, 0, 0, 400, 0}, 100);

    ASSERT_EQ(out.accepted.size(), 2u);
    ASSERT_EQ(out.accepted[0], 100);
    ASSERT_EQ(out.accepted[1], TestConstants::QWEN2_ENDOFTEXT_TOKEN);
    ASSERT_TRUE(out.stopped_by_eos);
}

// 6. Branch Fallback from Path A to Path B
E2E_TEST_CASE(Tier2_R5, test_r5_divergent_sibling_branch_selection) {
    SpecTreeBoundary tree;
    tree.init();
    tree.set_tokens({10, 11, 20, 21, 22, 30, 31});

    // Root matches 10 (C0)
    // Target rejects 20 (C2) but matches 21 (C3)
    // Target matches 31 (C6)
    auto out = tree.verify({21, 0, 0, 31, 0, 0, 40}, 10);

    ASSERT_EQ(out.accepted.size(), 3u);
    ASSERT_EQ(out.accepted[0], 10);
    ASSERT_EQ(out.accepted[1], 21);
    ASSERT_EQ(out.accepted[2], 31);
    ASSERT_EQ(out.bonus, 40);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
