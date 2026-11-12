#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>

using namespace hyperstream::test;

namespace {

// Spec-Tree Data Structures
struct SpecTreeNode {
    int32_t node_idx;
    int32_t token_id;
    int32_t parent_idx; // -1 for root
    int32_t depth;
    int32_t delta_pos;
    std::vector<int32_t> children;
};

struct SpecTree {
    static constexpr int TREE_SIZE = 7;
    std::array<SpecTreeNode, TREE_SIZE> nodes;
    std::array<float, TREE_SIZE * TREE_SIZE> attention_mask; // 1.0f or 0.0f
    std::vector<std::vector<int32_t>> paths;

    void init_topology() {
        // Tree Roster:
        // C0: parent -1, depth 1, delta_pos 0
        // C1: parent -1, depth 1, delta_pos 0
        // C2: parent 0,  depth 2, delta_pos 1
        // C3: parent 0,  depth 2, delta_pos 1
        // C4: parent 1,  depth 2, delta_pos 1
        // C5: parent 2,  depth 3, delta_pos 2
        // C6: parent 3,  depth 3, delta_pos 2

        nodes[0] = {0, 0, -1, 1, 0, {2, 3}};
        nodes[1] = {1, 0, -1, 1, 0, {4}};
        nodes[2] = {2, 0,  0, 2, 1, {5}};
        nodes[3] = {3, 0,  0, 2, 1, {6}};
        nodes[4] = {4, 0,  1, 2, 1, {}};
        nodes[5] = {5, 0,  2, 3, 2, {}};
        nodes[6] = {6, 0,  3, 3, 2, {}};

        // Build 7x7 2D tree attention mask
        attention_mask.fill(0.0f);
        for (int i = 0; i < TREE_SIZE; ++i) {
            // A node can always attend to itself
            attention_mask[i * TREE_SIZE + i] = 1.0f;
            // Traverse ancestors
            int cur_parent = nodes[i].parent_idx;
            while (cur_parent != -1) {
                attention_mask[i * TREE_SIZE + cur_parent] = 1.0f;
                cur_parent = nodes[cur_parent].parent_idx;
            }
        }

        // Distinct generation paths
        paths = {
            {0, 2, 5}, // Path A
            {0, 3, 6}, // Path B
            {1, 4}     // Path C
        };
    }

    // Set drafted token candidates into tree
    void set_candidates(const std::array<int32_t, TREE_SIZE>& tokens) {
        for (int i = 0; i < TREE_SIZE; ++i) {
            nodes[i].token_id = tokens[i];
        }
    }

    // Verification algorithm given target predictions for each node position
    struct VerifyResult {
        std::vector<int32_t> accepted_tokens;
        int32_t bonus_token;
        int chosen_path_idx;
    };

    VerifyResult verify(const std::array<int32_t, TREE_SIZE>& target_predictions, int32_t root_target_pred) const {
        // Target predictions:
        // root evaluates prediction for level 1 (compares against C0 / C1)
        // target_predictions[0] evaluates prediction for level 2 under C0 (compares against C2 / C3)
        // target_predictions[2] evaluates prediction for level 3 under C2 (compares against C5)
        // target_predictions[5] generates bonus token for Path A if all accepted

        VerifyResult res;
        res.bonus_token = -1;
        res.chosen_path_idx = -1;

        // Path A Check: {0, 2, 5}
        if (root_target_pred == nodes[0].token_id) {
            res.accepted_tokens.push_back(nodes[0].token_id);
            res.chosen_path_idx = 0;

            if (target_predictions[0] == nodes[2].token_id) {
                res.accepted_tokens.push_back(nodes[2].token_id);
                if (target_predictions[2] == nodes[5].token_id) {
                    res.accepted_tokens.push_back(nodes[5].token_id);
                    res.bonus_token = target_predictions[5];
                } else {
                    res.bonus_token = target_predictions[2]; // Correction token
                }
            } else if (target_predictions[0] == nodes[3].token_id) { // Fallback to Path B
                res.chosen_path_idx = 1;
                res.accepted_tokens.push_back(nodes[3].token_id);
                if (target_predictions[3] == nodes[6].token_id) {
                    res.accepted_tokens.push_back(nodes[6].token_id);
                    res.bonus_token = target_predictions[6];
                } else {
                    res.bonus_token = target_predictions[3];
                }
            } else {
                res.bonus_token = target_predictions[0];
            }
            return res;
        }

        // Path C Check: {1, 4} (when C0 is rejected)
        if (root_target_pred == nodes[1].token_id) {
            res.accepted_tokens.push_back(nodes[1].token_id);
            res.chosen_path_idx = 2;
            if (target_predictions[1] == nodes[4].token_id) {
                res.accepted_tokens.push_back(nodes[4].token_id);
                res.bonus_token = target_predictions[4];
            } else {
                res.bonus_token = target_predictions[1];
            }
            return res;
        }

        // Rejection of both C0 and C1: Fallback to root target prediction
        res.bonus_token = root_target_pred;
        return res;
    }
};

} // anonymous namespace

// 1. Spec-Tree 7-Candidate Topology and Parent Links
E2E_TEST_CASE(Tier1_R5, test_r5_spec_tree_7_candidate_topology_and_parent_links) {
    SpecTree tree;
    tree.init_topology();

    ASSERT_EQ(SpecTree::TREE_SIZE, 7);
    ASSERT_EQ(tree.nodes[0].parent_idx, -1);
    ASSERT_EQ(tree.nodes[1].parent_idx, -1);
    ASSERT_EQ(tree.nodes[2].parent_idx, 0);
    ASSERT_EQ(tree.nodes[3].parent_idx, 0);
    ASSERT_EQ(tree.nodes[4].parent_idx, 1);
    ASSERT_EQ(tree.nodes[5].parent_idx, 2);
    ASSERT_EQ(tree.nodes[6].parent_idx, 3);

    ASSERT_EQ(tree.nodes[0].children.size(), 2u);
    ASSERT_EQ(tree.nodes[0].children[0], 2);
    ASSERT_EQ(tree.nodes[0].children[1], 3);

    ASSERT_EQ(tree.nodes[1].children.size(), 1u);
    ASSERT_EQ(tree.nodes[1].children[0], 4);

    ASSERT_EQ(tree.nodes[2].children.size(), 1u);
    ASSERT_EQ(tree.nodes[2].children[0], 5);

    ASSERT_EQ(tree.nodes[3].children.size(), 1u);
    ASSERT_EQ(tree.nodes[3].children[0], 6);

    ASSERT_EQ(tree.nodes[4].children.size(), 0u);
    ASSERT_EQ(tree.nodes[5].children.size(), 0u);
    ASSERT_EQ(tree.nodes[6].children.size(), 0u);
}

// 2. Exact 7x7 2D Tree Attention Mask Matrix
E2E_TEST_CASE(Tier1_R5, test_r5_2d_tree_attention_mask_exact_matrix) {
    SpecTree tree;
    tree.init_topology();

    const std::array<float, 49> expected_mask = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, // C0
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, // C1
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, // C2 (attends to C0 and self)
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, // C3 (attends to C0 and self)
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // C4 (attends to C1 and self)
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // C5 (attends to C0, C2, and self)
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f  // C6 (attends to C0, C3, and self)
    };

    for (int i = 0; i < 49; ++i) {
        ASSERT_EQ(tree.attention_mask[i], expected_mask[i]);
    }
}

// 3. Tree Paths Traversal Order & Depth Invariants
E2E_TEST_CASE(Tier1_R5, test_r5_tree_paths_traversal_order) {
    SpecTree tree;
    tree.init_topology();

    ASSERT_EQ(tree.paths.size(), 3u);

    // Path A: Length 3
    ASSERT_EQ(tree.paths[0].size(), 3u);
    ASSERT_EQ(tree.paths[0][0], 0);
    ASSERT_EQ(tree.paths[0][1], 2);
    ASSERT_EQ(tree.paths[0][2], 5);

    // Path B: Length 3
    ASSERT_EQ(tree.paths[1].size(), 3u);
    ASSERT_EQ(tree.paths[1][0], 0);
    ASSERT_EQ(tree.paths[1][1], 3);
    ASSERT_EQ(tree.paths[1][2], 6);

    // Path C: Length 2
    ASSERT_EQ(tree.paths[2].size(), 2u);
    ASSERT_EQ(tree.paths[2][0], 1);
    ASSERT_EQ(tree.paths[2][1], 4);
}

// 4. Single-Pass Target Verification Acceptance Logic (All Branches)
E2E_TEST_CASE(Tier1_R5, test_r5_single_pass_target_verification_acceptance_logic) {
    SpecTree tree;
    tree.init_topology();
    tree.set_candidates({100, 101, 200, 201, 202, 300, 301});

    // Scenario 1: Full acceptance of Path A (3 accepted + bonus)
    {
        std::array<int32_t, 7> target_preds = {200, 999, 300, 999, 999, 400, 999};
        auto res = tree.verify(target_preds, 100);
        ASSERT_EQ(res.chosen_path_idx, 0);
        ASSERT_EQ(res.accepted_tokens.size(), 3u);
        ASSERT_EQ(res.accepted_tokens[0], 100);
        ASSERT_EQ(res.accepted_tokens[1], 200);
        ASSERT_EQ(res.accepted_tokens[2], 300);
        ASSERT_EQ(res.bonus_token, 400); // 4 tokens emitted in 1 step!
    }

    // Scenario 2: Divergence to Path B (C0 ok, C2 rejected, C3 and C6 accepted)
    {
        std::array<int32_t, 7> target_preds = {201, 999, 999, 301, 999, 999, 450};
        auto res = tree.verify(target_preds, 100);
        ASSERT_EQ(res.chosen_path_idx, 1);
        ASSERT_EQ(res.accepted_tokens.size(), 3u);
        ASSERT_EQ(res.accepted_tokens[0], 100);
        ASSERT_EQ(res.accepted_tokens[1], 201);
        ASSERT_EQ(res.accepted_tokens[2], 301);
        ASSERT_EQ(res.bonus_token, 450);
    }

    // Scenario 3: Divergence to Path C (C0 rejected, C1 and C4 accepted)
    {
        std::array<int32_t, 7> target_preds = {999, 202, 999, 999, 500, 999, 999};
        auto res = tree.verify(target_preds, 101);
        ASSERT_EQ(res.chosen_path_idx, 2);
        ASSERT_EQ(res.accepted_tokens.size(), 2u);
        ASSERT_EQ(res.accepted_tokens[0], 101);
        ASSERT_EQ(res.accepted_tokens[1], 202);
        ASSERT_EQ(res.bonus_token, 500);
    }

    // Scenario 4: Both C0 and C1 rejected (0 draft accepted, 1 target token emitted)
    {
        std::array<int32_t, 7> target_preds = {999, 999, 999, 999, 999, 999, 999};
        auto res = tree.verify(target_preds, 777);
        ASSERT_EQ(res.chosen_path_idx, -1);
        ASSERT_EQ(res.accepted_tokens.size(), 0u);
        ASSERT_EQ(res.bonus_token, 777);
    }
}

// 5. Relative Position Offsets Delta Pos
E2E_TEST_CASE(Tier1_R5, test_r5_relative_position_offsets_derivation) {
    SpecTree tree;
    tree.init_topology();

    ASSERT_EQ(tree.nodes[0].delta_pos, 0);
    ASSERT_EQ(tree.nodes[1].delta_pos, 0);
    ASSERT_EQ(tree.nodes[2].delta_pos, 1);
    ASSERT_EQ(tree.nodes[3].delta_pos, 1);
    ASSERT_EQ(tree.nodes[4].delta_pos, 1);
    ASSERT_EQ(tree.nodes[5].delta_pos, 2);
    ASSERT_EQ(tree.nodes[6].delta_pos, 2);
}

// 6. Causal Masking Invariant: Strict Ban on Sibling/Divergent Attention
E2E_TEST_CASE(Tier1_R5, test_r5_causal_masking_invariant_no_future_or_sibling_leak) {
    SpecTree tree;
    tree.init_topology();

    auto can_attend = [&](int i, int j) -> bool {
        return tree.attention_mask[i * SpecTree::TREE_SIZE + j] == 1.0f;
    };

    // Sibling pairs must never attend to each other
    ASSERT_FALSE(can_attend(0, 1));
    ASSERT_FALSE(can_attend(1, 0));
    ASSERT_FALSE(can_attend(2, 3));
    ASSERT_FALSE(can_attend(3, 2));

    // Cross-branch nodes must never attend to each other
    ASSERT_FALSE(can_attend(4, 0)); // C4 in branch C cannot attend to C0
    ASSERT_FALSE(can_attend(2, 1)); // C2 in branch A cannot attend to C1
    ASSERT_FALSE(can_attend(5, 3)); // C5 in branch A cannot attend to C3
    ASSERT_FALSE(can_attend(6, 2)); // C6 in branch B cannot attend to C2

    // Future nodes must never be attended to
    ASSERT_FALSE(can_attend(0, 2));
    ASSERT_FALSE(can_attend(0, 5));
    ASSERT_FALSE(can_attend(1, 4));
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
