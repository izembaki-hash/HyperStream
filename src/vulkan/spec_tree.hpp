#ifndef HYPERSTREAM_SPEC_TREE_HPP
#define HYPERSTREAM_SPEC_TREE_HPP

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>

namespace hyperstream {

struct SpecTreeNode {
    int32_t node_idx = 0;
    int32_t token_id = 0;
    int32_t parent_idx = -1; // -1 for root
    int32_t depth = 1;
    int32_t delta_pos = 0;
    std::vector<int32_t> children;
};

struct SpecTree {
    static constexpr int TREE_SIZE = 7;
    std::array<SpecTreeNode, TREE_SIZE> nodes;
    std::array<float, TREE_SIZE * TREE_SIZE> attention_mask; // 7x7 matrix: 1.0f or 0.0f
    std::vector<std::vector<int32_t>> paths;

    SpecTree() {
        init_topology();
    }

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

        // Distinct candidate generation paths
        paths = {
            {0, 2, 5}, // Path A (depth 3)
            {0, 3, 6}, // Path B (depth 3)
            {1, 4}     // Path C (depth 2)
        };
    }

    void set_candidates(const std::array<int32_t, TREE_SIZE>& tokens) {
        for (int i = 0; i < TREE_SIZE; ++i) {
            nodes[i].token_id = tokens[i];
        }
    }

    struct VerifyResult {
        std::vector<int32_t> accepted_tokens;
        int32_t bonus_token = -1;
        int chosen_path_idx = -1;
    };

    // Evaluates target model predictions against the tree
    // target_preds[i] represents the token predicted by the target model at node i's input
    VerifyResult verify(int32_t base_token_prediction, const std::array<int32_t, TREE_SIZE>& target_preds) const {
        VerifyResult result;
        int best_path = -1;
        size_t max_accepted = 0;
        int32_t best_bonus = -1;
        std::vector<int32_t> best_accepted;

        for (size_t p = 0; p < paths.size(); ++p) {
            const auto& path = paths[p];
            std::vector<int32_t> cur_acc;
            int32_t cur_bonus = -1;

            // First node in path must match base_token_prediction
            if (nodes[path[0]].token_id == base_token_prediction) {
                cur_acc.push_back(nodes[path[0]].token_id);
                cur_bonus = target_preds[path[0]];

                // Check subsequent nodes along this path
                for (size_t step = 1; step < path.size(); ++step) {
                    int node_idx = path[step];
                    int prev_idx = path[step - 1];
                    if (nodes[node_idx].token_id == target_preds[prev_idx]) {
                        cur_acc.push_back(nodes[node_idx].token_id);
                        cur_bonus = target_preds[node_idx];
                    } else {
                        break; // Stop at first mismatch in path
                    }
                }
            }

            if (cur_acc.size() > max_accepted) {
                max_accepted = cur_acc.size();
                best_path = static_cast<int>(p);
                best_accepted = cur_acc;
                best_bonus = cur_bonus;
            }
        }

        result.accepted_tokens = best_accepted;
        result.bonus_token = (best_bonus != -1) ? best_bonus : base_token_prediction;
        result.chosen_path_idx = best_path;
        return result;
    }
};

} // namespace hyperstream

#endif // HYPERSTREAM_SPEC_TREE_HPP
