#include "e2e/common/e2e_test_framework.hpp"
#include "vulkan/spec_tree.hpp"
#include <iostream>

using namespace hyperstream;
using namespace hyperstream::test;

E2E_TEST_CASE(SpecTreeTest, test_topology_and_mask) {
    SpecTree tree;
    ASSERT_EQ(SpecTree::TREE_SIZE, 7);

    // Verify self-attention
    for (int i = 0; i < 7; ++i) {
        ASSERT_EQ(tree.attention_mask[i * 7 + i], 1.0f);
    }

    // Node 2 parent is 0, so node 2 attends to node 0
    ASSERT_EQ(tree.attention_mask[2 * 7 + 0], 1.0f);

    // Node 5 parent is 2, ancestor is 0: attends to 5, 2, 0
    ASSERT_EQ(tree.attention_mask[5 * 7 + 5], 1.0f);
    ASSERT_EQ(tree.attention_mask[5 * 7 + 2], 1.0f);
    ASSERT_EQ(tree.attention_mask[5 * 7 + 0], 1.0f);

    // Node 5 must NOT attend to sibling branch node 1, 3, 4, 6
    ASSERT_EQ(tree.attention_mask[5 * 7 + 1], 0.0f);
    ASSERT_EQ(tree.attention_mask[5 * 7 + 3], 0.0f);
    ASSERT_EQ(tree.attention_mask[5 * 7 + 4], 0.0f);
    ASSERT_EQ(tree.attention_mask[5 * 7 + 6], 0.0f);
}

E2E_TEST_CASE(SpecTreeTest, test_verification_path_acceptance) {
    SpecTree tree;
    // Set candidate tokens in tree
    // Nodes: 0, 1, 2, 3, 4, 5, 6
    std::array<int32_t, 7> draft_tokens = {101, 102, 201, 202, 203, 301, 302};
    tree.set_candidates(draft_tokens);

    // Scenario: Target model predicted 101 at root
    // then predicted 201 after 101
    // then predicted 301 after 201
    // and predicted 401 after 301 (bonus)
    std::array<int32_t, 7> target_preds = {
        201, // after 0 (which is 101) -> 201
        999, // after 1
        301, // after 2 (which is 201) -> 301
        999, // after 3
        999, // after 4
        401, // after 5 (which is 301) -> 401 (bonus)
        999  // after 6
    };

    auto res = tree.verify(101, target_preds);

    ASSERT_EQ(res.chosen_path_idx, 0); // Path A {0, 2, 5}
    ASSERT_EQ(res.accepted_tokens.size(), 3u);
    ASSERT_EQ(res.accepted_tokens[0], 101);
    ASSERT_EQ(res.accepted_tokens[1], 201);
    ASSERT_EQ(res.accepted_tokens[2], 301);
    ASSERT_EQ(res.bonus_token, 401);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
