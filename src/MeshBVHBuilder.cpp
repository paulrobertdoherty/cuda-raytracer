#include "MeshBVHBuilder.h"

#include <algorithm>
#include <limits>
#include <numeric>

// ---- helpers ----------------------------------------------------------------

static constexpr int NUM_BINS = 16;

struct TriAABB {
    glm::vec3 bmin;
    glm::vec3 bmax;
    glm::vec3 centroid;
};

static TriAABB compute_tri_aabb(const glm::vec3* positions,
                                const unsigned int* indices,
                                int tri_id) {
    glm::vec3 v0 = positions[indices[3 * tri_id + 0]];
    glm::vec3 v1 = positions[indices[3 * tri_id + 1]];
    glm::vec3 v2 = positions[indices[3 * tri_id + 2]];

    TriAABB t;
    t.bmin = glm::min(glm::min(v0, v1), v2);
    t.bmax = glm::max(glm::max(v0, v1), v2);
    t.centroid = (v0 + v1 + v2) / 3.0f;
    return t;
}

static float surface_area(const glm::vec3& bmin, const glm::vec3& bmax) {
    glm::vec3 d = bmax - bmin;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// ---- SAH binned split -------------------------------------------------------

struct SplitResult {
    int axis;
    float pos;
    bool found;
};

static SplitResult sah_binned_split(
    const std::vector<TriAABB>& tri_info,
    const int* tri_ids,
    int start, int count,
    const glm::vec3& parent_min, const glm::vec3& parent_max)
{
    SplitResult best{-1, 0.0f, false};
    float best_cost = std::numeric_limits<float>::max();

    float parent_sa = surface_area(parent_min, parent_max);
    if (parent_sa <= 0.0f) return best;

    for (int axis = 0; axis < 3; axis++) {
        float axis_min = parent_min[axis];
        float axis_max = parent_max[axis];
        float extent = axis_max - axis_min;
        if (extent < 1e-6f) continue;

        float bin_width = extent / NUM_BINS;

        // Accumulate per-bin data.
        struct Bin {
            glm::vec3 bmin{std::numeric_limits<float>::max()};
            glm::vec3 bmax{std::numeric_limits<float>::lowest()};
            int count = 0;
        };
        Bin bins[NUM_BINS];

        for (int i = 0; i < count; i++) {
            const TriAABB& ti = tri_info[tri_ids[start + i]];
            int b = (int)((ti.centroid[axis] - axis_min) / bin_width);
            b = std::clamp(b, 0, NUM_BINS - 1);
            bins[b].bmin = glm::min(bins[b].bmin, ti.bmin);
            bins[b].bmax = glm::max(bins[b].bmax, ti.bmax);
            bins[b].count++;
        }

        // Prefix sweep from left.
        glm::vec3 left_min[NUM_BINS - 1];
        glm::vec3 left_max[NUM_BINS - 1];
        int left_count[NUM_BINS - 1];
        {
            glm::vec3 cmin = glm::vec3(std::numeric_limits<float>::max());
            glm::vec3 cmax = glm::vec3(std::numeric_limits<float>::lowest());
            int ccount = 0;
            for (int i = 0; i < NUM_BINS - 1; i++) {
                cmin = glm::min(cmin, bins[i].bmin);
                cmax = glm::max(cmax, bins[i].bmax);
                ccount += bins[i].count;
                left_min[i] = cmin;
                left_max[i] = cmax;
                left_count[i] = ccount;
            }
        }

        // Suffix sweep from right.
        glm::vec3 right_min[NUM_BINS - 1];
        glm::vec3 right_max[NUM_BINS - 1];
        int right_count[NUM_BINS - 1];
        {
            glm::vec3 cmin = glm::vec3(std::numeric_limits<float>::max());
            glm::vec3 cmax = glm::vec3(std::numeric_limits<float>::lowest());
            int ccount = 0;
            for (int i = NUM_BINS - 1; i >= 1; i--) {
                cmin = glm::min(cmin, bins[i].bmin);
                cmax = glm::max(cmax, bins[i].bmax);
                ccount += bins[i].count;
                right_min[i - 1] = cmin;
                right_max[i - 1] = cmax;
                right_count[i - 1] = ccount;
            }
        }

        // Evaluate each split plane.
        for (int i = 0; i < NUM_BINS - 1; i++) {
            if (left_count[i] == 0 || right_count[i] == 0) continue;
            float lsa = surface_area(left_min[i], left_max[i]);
            float rsa = surface_area(right_min[i], right_max[i]);
            float cost = (static_cast<float>(left_count[i]) * lsa + static_cast<float>(right_count[i]) * rsa) / parent_sa;
            if (cost < best_cost) {
                best_cost = cost;
                best.axis = axis;
                best.pos = axis_min + static_cast<float>(i + 1) * bin_width;
                best.found = true;
            }
        }
    }

    return best;
}

// ---- recursive build --------------------------------------------------------

static int build_recursive(
    std::vector<MeshBVHNode>& nodes,
    int* tri_ids,
    int start, int count,
    const std::vector<TriAABB>& tri_info,
    int max_leaf_size)
{
    // Reserve a slot.
    int node_idx = (int)nodes.size();
    nodes.push_back(MeshBVHNode{});

    // Compute AABB of all triangles in [start, start+count).
    glm::vec3 bmin(std::numeric_limits<float>::max());
    glm::vec3 bmax(std::numeric_limits<float>::lowest());
    for (int i = 0; i < count; i++) {
        const TriAABB& ti = tri_info[tri_ids[start + i]];
        bmin = glm::min(bmin, ti.bmin);
        bmax = glm::max(bmax, ti.bmax);
    }

    nodes[node_idx].aabb_min = bmin;
    nodes[node_idx].aabb_max = bmax;

    // Leaf termination.
    if (count <= max_leaf_size) {
        nodes[node_idx].tri_start = start;
        nodes[node_idx].tri_count = count;
        nodes[node_idx].left_child = -1;
        nodes[node_idx].right_child = -1;
        return node_idx;
    }

    // Find best SAH split.
    SplitResult split = sah_binned_split(tri_info, tri_ids, start, count,
                                          bmin, bmax);

    int mid = start;
    if (split.found) {
        // Partition tri_ids around the split plane.
        int lo = start;
        int hi = start + count - 1;
        while (lo <= hi) {
            if (tri_info[tri_ids[lo]].centroid[split.axis] < split.pos) {
                lo++;
            } else {
                std::swap(tri_ids[lo], tri_ids[hi]);
                hi--;
            }
        }
        mid = lo;
    }

    // Fallback: if everything ended up on one side, split in half.
    if (mid == start || mid == start + count) {
        mid = start + count / 2;
    }

    // Recurse.
    int left  = build_recursive(nodes, tri_ids, start, mid - start,
                                tri_info, max_leaf_size);
    int right = build_recursive(nodes, tri_ids, mid, start + count - mid,
                                tri_info, max_leaf_size);

    nodes[node_idx].left_child = left;
    nodes[node_idx].right_child = right;
    nodes[node_idx].tri_start = 0;
    nodes[node_idx].tri_count = 0; // internal node
    return node_idx;
}

// ---- public API -------------------------------------------------------------

MeshBVHBuildResult build_mesh_bvh(
    const glm::vec3* positions,
    const unsigned int* indices,
    int tri_count,
    int max_leaf_size)
{
    MeshBVHBuildResult result;

    if (tri_count <= 0) {
        return result;
    }

    // Precompute per-triangle AABBs and centroids.
    std::vector<TriAABB> tri_info(tri_count);
    for (int i = 0; i < tri_count; i++) {
        tri_info[i] = compute_tri_aabb(positions, indices, i);
    }

    // Working array of triangle IDs that gets partitioned in place.
    std::vector<int> tri_ids(tri_count);
    std::iota(tri_ids.begin(), tri_ids.end(), 0);

    // Reserve a rough estimate to reduce reallocations.
    result.nodes.reserve(2 * tri_count / max_leaf_size);

    build_recursive(result.nodes, tri_ids.data(), 0, tri_count,
                    tri_info, max_leaf_size);

    result.reordered_tri_ids = std::move(tri_ids);
    return result;
}
