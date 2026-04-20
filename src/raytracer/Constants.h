#pragma once

// Shared numeric constants for the ray-tracing kernels. Keeping these named
// and in one place makes it obvious that "1e-8f" and "0.001f" are two
// different epsilons that serve different purposes.

// Parallel-ray cull threshold. Used in Moller-Trumbore and ray-plane
// intersections to reject rays whose direction is (numerically) parallel to
// the surface, which would otherwise produce a division by a near-zero
// denominator.
constexpr float RAY_PARALLEL_EPS = 1e-8f;

// Self-intersection offset for secondary rays. Applied to t_min when tracing
// bounce/shadow/NEE rays from a hit point so we do not re-intersect the
// surface we just left. Tuned for scene-scale geometry (~unit-sized objects).
constexpr float T_SELF_INTERSECT = 0.001f;
