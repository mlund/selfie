//! Octree over panel centroids for the Barnes-Hut treecode.
//!
//! Flat storage: `Tree::nodes` is indexed by every helper. A node is
//! either a leaf (holds `panel_ids`, no `children`) or internal (holds
//! child indices into `Tree::nodes`, no `panel_ids`). Build is iterative
//! with an explicit stack — recursion-free so the tree depth never
//! collides with Rust's thread stack on pathological meshes.
//!
//! Reference implementation: the `barnes_hut` crate's public octree
//! (David O'Connor, MIT). Adapted to `glam::DVec3` and a domain-
//! specific `Node` payload; no runtime dep pulled in.

use glam::DVec3;

/// Axis-aligned cube. `half_width` is half the edge length; a point
/// `p` lies inside the cube iff `|p_i − center_i| ≤ half_width` for
/// each axis.
#[derive(Clone, Copy, Debug)]
pub(super) struct Cube {
    pub(super) center: DVec3,
    pub(super) half_width: f64,
}

impl Cube {
    /// Smallest cube that contains every point in `centroids`, padded
    /// by `eps_pad` to keep boundary points strictly inside.
    fn fit(centroids: &[DVec3], eps_pad: f64) -> Self {
        // why: folding into two 3-vectors at once avoids six separate
        // scans of the input for each min / max axis.
        let (lo, hi) = centroids.iter().fold(
            (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
            |(lo, hi), &c| (lo.min(c), hi.max(c)),
        );
        let center = 0.5 * (lo + hi);
        let extent = hi - lo;
        let half_width = 0.5 * extent.max_element() + eps_pad;
        Self {
            center,
            half_width,
        }
    }

    /// Split into 8 equal-volume octants. Octant `i`'s bit pattern
    /// matches the partitioning in [`partition_octant`].
    fn octants(&self) -> [Self; 8] {
        let h = 0.5 * self.half_width;
        std::array::from_fn(|i| {
            let sx = if i & 0b001 != 0 { 1.0 } else { -1.0 };
            let sy = if i & 0b010 != 0 { 1.0 } else { -1.0 };
            let sz = if i & 0b100 != 0 { 1.0 } else { -1.0 };
            Self {
                center: self.center + h * DVec3::new(sx, sy, sz),
                half_width: h,
            }
        })
    }

    /// Conservative bounding-sphere radius for the MAC test (cube
    /// half-diagonal).
    pub(super) fn bounding_radius(&self) -> f64 {
        // √3 · half_width; using the constant form for f64 determinism.
        self.half_width * 1.732_050_807_568_877_2
    }
}

/// Octant index of `p` within `cube` — a bit-packed (x, y, z) sign
/// word matching [`Cube::octants`].
fn partition_octant(p: DVec3, cube: &Cube) -> usize {
    (usize::from(p.x >= cube.center.x))
        | (usize::from(p.y >= cube.center.y) << 1)
        | (usize::from(p.z >= cube.center.z) << 2)
}

/// A node in the tree. Exactly one of `children` and `panel_ids` is
/// non-empty; an internal node routes to its 8 (or fewer) children,
/// a leaf holds the panel indices that landed in its cube.
#[derive(Debug)]
pub(super) struct Node {
    pub(super) bbox: Cube,
    pub(super) children: Vec<u32>,
    pub(super) panel_ids: Vec<u32>,
}

impl Node {
    pub(super) fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Flat-storage octree. `nodes[0]` is the root; every other node is
/// referenced from some parent's `children`.
#[derive(Debug)]
pub(super) struct Tree {
    pub(super) nodes: Vec<Node>,
    pub(super) n_crit: usize,
}

impl Tree {
    /// Build an octree of `centroids`. A node containing > `n_crit`
    /// panels is split into octants; below that (or at max depth) it
    /// is kept as a leaf.
    pub(super) fn new(centroids: &[DVec3], n_crit: usize) -> Self {
        // why: cube-fit pad keeps boundary centroids strictly inside
        // the root cube so `partition_octant`'s `≥ center` branch
        // never disagrees with `inside the cube` at the edge.
        const EPS_PAD: f64 = 1e-9;
        // why: prevents runaway subdivision when two centroids
        // coincide (e.g. a degenerate mesh); at depth 24 the cube is
        // 2⁻²⁴ · root, below any realistic physical feature.
        const MAX_DEPTH: u32 = 24;

        debug_assert!(!centroids.is_empty(), "empty centroid list");
        debug_assert!(n_crit >= 1, "n_crit must be ≥ 1");

        let root_cube = Cube::fit(centroids, EPS_PAD);
        let root_ids: Vec<u32> = (0..centroids.len() as u32).collect();

        let mut nodes: Vec<Node> = Vec::with_capacity(2 * centroids.len() / n_crit + 1);

        // Stack entries: (panel_ids, cube, parent_id, depth). Parent
        // is `None` only for the root.
        let mut stack: Vec<(Vec<u32>, Cube, Option<u32>, u32)> =
            vec![(root_ids, root_cube, None, 0)];

        while let Some((ids, cube, parent, depth)) = stack.pop() {
            let node_id = nodes.len() as u32;
            if let Some(pid) = parent {
                nodes[pid as usize].children.push(node_id);
            }

            if ids.len() <= n_crit || depth >= MAX_DEPTH {
                nodes.push(Node {
                    bbox: cube,
                    children: Vec::new(),
                    panel_ids: ids,
                });
                continue;
            }

            // Reserve the internal-node slot before pushing children,
            // so the children's `parent` index is stable.
            nodes.push(Node {
                bbox: cube,
                children: Vec::with_capacity(8),
                panel_ids: Vec::new(),
            });

            let mut buckets: [Vec<u32>; 8] = Default::default();
            for &pid in &ids {
                buckets[partition_octant(centroids[pid as usize], &cube)].push(pid);
            }
            let octants = cube.octants();
            // why: push non-empty buckets only; empty octants would
            // produce useless leaves that still contribute to the
            // MAC walk.
            for (o, bucket) in octants.into_iter().zip(buckets).rev() {
                if !bucket.is_empty() {
                    stack.push((bucket, o, Some(node_id), depth + 1));
                }
            }
        }

        Self { nodes, n_crit }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jittered_grid(n_per_axis: usize) -> Vec<DVec3> {
        let mut pts = Vec::with_capacity(n_per_axis.pow(3));
        for i in 0..n_per_axis {
            for j in 0..n_per_axis {
                for k in 0..n_per_axis {
                    let jitter = 1e-3 * (i + 3 * j + 7 * k) as f64;
                    pts.push(DVec3::new(
                        i as f64 + jitter,
                        j as f64 + 0.5 * jitter,
                        k as f64 + 0.25 * jitter,
                    ));
                }
            }
        }
        pts
    }

    #[test]
    fn every_point_lands_in_exactly_one_leaf() {
        let pts = jittered_grid(5);
        let tree = Tree::new(&pts, 8);

        let mut counts = vec![0_u32; pts.len()];
        for node in &tree.nodes {
            if node.is_leaf() {
                for &id in &node.panel_ids {
                    counts[id as usize] += 1;
                }
            }
        }
        assert!(
            counts.iter().all(|&c| c == 1),
            "expected every point in exactly one leaf; counts = {counts:?}"
        );
    }

    #[test]
    fn leaf_panel_counts_respect_n_crit() {
        let pts = jittered_grid(6);
        let n_crit = 12;
        let tree = Tree::new(&pts, n_crit);
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.is_leaf() {
                assert!(
                    node.panel_ids.len() <= n_crit.max(1),
                    "leaf {i} has {} panels > n_crit {n_crit}",
                    node.panel_ids.len()
                );
            }
        }
    }

    #[test]
    fn single_point_is_a_leaf() {
        let pts = vec![DVec3::new(1.0, 2.0, 3.0)];
        let tree = Tree::new(&pts, 1);
        assert_eq!(tree.nodes.len(), 1);
        assert!(tree.nodes[0].is_leaf());
        assert_eq!(tree.nodes[0].panel_ids, &[0]);
    }

    #[test]
    fn small_n_crit_actually_subdivides() {
        let pts = jittered_grid(4);
        let tree = Tree::new(&pts, 4);
        let n_internal = tree.nodes.iter().filter(|n| !n.is_leaf()).count();
        assert!(
            n_internal > 0,
            "expected some internal nodes at n_crit=4 with 64 points"
        );
    }

    #[test]
    fn bounding_radius_covers_all_leaf_points() {
        let pts = jittered_grid(5);
        let tree = Tree::new(&pts, 8);
        for node in &tree.nodes {
            if !node.is_leaf() {
                continue;
            }
            let r = node.bbox.bounding_radius();
            for &id in &node.panel_ids {
                let d = (pts[id as usize] - node.bbox.center).length();
                assert!(
                    d <= r + 1e-12,
                    "point {id} at dist {d} > bounding radius {r}"
                );
            }
        }
    }
}
