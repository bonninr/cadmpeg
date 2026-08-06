// SPDX-License-Identifier: Apache-2.0
//! Identity lookups and navigation over the solved topology.
//!
//! Both emitters need the same walks — a face's edges, an edge's solved
//! endpoints, the faces across a face's boundary — so they share one index
//! rather than each building its own.

use std::collections::{HashMap, HashSet};

use cadmpeg_ir::document::CadIr;
use cadmpeg_ir::geometry::{Curve, Surface};
use cadmpeg_ir::topology::{Coedge, Edge, Face, Loop, Point, Vertex};

use crate::geom::Vec3;

/// Borrowed lookups into one document's topology arenas.
pub(crate) struct Topology<'a> {
    pub surfaces: HashMap<&'a str, &'a Surface>,
    pub curves: HashMap<&'a str, &'a Curve>,
    loops: HashMap<&'a str, &'a Loop>,
    coedges: HashMap<&'a str, &'a Coedge>,
    edges: HashMap<&'a str, &'a Edge>,
    vertices: HashMap<&'a str, &'a Vertex>,
    points: HashMap<&'a str, &'a Point>,
    faces: HashMap<&'a str, &'a Face>,
    /// Every coedge that uses an edge, which is how adjacency is found.
    coedges_by_edge: HashMap<&'a str, Vec<&'a Coedge>>,
}

impl<'a> Topology<'a> {
    pub(crate) fn new(ir: &'a CadIr) -> Self {
        let mut coedges_by_edge: HashMap<&str, Vec<&Coedge>> = HashMap::new();
        for coedge in &ir.model.coedges {
            coedges_by_edge
                .entry(coedge.edge.as_str())
                .or_default()
                .push(coedge);
        }
        Self {
            surfaces: ir
                .model
                .surfaces
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect(),
            curves: ir
                .model
                .curves
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect(),
            loops: ir
                .model
                .loops
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect(),
            coedges: ir
                .model
                .coedges
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect(),
            edges: ir
                .model
                .edges
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect(),
            vertices: ir
                .model
                .vertices
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect(),
            points: ir
                .model
                .points
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect(),
            faces: ir
                .model
                .faces
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect(),
            coedges_by_edge,
        }
    }

    pub(crate) fn edge(&self, identity: &str) -> Option<&'a Edge> {
        self.edges.get(identity).copied()
    }

    pub(crate) fn loops_of<'b>(&'b self, face: &'b Face) -> impl Iterator<Item = &'a Loop> + 'b {
        face.loops
            .iter()
            .filter_map(|identity| self.loops.get(identity.as_str()).copied())
    }

    pub(crate) fn coedges_of<'b>(
        &'b self,
        owner: &'b Loop,
    ) -> impl Iterator<Item = &'a Coedge> + 'b {
        owner
            .coedges
            .iter()
            .filter_map(|identity| self.coedges.get(identity.as_str()).copied())
    }

    /// Solved endpoints of an edge, in start-then-end order.
    pub(crate) fn edge_points(&self, edge: &Edge) -> Vec<Vec3> {
        [&edge.start, &edge.end]
            .into_iter()
            .filter_map(|vertex| self.vertices.get(vertex.as_str()))
            .filter_map(|vertex| self.points.get(vertex.point.as_str()))
            .map(|point| Vec3::from(point.position))
            .collect()
    }

    /// Every edge bounding a face, without repetition.
    pub(crate) fn face_edges(&self, face: &Face) -> Vec<&'a Edge> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for owner in self.loops_of(face) {
            for coedge in self.coedges_of(owner) {
                if let Some(edge) = self.edges.get(coedge.edge.as_str()) {
                    if seen.insert(edge.id.as_str()) {
                        out.push(*edge);
                    }
                }
            }
        }
        out
    }

    /// Edges a face uses more than once, which is how a closed carrier records
    /// its seam.
    pub(crate) fn seam_edges(&self, face: &Face) -> HashSet<&'a str> {
        let mut uses: HashMap<&str, usize> = HashMap::new();
        for owner in self.loops_of(face) {
            for coedge in self.coedges_of(owner) {
                *uses.entry(coedge.edge.as_str()).or_default() += 1;
            }
        }
        uses.into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|(edge, _)| edge)
            .collect()
    }

    /// Faces sharing an edge with the given face.
    ///
    /// This is what makes a blend's support carriers reachable: the two faces a
    /// fillet was tangent to are exactly its neighbours.
    pub(crate) fn neighbours(&self, face_id: &str) -> Vec<String> {
        let Some(face) = self.faces.get(face_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for edge in self.face_edges(face) {
            let Some(uses) = self.coedges_by_edge.get(edge.id.as_str()) else {
                continue;
            };
            for coedge in uses {
                let Some(owner) = self.loops.get(coedge.owner_loop.as_str()) else {
                    continue;
                };
                let neighbour = owner.face.as_str();
                if neighbour != face_id && !out.iter().any(|found| found == neighbour) {
                    out.push(neighbour.to_owned());
                }
            }
        }
        out
    }

    /// Axial extent of a face along a direction, from its solved edge geometry.
    pub(crate) fn face_span(&self, face_id: &str, axis: Vec3) -> Option<(f64, f64)> {
        let face = self.faces.get(face_id)?;
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for edge in self.face_edges(face) {
            for point in self.edge_points(edge) {
                let station = point.dot(axis);
                low = low.min(station);
                high = high.max(station);
            }
        }
        if low.is_finite() && high.is_finite() {
            Some((low, high))
        } else {
            None
        }
    }

    /// Axial extent of every face built on one carrier, in absolute stations.
    ///
    /// Stations are projections onto the axis direction rather than offsets from
    /// the carrier origin, so they compare directly against a sketch plane's own
    /// station.
    pub(crate) fn surface_span(&self, surface_id: &str, axis: Vec3) -> Option<(f64, f64)> {
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for face in self.faces.values() {
            if face.surface.as_str() != surface_id {
                continue;
            }
            if let Some((face_low, face_high)) = self.face_span(face.id.as_str(), axis) {
                low = low.min(face_low);
                high = high.max(face_high);
            }
        }
        if low.is_finite() && high.is_finite() {
            Some((low, high))
        } else {
            None
        }
    }
}
