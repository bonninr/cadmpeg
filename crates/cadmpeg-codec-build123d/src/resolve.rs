// SPDX-License-Identifier: Apache-2.0
//! Selection resolvers for the parametric path.
//!
//! Two selections in the neutral feature model are routinely left unresolved,
//! because the source stores them as native persistent identities that survive
//! edits: an extrude's **profile**, and a fillet's **edge set**. Neither can be
//! read straight out of the IR for those documents.
//!
//! Both are recovered here from the solved B-rep, in closed form. No geometry is
//! evaluated: the rules only read carriers the decoder already solved, which is
//! what lets an exporter run them at all. Every recovered value is reported as
//! inferred, and a rule that cannot decide yields loss rather than a guess.

use std::collections::{BTreeMap, HashMap};

use cadmpeg_ir::document::CadIr;
use cadmpeg_ir::features::{
    BooleanOp, ExtrudeExtent, Feature, FeatureDefinition, ProfileRef, SketchProfileRegion,
    Termination,
};
use cadmpeg_ir::geometry::SurfaceGeometry;
use cadmpeg_ir::sketches::{Sketch, SketchEntity, SketchGeometry, SketchPlacement};

use crate::geom::{self, Vec3};

/// Tolerance for matching a solved radius against a sketch radius.
const RADIUS_TOLERANCE: f64 = 1e-3;
/// Tolerance for matching a solved span against an expected one.
const SPAN_TOLERANCE: f64 = 1e-3;

/// A plane frame plus the sketch-space mapping the profile geometry lives in.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SketchFrame {
    pub origin: Vec3,
    pub normal: Vec3,
    pub x_axis: Vec3,
    pub y_axis: Vec3,
}

impl SketchFrame {
    pub(crate) fn new(origin: Vec3, normal: Vec3, u_axis: Vec3) -> Self {
        let normal = normal.unit();
        let x_axis = u_axis.reject(normal).unit();
        Self {
            origin,
            normal,
            x_axis,
            y_axis: normal.cross(x_axis),
        }
    }

    /// Sketch (u, v) to model space.
    pub(crate) fn to_model(self, u: f64, v: f64) -> Vec3 {
        self.origin
            .add(self.x_axis.scale(u))
            .add(self.y_axis.scale(v))
    }

    /// Model space to sketch (u, v).
    pub(crate) fn to_sketch(self, point: Vec3) -> (f64, f64) {
        let delta = point.sub(self.origin);
        (delta.dot(self.x_axis), delta.dot(self.y_axis))
    }

    /// Axial station of the plane along its own normal.
    pub(crate) fn station(&self) -> f64 {
        self.origin.dot(self.normal)
    }
}

/// Reads a sketch's resolved placement, or `None` when the decoder left it open.
pub(crate) fn sketch_frame(sketch: &Sketch) -> Option<SketchFrame> {
    match &sketch.placement {
        SketchPlacement::Resolved {
            origin,
            normal,
            u_axis,
        } => Some(SketchFrame::new(
            Vec3::from(*origin),
            Vec3::from(*normal),
            Vec3::from(*u_axis),
        )),
        SketchPlacement::Unresolved => None,
    }
}

// ---------------------------------------------------------------------------
// planar arrangement
// ---------------------------------------------------------------------------

/// A closed profile loop reduced to what region nesting needs.
#[derive(Clone)]
pub(crate) struct Ring {
    /// Index into the sketch's profile table.
    pub index: usize,
    /// Enclosed area, used to order containment.
    pub area: f64,
    /// A point strictly inside the ring.
    pub interior: (f64, f64),
    /// Circle geometry when the ring is a lone circle, which is the case that
    /// can be matched against a solved cylinder.
    pub circle: Option<((f64, f64), f64)>,
    /// Sampled boundary, used for the point-in-ring test.
    polygon: Vec<(f64, f64)>,
}

impl Ring {
    /// Whether a sketch-space point lies inside the ring.
    pub(crate) fn contains(&self, point: (f64, f64)) -> bool {
        if let Some((center, radius)) = self.circle {
            let du = point.0 - center.0;
            let dv = point.1 - center.1;
            return du.mul_add(du, dv * dv) < radius * radius;
        }
        let mut inside = false;
        let count = self.polygon.len();
        for index in 0..count {
            let (x1, y1) = self.polygon[index];
            let (x2, y2) = self.polygon[(index + 1) % count];
            if (y1 > point.1) != (y2 > point.1) {
                let crossing = (point.1 - y1).mul_add((x2 - x1) / (y2 - y1), x1);
                if point.0 < crossing {
                    inside = !inside;
                }
            }
        }
        inside
    }
}

/// One atomic planar region: an exterior loop minus its immediate holes.
///
/// This is the same shape as `SketchProfileRegion::Loops`, so a resolved
/// selection and an inferred one describe a profile the same way.
#[derive(Debug, Clone)]
pub(crate) struct Region {
    pub outer: usize,
    pub holes: Vec<usize>,
}

/// The planar arrangement of one sketch's profile loops.
///
/// Only loop *nesting* is computed, which is what the whole-loop region form
/// encodes. Loops that cross one another need trimmed boundaries and are
/// refused rather than approximated.
pub(crate) struct Arrangement {
    pub rings: Vec<Ring>,
    pub regions: Vec<Region>,
    /// Which ring each sketch entity belongs to, for entity-level selections.
    pub ring_of_entity: HashMap<String, usize>,
}

impl Arrangement {
    pub(crate) fn build(sketch: &Sketch, entities: &HashMap<&str, &SketchEntity>) -> Option<Self> {
        let mut rings = Vec::new();
        let mut ring_of_entity = HashMap::new();
        for (index, loop_uses) in sketch.profiles.iter().enumerate() {
            rings.push(ring(index, loop_uses, entities)?);
            for entity_use in loop_uses {
                ring_of_entity.insert(entity_use.entity.0.clone(), index);
            }
        }
        if rings.is_empty() {
            return None;
        }
        let mut children: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for inner in &rings {
            let mut parent: Option<&Ring> = None;
            for outer in &rings {
                if outer.index == inner.index || outer.area <= inner.area {
                    continue;
                }
                if !outer.contains(inner.interior) {
                    continue;
                }
                if parent.is_none_or(|current| outer.area < current.area) {
                    parent = Some(outer);
                }
            }
            if let Some(parent) = parent {
                children.entry(parent.index).or_default().push(inner.index);
            }
        }
        let regions = rings
            .iter()
            .map(|ring| Region {
                outer: ring.index,
                holes: children.get(&ring.index).cloned().unwrap_or_default(),
            })
            .collect();
        Some(Self {
            rings,
            regions,
            ring_of_entity,
        })
    }

    /// Every atomic region contained in a loop, including the loop's own.
    pub(crate) fn regions_within(&self, outer: usize) -> Vec<Region> {
        let mut out = vec![self.regions[outer].clone()];
        let mut frontier = self.regions[outer].holes.clone();
        while let Some(index) = frontier.pop() {
            out.push(self.regions[index].clone());
            frontier.extend(self.regions[index].holes.iter().copied());
        }
        out
    }

    /// A point inside a region and outside all of its holes.
    pub(crate) fn sample(&self, region: &Region) -> (f64, f64) {
        let outer = &self.rings[region.outer];
        let candidate = outer.interior;
        if !region
            .holes
            .iter()
            .any(|hole| self.rings[*hole].contains(candidate))
        {
            return candidate;
        }
        // Walk the ring's own samples: one of them lies clear of every hole
        // whenever the region has any area at all.
        for point in &outer.polygon {
            let probe = (
                f64::midpoint(point.0, candidate.0),
                f64::midpoint(point.1, candidate.1),
            );
            if outer.contains(probe)
                && !region
                    .holes
                    .iter()
                    .any(|hole| self.rings[*hole].contains(probe))
            {
                return probe;
            }
        }
        candidate
    }
}

/// Reduces one profile loop to a ring, or `None` for curve families that cannot
/// be closed analytically here.
fn ring(
    index: usize,
    uses: &[cadmpeg_ir::sketches::SketchEntityUse],
    entities: &HashMap<&str, &SketchEntity>,
) -> Option<Ring> {
    let mut polygon = Vec::new();
    let mut circle = None;
    for entity_use in uses {
        let entity = entities.get(entity_use.entity.0.as_str())?;
        match &entity.geometry {
            SketchGeometry::Circle { center, radius } => {
                if uses.len() == 1 {
                    circle = Some(((center.u, center.v), radius.0));
                }
                polygon.extend(sample_arc(
                    (center.u, center.v),
                    radius.0,
                    0.0,
                    std::f64::consts::TAU,
                ));
            }
            SketchGeometry::Arc {
                center,
                radius,
                start_angle,
                end_angle,
            } => polygon.extend(sample_arc(
                (center.u, center.v),
                radius.0,
                start_angle.0,
                end_angle.0,
            )),
            SketchGeometry::Line { start, end } => {
                polygon.push((start.u, start.v));
                polygon.push((end.u, end.v));
            }
            _ => return None,
        }
    }
    if polygon.len() < 3 {
        return None;
    }
    let area = shoelace(&polygon).abs();
    let interior = interior_point(&polygon, circle);
    Some(Ring {
        index,
        area,
        interior,
        circle,
        polygon,
    })
}

fn sample_arc(center: (f64, f64), radius: f64, start: f64, end: f64) -> Vec<(f64, f64)> {
    const SAMPLES: usize = 24;
    (0..=SAMPLES)
        .map(|step| {
            let angle = (end - start).mul_add(step as f64 / SAMPLES as f64, start);
            (
                radius.mul_add(angle.cos(), center.0),
                radius.mul_add(angle.sin(), center.1),
            )
        })
        .collect()
}

fn shoelace(polygon: &[(f64, f64)]) -> f64 {
    let mut sum = 0.0;
    for index in 0..polygon.len() {
        let (x1, y1) = polygon[index];
        let (x2, y2) = polygon[(index + 1) % polygon.len()];
        sum += x1.mul_add(y2, -(x2 * y1));
    }
    sum / 2.0
}

fn interior_point(polygon: &[(f64, f64)], circle: Option<((f64, f64), f64)>) -> (f64, f64) {
    if let Some((center, _)) = circle {
        return center;
    }
    let count = polygon.len() as f64;
    (
        polygon.iter().map(|point| point.0).sum::<f64>() / count,
        polygon.iter().map(|point| point.1).sum::<f64>() / count,
    )
}

// ---------------------------------------------------------------------------
// solid-of-revolution classification
// ---------------------------------------------------------------------------

/// Exact inside/outside test for a solid whose faces share one axis.
///
/// Every lateral face of such a solid is a surface of revolution about the
/// common axis, so a point classifies by counting the lateral faces whose
/// meridian lies outward of it at the same axial station: an odd count is
/// inside. Faces perpendicular to the axis contribute nothing to a radial ray.
///
/// The classifier reports itself unavailable as soon as a face is not coaxial,
/// which keeps every answer it does give exact rather than approximate.
pub(crate) struct Meridian {
    axis: Vec3,
    origin: Vec3,
    /// (carrier geometry, axial span) for each lateral face.
    lateral: Vec<(SurfaceGeometry, f64, f64)>,
}

impl Meridian {
    pub(crate) fn build(
        ir: &CadIr,
        face_span: &dyn Fn(&str) -> Option<(f64, f64)>,
    ) -> Option<Self> {
        let surfaces: HashMap<&str, &SurfaceGeometry> = ir
            .model
            .surfaces
            .iter()
            .map(|surface| (surface.id.as_str(), &surface.geometry))
            .collect();
        let mut axis: Option<Vec3> = None;
        for face in &ir.model.faces {
            let geometry = surfaces.get(face.surface.as_str())?;
            let (_, face_axis, _) = geom::surface_frame(geometry)?;
            match axis {
                None => axis = Some(face_axis),
                Some(common) if common.is_parallel_to(face_axis) => {}
                Some(_) => return None,
            }
        }
        let axis = axis?;
        let mut lateral = Vec::new();
        for face in &ir.model.faces {
            let geometry = surfaces.get(face.surface.as_str())?;
            if matches!(geometry, SurfaceGeometry::Plane { .. }) {
                continue;
            }
            let (low, high) = face_span(face.id.as_str())?;
            lateral.push(((*geometry).clone(), low, high));
        }
        if lateral.is_empty() {
            return None;
        }
        Some(Self {
            axis,
            origin: Vec3::new(0.0, 0.0, 0.0),
            lateral,
        })
    }

    /// True when the model point lies inside the solid.
    pub(crate) fn contains(&self, point: Vec3) -> bool {
        let delta = point.sub(self.origin);
        let station = delta.dot(self.axis);
        let radius = delta.reject(self.axis).length();
        let mut crossings = 0usize;
        for (geometry, low, high) in &self.lateral {
            if station < low - geom::DIRECTION_TOLERANCE
                || station > high + geom::DIRECTION_TOLERANCE
            {
                continue;
            }
            crossings += Self::outward_crossings(geometry, station, radius);
        }
        crossings % 2 == 1
    }

    /// Radii greater than `radius` where the carrier's meridian sits at `station`.
    fn outward_crossings(geometry: &SurfaceGeometry, station: f64, radius: f64) -> usize {
        let Some((origin, axis, _)) = geom::surface_frame(geometry) else {
            return 0;
        };
        let local = station - origin.dot(axis);
        let candidates: Vec<f64> = match geometry {
            SurfaceGeometry::Cylinder { radius, .. } => vec![*radius],
            SurfaceGeometry::Cone {
                radius, half_angle, ..
            } => vec![local.mul_add(half_angle.tan(), *radius)],
            SurfaceGeometry::Sphere { radius, .. } => {
                let squared = radius.mul_add(*radius, -(local * local));
                if squared > 0.0 {
                    vec![squared.sqrt()]
                } else {
                    Vec::new()
                }
            }
            SurfaceGeometry::Torus {
                major_radius,
                minor_radius,
                ..
            } => {
                let minor = minor_radius.abs();
                let squared = minor.mul_add(minor, -(local * local));
                if squared > 0.0 {
                    let offset = squared.sqrt();
                    vec![major_radius - offset, major_radius + offset]
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        };
        candidates
            .into_iter()
            .filter(|candidate| *candidate > radius + geom::DIRECTION_TOLERANCE)
            .count()
    }
}

// ---------------------------------------------------------------------------
// extrude spans and profile resolution
// ---------------------------------------------------------------------------

/// Travel of an extrude either side of its profile plane.
///
/// Covers the three extent shapes. A two-sided extent names its second side as
/// the one opposite the direction; a symmetric extent states total travel split
/// evenly. `None` means a side is not blind, which needs target geometry this
/// encoder does not resolve.
pub(crate) fn extrude_travel(extent: &ExtrudeExtent) -> Option<(f64, f64)> {
    fn blind(termination: &Termination) -> Option<f64> {
        match termination {
            Termination::Blind { length } => Some(length.0),
            _ => None,
        }
    }
    match extent {
        ExtrudeExtent::OneSided { side } => Some((0.0, blind(&side.termination)?)),
        ExtrudeExtent::Symmetric { side } => {
            let total = blind(&side.termination)?;
            Some((total / 2.0, total / 2.0))
        }
        ExtrudeExtent::TwoSided { first, second } => {
            Some((blind(&second.termination)?, blind(&first.termination)?))
        }
    }
}

/// The axial span an extrude sweeps, in stations along the sketch normal.
pub(crate) fn extrude_span(frame: &SketchFrame, extent: &ExtrudeExtent) -> Option<(f64, f64)> {
    let (back, forward) = extrude_travel(extent)?;
    let start = frame.station();
    let low = start - back;
    let high = start + forward;
    Some(if low <= high {
        (low, high)
    } else {
        (high, low)
    })
}

/// A profile resolved to concrete regions, with how it was arrived at.
pub(crate) struct ResolvedProfile {
    pub sketch: String,
    pub regions: Vec<Region>,
    /// `None` when the profile was read from the IR rather than inferred.
    pub inference: Option<String>,
}

/// Normalizes any `ProfileRef` into concrete sketch regions.
///
/// The exact variants are read straight through. `SketchSelection` is the one
/// that needs recovering, and it is handled by [`infer_profile`].
pub(crate) fn resolve_profile(
    profile: &ProfileRef,
    arrangements: &HashMap<String, Arrangement>,
) -> Result<ResolvedProfile, String> {
    let (sketch, regions) = match profile {
        ProfileRef::SketchRegions { sketch, regions } => {
            let mut out = Vec::new();
            for region in regions {
                match region {
                    SketchProfileRegion::Loops { outer, holes } => out.push(Region {
                        outer: *outer as usize,
                        holes: holes.iter().map(|hole| *hole as usize).collect(),
                    }),
                    SketchProfileRegion::Trimmed { .. } => {
                        return Err(
                            "the trimmed-boundary region form needs curve splitting this \
                             encoder does not perform"
                                .to_owned(),
                        )
                    }
                }
            }
            (sketch.0.as_str(), out)
        }
        ProfileRef::SketchProfiles { sketch, profiles } => {
            let arrangement = arrangements
                .get(sketch.0.as_str())
                .ok_or_else(|| format!("sketch {} has no usable planar arrangement", sketch.0))?;
            let regions = profiles
                .iter()
                .filter_map(|index| arrangement.regions.get(*index as usize).cloned())
                .collect();
            (sketch.0.as_str(), regions)
        }
        ProfileRef::Sketch(sketch) => {
            let arrangement = arrangements
                .get(sketch.0.as_str())
                .ok_or_else(|| format!("sketch {} has no usable planar arrangement", sketch.0))?;
            (sketch.0.as_str(), arrangement.regions.clone())
        }
        ProfileRef::SketchEntities { sketch, entities } => {
            let arrangement = arrangements
                .get(sketch.0.as_str())
                .ok_or_else(|| format!("sketch {} has no usable planar arrangement", sketch.0))?;
            let selected: Vec<Region> = arrangement
                .regions
                .iter()
                .filter(|region| {
                    entities.iter().any(|entity| {
                        arrangement.ring_of_entity.get(&entity.0) == Some(&region.outer)
                    })
                })
                .cloned()
                .collect();
            if selected.is_empty() {
                return Err("the selected sketch entities close no profile loop".to_owned());
            }
            (sketch.0.as_str(), selected)
        }
        ProfileRef::SketchSelection { sketch, .. } => {
            return Err(format!(
                "profile is a native selection in sketch {}; it needs inference",
                sketch.0
            ))
        }
        other => return Err(format!("unsupported profile reference: {other:?}")),
    };
    Ok(ResolvedProfile {
        sketch: sketch.to_owned(),
        regions,
        inference: None,
    })
}

/// One candidate attribution of a solved carrier to a profile loop.
struct Candidate {
    cost: f64,
    ring: usize,
    surface: String,
}

/// Attributes one solved lateral carrier to each extrude, globally.
///
/// A solved face is created by exactly one feature, so this is an assignment
/// problem rather than a per-feature nearest match. Choosing independently lets
/// a shallow cut claim the carrier a deeper extrude needs, which then leaves the
/// deeper one with nothing.
///
/// Returns the chosen ring index per feature id.
pub(crate) fn assign_outer_rings(
    features: &[(&Feature, &SketchFrame, &Arrangement, (f64, f64))],
    ir: &CadIr,
    axial_span: &dyn Fn(&str) -> Option<(f64, f64)>,
) -> HashMap<String, usize> {
    let per_feature: Vec<Vec<Candidate>> = features
        .iter()
        .map(|(_, frame, arrangement, span)| {
            let mut candidates = Vec::new();
            for ring in &arrangement.rings {
                candidates.extend(ring_candidates(ring, frame, *span, ir, axial_span));
            }
            candidates.sort_by(|left, right| left.cost.total_cmp(&right.cost));
            candidates
        })
        .collect();

    let mut best: Option<(f64, Vec<Option<usize>>)> = None;
    let mut used = Vec::new();
    let mut current = Vec::new();
    search(&per_feature, 0, &mut used, &mut current, 0.0, &mut best);

    let mut out = HashMap::new();
    if let Some((_, assignment)) = best {
        for ((feature, _, _, _), ring) in features.iter().zip(assignment) {
            if let Some(ring) = ring {
                out.insert(feature.id.as_str().to_owned(), ring);
            }
        }
    }
    out
}

fn search(
    candidates: &[Vec<Candidate>],
    depth: usize,
    used: &mut Vec<String>,
    current: &mut Vec<Option<usize>>,
    cost: f64,
    best: &mut Option<(f64, Vec<Option<usize>>)>,
) {
    if best
        .as_ref()
        .is_some_and(|(best_cost, _)| cost >= *best_cost)
    {
        return;
    }
    if depth == candidates.len() {
        *best = Some((cost, current.clone()));
        return;
    }
    let mut placed = false;
    for candidate in &candidates[depth] {
        if used.contains(&candidate.surface) {
            continue;
        }
        placed = true;
        used.push(candidate.surface.clone());
        current.push(Some(candidate.ring));
        search(
            candidates,
            depth + 1,
            used,
            current,
            cost + candidate.cost,
            best,
        );
        current.pop();
        used.pop();
    }
    if !placed {
        // Leaving a feature unattributed is allowed, but costs more than any
        // real assignment so it is only chosen when nothing else fits.
        current.push(None);
        search(candidates, depth + 1, used, current, cost + 1e6, best);
        current.pop();
    }
}

/// Solved carriers that could be the swept boundary of one ring.
fn ring_candidates(
    ring: &Ring,
    frame: &SketchFrame,
    span: (f64, f64),
    ir: &CadIr,
    axial_span: &dyn Fn(&str) -> Option<(f64, f64)>,
) -> Vec<Candidate> {
    let Some((center, radius)) = ring.circle else {
        return Vec::new();
    };
    let center = frame.to_model(center.0, center.1);
    let mut out = Vec::new();
    for surface in &ir.model.surfaces {
        let SurfaceGeometry::Cylinder {
            radius: solved_radius,
            ..
        } = &surface.geometry
        else {
            continue;
        };
        if (solved_radius - radius).abs() > RADIUS_TOLERANCE {
            continue;
        }
        let Some((origin, axis, _)) = geom::surface_frame(&surface.geometry) else {
            continue;
        };
        if !axis.is_parallel_to(frame.normal) {
            continue;
        }
        // The carrier axis must pass through the circle's own centre.
        if origin.sub(center).reject(axis).length() > RADIUS_TOLERANCE {
            continue;
        }
        let Some((low, high)) = axial_span(surface.id.as_str()) else {
            continue;
        };
        // A later blend can trim the solved face, so containment is the test
        // rather than equality.
        if low < span.0 - SPAN_TOLERANCE || high > span.1 + SPAN_TOLERANCE {
            continue;
        }
        out.push(Candidate {
            cost: (low - span.0).abs() + (high - span.1).abs(),
            ring: ring.index,
            surface: surface.id.as_str().to_owned(),
        });
    }
    out
}

/// Recovers a native profile selection from the solved body.
///
/// Two steps. The exterior loop comes from [`assign_outer_rings`]. Its holes are
/// then found by starting from the filled interior and dropping every atomic
/// region whose material is absent from the final solid and is not explained by
/// a later subtractive feature — a region a later cut removes is kept, because
/// the cut re-creates the void.
///
/// The result reproduces the final solid. It is not a claim about which profile
/// the user originally clicked, and it is reported as inferred for that reason.
pub(crate) fn infer_profile(
    sketch: &str,
    outer: usize,
    arrangement: &Arrangement,
    frame: &SketchFrame,
    span: (f64, f64),
    meridian: Option<&Meridian>,
    later_cut_covers: &dyn Fn(Vec3) -> bool,
) -> ResolvedProfile {
    let candidates = arrangement.regions_within(outer);
    let total = candidates.len();
    let mut kept = Vec::new();
    for region in candidates {
        if let Some(meridian) = meridian {
            let (u, v) = arrangement.sample(&region);
            let point = frame.to_model(u, v);
            let middle = point.add(
                frame
                    .normal
                    .scale(f64::midpoint(span.0, span.1) - point.dot(frame.normal)),
            );
            if !meridian.contains(middle) && !later_cut_covers(middle) {
                continue;
            }
        }
        kept.push(region);
    }
    if kept.is_empty() {
        kept.push(arrangement.regions[outer].clone());
    }
    let inference = format!(
        "exterior loop {outer} attributed to a solved carrier; {} of {total} atomic \
         regions retained",
        kept.len()
    );
    ResolvedProfile {
        sketch: sketch.to_owned(),
        regions: kept,
        inference: Some(inference),
    }
}

/// Whether a boolean operation removes material.
pub(crate) const fn is_subtractive(op: BooleanOp) -> bool {
    matches!(op, BooleanOp::Cut)
}

/// The extrude definition of a feature, when it has one.
pub(crate) fn as_extrude(feature: &Feature) -> Option<(&ProfileRef, &ExtrudeExtent, &BooleanOp)> {
    match &feature.definition {
        FeatureDefinition::Extrude {
            profile,
            extent,
            op,
            ..
        } => Some((profile, extent, op)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// blend edges
// ---------------------------------------------------------------------------

/// A circle in model space: the pre-blend edge a fillet consumed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EdgeCircle {
    pub center: Vec3,
    pub radius: f64,
}

/// Recovers the sharp edges a blend replaced.
///
/// A blend face is bounded by exactly the two faces it was tangent to, so the
/// edge it replaced is the intersection of those two support carriers. That is
/// closed-form for every analytic pair a blend can join, and it needs no
/// historical identity — which is what makes it usable here, since the identity
/// is exactly what the decoder could not resolve.
pub(crate) fn blend_edges(
    ir: &CadIr,
    radius: f64,
    neighbours: &dyn Fn(&str) -> Vec<String>,
) -> Vec<EdgeCircle> {
    let surfaces: HashMap<&str, &SurfaceGeometry> = ir
        .model
        .surfaces
        .iter()
        .map(|surface| (surface.id.as_str(), &surface.geometry))
        .collect();
    // A neighbour is named by its face, so reaching its carrier needs the
    // face-to-surface hop as well.
    let carrier_of_face: HashMap<&str, &str> = ir
        .model
        .faces
        .iter()
        .map(|face| (face.id.as_str(), face.surface.as_str()))
        .collect();
    let mut out = Vec::new();
    for face in &ir.model.faces {
        let Some(geometry) = surfaces.get(face.surface.as_str()) else {
            continue;
        };
        let SurfaceGeometry::Torus {
            major_radius,
            minor_radius,
            ..
        } = geometry
        else {
            continue;
        };
        if (minor_radius.abs() - radius).abs() > 1e-6 {
            continue;
        }
        let supports: Vec<&SurfaceGeometry> = neighbours(face.id.as_str())
            .into_iter()
            .filter_map(|neighbour| carrier_of_face.get(neighbour.as_str()).copied())
            .filter_map(|carrier| surfaces.get(carrier).copied())
            .collect();
        let mut circles = Vec::new();
        for first in 0..supports.len() {
            for second in (first + 1)..supports.len() {
                circles.extend(intersect(supports[first], supports[second]));
            }
        }
        if let Some(edge) = pick_sharp_edge(&circles, geometry, *major_radius, minor_radius.abs()) {
            out.push(edge);
        }
    }
    out
}

/// Circles shared by two analytic carriers, for the pairs a blend can join.
fn intersect(first: &SurfaceGeometry, second: &SurfaceGeometry) -> Vec<EdgeCircle> {
    let (Some((first_origin, first_axis, _)), Some((second_origin, second_axis, _))) =
        (geom::surface_frame(first), geom::surface_frame(second))
    else {
        return Vec::new();
    };
    if !first_axis.is_parallel_to(second_axis) {
        return Vec::new();
    }
    // A cylinder or cone meeting a perpendicular plane gives one circle; this
    // is the pair every ordinary fillet sits on.
    let pairs = [
        (first, first_origin, first_axis, second, second_origin),
        (second, second_origin, second_axis, first, first_origin),
    ];
    for (revolved, origin, axis, plane, plane_origin) in pairs {
        if !matches!(plane, SurfaceGeometry::Plane { .. }) {
            continue;
        }
        let station = plane_origin.sub(origin).dot(axis);
        let radius = match revolved {
            SurfaceGeometry::Cylinder { radius, .. } => *radius,
            SurfaceGeometry::Cone {
                radius, half_angle, ..
            } => station.mul_add(half_angle.tan(), *radius),
            _ => continue,
        };
        return vec![EdgeCircle {
            center: origin.add(axis.scale(station)),
            radius,
        }];
    }
    Vec::new()
}

/// The intersection circle a blend of this size actually replaced.
///
/// The sharp edge lies within one tube diameter of the tube's centre circle;
/// anything further away belongs to a different corner of the body.
fn pick_sharp_edge(
    circles: &[EdgeCircle],
    geometry: &SurfaceGeometry,
    major_radius: f64,
    minor_radius: f64,
) -> Option<EdgeCircle> {
    let (origin, axis, _) = geom::surface_frame(geometry)?;
    let mut best: Option<(f64, EdgeCircle)> = None;
    for circle in circles {
        let station = circle.center.sub(origin).dot(axis);
        let distance = station.hypot(circle.radius - major_radius);
        if distance > 2.0f64.mul_add(minor_radius, 1e-6) {
            continue;
        }
        if best.as_ref().is_none_or(|(current, _)| distance < *current) {
            best = Some((distance, *circle));
        }
    }
    best.map(|(_, circle)| circle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cadmpeg_ir::math::{Point3, Vector3};

    fn plane(height: f64) -> SurfaceGeometry {
        SurfaceGeometry::Plane {
            origin: Point3 {
                x: 0.0,
                y: height,
                z: 0.0,
            },
            normal: Vector3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            u_axis: Vector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        }
    }

    fn cylinder(radius: f64) -> SurfaceGeometry {
        SurfaceGeometry::Cylinder {
            origin: Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            axis: Vector3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            ref_direction: Vector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            radius,
        }
    }

    #[test]
    fn a_wall_meeting_a_face_gives_the_corner_circle() {
        let circles = intersect(&cylinder(3.75), &plane(1.0));
        assert_eq!(circles.len(), 1);
        assert!((circles[0].radius - 3.75).abs() < 1e-12);
        assert!((circles[0].center.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_skew_plane_shares_no_circle() {
        let skew = SurfaceGeometry::Plane {
            origin: Point3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            normal: Vector3 {
                x: 0.0,
                y: 1.0,
                z: 1.0,
            },
            u_axis: Vector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        };
        assert!(intersect(&cylinder(3.75), &skew).is_empty());
    }

    #[test]
    fn a_blind_one_sided_extent_travels_forward_only() {
        let extent = ExtrudeExtent::OneSided {
            side: cadmpeg_ir::features::ExtrudeSide {
                termination: Termination::Blind {
                    length: cadmpeg_ir::features::Length(4.0),
                },
                draft: None,
                offset: None,
            },
        };
        assert_eq!(extrude_travel(&extent), Some((0.0, 4.0)));
    }

    #[test]
    fn a_symmetric_extent_splits_its_travel() {
        let extent = ExtrudeExtent::Symmetric {
            side: cadmpeg_ir::features::ExtrudeSide {
                termination: Termination::Blind {
                    length: cadmpeg_ir::features::Length(10.0),
                },
                draft: None,
                offset: None,
            },
        };
        assert_eq!(extrude_travel(&extent), Some((5.0, 5.0)));
    }

    #[test]
    fn a_non_blind_extent_is_refused() {
        let extent = ExtrudeExtent::OneSided {
            side: cadmpeg_ir::features::ExtrudeSide {
                termination: Termination::ThroughAll,
                draft: None,
                offset: None,
            },
        };
        assert_eq!(extrude_travel(&extent), None);
    }
}
