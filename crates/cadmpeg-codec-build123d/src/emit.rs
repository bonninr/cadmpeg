// SPDX-License-Identifier: Apache-2.0
//! Writes the feature timeline as a build123d program.
//!
//! Where the B-rep path reproduces geometry, this path reproduces *intent*: a
//! sketch with a circle in it, an extrude with a length, a fillet with a
//! radius. That is what makes the output worth editing, and it is only possible
//! for documents whose decoder recovered a feature history.
//!
//! Feature families outside the supported set are skipped with a loss note and
//! the emitted program still runs. A partial model is more useful than none,
//! provided the report says plainly what is missing.

use std::collections::{BTreeMap, HashMap};

use cadmpeg_ir::document::CadIr;
use cadmpeg_ir::features::{
    BooleanOp, ExtrudeExtent, Feature, FeatureDefinition, ProfileRef, RadiusSpec,
};
use cadmpeg_ir::report::{LossKind, LossNote};
use cadmpeg_ir::sketches::SketchEntity;

use crate::geom::{self, Vec3};
use crate::resolve::{self, Arrangement, Meridian, Region, ResolvedProfile, Ring, SketchFrame};
use crate::topo::Topology;

/// Writes the parametric program, or `None` when the document carries no
/// feature this path can express.
pub(crate) fn write(ir: &CadIr) -> Option<(String, Vec<LossNote>, BTreeMap<String, usize>)> {
    let mut writer = Writer::new(ir);
    writer.plan();
    if !writer.planned.iter().any(|plan| plan.profile.is_some()) {
        return None;
    }
    Some(writer.finish())
}

/// One extrude with everything resolution needs already looked up.
struct Planned<'a> {
    feature: &'a Feature,
    sketch: String,
    frame: SketchFrame,
    span: (f64, f64),
    op: BooleanOp,
    extent: &'a ExtrudeExtent,
    profile: Option<ResolvedProfile>,
}

struct Writer<'a> {
    ir: &'a CadIr,
    topo: Topology<'a>,
    arrangements: HashMap<String, Arrangement>,
    frames: HashMap<String, SketchFrame>,
    planned: Vec<Planned<'a>>,
    /// Exterior loop attributed to each extrude that needed inference.
    rings: HashMap<String, usize>,
    /// Extrudes awaiting inference, with the reason their selection was open.
    pending: Vec<(usize, String)>,
    meridian: Option<Meridian>,
    lines: Vec<String>,
    losses: Vec<LossNote>,
    counts: BTreeMap<String, usize>,
    sketch_seq: usize,
    blend_seq: usize,
    bodies_started: usize,
}

impl<'a> Writer<'a> {
    fn new(ir: &'a CadIr) -> Self {
        Self {
            ir,
            topo: Topology::new(ir),
            arrangements: HashMap::new(),
            frames: HashMap::new(),
            planned: Vec::new(),
            rings: HashMap::new(),
            pending: Vec::new(),
            meridian: None,
            lines: Vec::new(),
            losses: Vec::new(),
            counts: BTreeMap::new(),
            sketch_seq: 0,
            blend_seq: 0,
            bodies_started: 0,
        }
    }

    // -- resolution ---------------------------------------------------------

    /// Resolves every selection before a line of output is written.
    ///
    /// Order matters: exterior loops are attributed globally, so one extrude's
    /// choice constrains the rest, and that has to settle before any single
    /// profile is decided.
    fn plan(&mut self) {
        let entity_refs: HashMap<&str, &SketchEntity> = self
            .ir
            .model
            .sketch_entities
            .iter()
            .map(|entity| (entity.id.0.as_str(), entity))
            .collect();
        for sketch in &self.ir.model.sketches {
            let (Some(frame), Some(arrangement)) = (
                resolve::sketch_frame(sketch),
                Arrangement::build(sketch, &entity_refs),
            ) else {
                continue;
            };
            self.frames.insert(sketch.id.0.clone(), frame);
            self.arrangements.insert(sketch.id.0.clone(), arrangement);
        }

        let mut features: Vec<&Feature> = self.ir.model.features.iter().collect();
        features.sort_by(|left, right| {
            left.ordinal
                .cmp(&right.ordinal)
                .then_with(|| left.id.as_str().cmp(right.id.as_str()))
        });

        for feature in features {
            if feature.suppressed == Some(true) {
                continue;
            }
            let Some((profile, extent, op)) = resolve::as_extrude(feature) else {
                continue;
            };
            let Some(sketch) = profile_sketch(profile) else {
                continue;
            };
            let Some(frame) = self.frames.get(sketch).copied() else {
                continue;
            };
            let Some(span) = resolve::extrude_span(&frame, extent) else {
                self.losses.push(LossNote::new(
                    LossKind::ParametricRecordOmitted,
                    format!(
                        "{}: its extent is not blind on every side; through-all, to-face \
                         and to-vertex laws need target geometry this encoder does not \
                         resolve",
                        feature_name(feature)
                    ),
                ));
                continue;
            };
            self.planned.push(Planned {
                feature,
                sketch: sketch.to_owned(),
                frame,
                span,
                op: *op,
                extent,
                profile: None,
            });
        }

        self.attribute_exterior_loops();
        self.build_classifier();
        self.infer_open_profiles();
    }

    fn attribute_exterior_loops(&mut self) {
        for index in 0..self.planned.len() {
            let Some((profile_ref, _, _)) = resolve::as_extrude(self.planned[index].feature) else {
                continue;
            };
            match resolve::resolve_profile(profile_ref, &self.arrangements) {
                Ok(resolved) => self.planned[index].profile = Some(resolved),
                Err(reason) => self.pending.push((index, reason)),
            }
        }
        if self.pending.is_empty() {
            return;
        }
        let axis = self.reference_axis();
        let inputs: Vec<(&Feature, &SketchFrame, &Arrangement, (f64, f64))> = self
            .pending
            .iter()
            .filter_map(|(index, _)| {
                let plan = &self.planned[*index];
                self.arrangements
                    .get(&plan.sketch)
                    .map(|arrangement| (plan.feature, &plan.frame, arrangement, plan.span))
            })
            .collect();
        self.rings = resolve::assign_outer_rings(&inputs, self.ir, &|surface| {
            self.topo.surface_span(surface, axis)
        });
    }

    fn build_classifier(&mut self) {
        let axis = self.reference_axis();
        self.meridian = Meridian::build(self.ir, &|face| self.topo.face_span(face, axis));
        if self.meridian.is_none() && !self.pending.is_empty() {
            self.losses.push(LossNote::new(
                LossKind::CarrierSummary,
                "the body is not a solid of revolution, so inferred profiles keep every \
                 atomic region of their exterior loop rather than testing each for \
                 material"
                    .to_owned(),
            ));
        }
    }

    fn infer_open_profiles(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        for (index, reason) in pending {
            let feature_id = self.planned[index].feature.id.as_str().to_owned();
            let Some(outer) = self.rings.get(&feature_id).copied() else {
                self.losses.push(LossNote::new(
                    LossKind::ParametricRecordOmitted,
                    format!(
                        "{}: {reason}, and no solved carrier could be attributed to any of \
                         its profile loops",
                        feature_name(self.planned[index].feature)
                    ),
                ));
                continue;
            };
            let sketch = self.planned[index].sketch.clone();
            let span = self.planned[index].span;
            let frame = self.planned[index].frame;
            let covered = self.later_cut_coverage(index);
            let Some(arrangement) = self.arrangements.get(&sketch) else {
                continue;
            };
            self.planned[index].profile = Some(resolve::infer_profile(
                &sketch,
                outer,
                arrangement,
                &frame,
                span,
                self.meridian.as_ref(),
                &covered,
            ));
        }
    }

    /// Whether a later subtractive extrude removes a probe point.
    ///
    /// Only the exterior loop and span of the later cut are consulted, never its
    /// holes: those may themselves still be under inference, and the exterior is
    /// enough to answer the question.
    fn later_cut_coverage(&self, index: usize) -> impl Fn(Vec3) -> bool + 'static {
        let ordinal = self.planned[index].feature.ordinal;
        // Owned so the closure outlives the borrow of the planner.
        let cuts: Vec<(SketchFrame, (f64, f64), Ring)> = self
            .planned
            .iter()
            .filter(|plan| plan.feature.ordinal > ordinal && resolve::is_subtractive(plan.op))
            .filter_map(|plan| {
                let outer = self.rings.get(plan.feature.id.as_str())?;
                let arrangement = self.arrangements.get(&plan.sketch)?;
                Some((
                    plan.frame,
                    plan.span,
                    arrangement.rings.get(*outer)?.clone(),
                ))
            })
            .collect();
        move |probe: Vec3| {
            cuts.iter().any(|(frame, span, ring)| {
                let station = probe.dot(frame.normal);
                if station < span.0 - geom::DIRECTION_TOLERANCE
                    || station > span.1 + geom::DIRECTION_TOLERANCE
                {
                    return false;
                }
                ring.contains(frame.to_sketch(probe))
            })
        }
    }

    /// The axis stations are measured along. Every sketch in a resolvable
    /// history shares it; the first plane's normal names it.
    fn reference_axis(&self) -> Vec3 {
        self.planned
            .first()
            .map_or(Vec3::new(0.0, 0.0, 1.0), |plan| plan.frame.normal)
    }

    // -- emission -----------------------------------------------------------

    fn finish(mut self) -> (String, Vec<LossNote>, BTreeMap<String, usize>) {
        self.header();
        self.push("with BuildPart() as part:");
        let mut emitted = 0usize;

        let mut features: Vec<&Feature> = self.ir.model.features.iter().collect();
        features.sort_by(|left, right| {
            left.ordinal
                .cmp(&right.ordinal)
                .then_with(|| left.id.as_str().cmp(right.id.as_str()))
        });
        for feature in features {
            if feature.suppressed == Some(true) {
                self.losses.push(LossNote::new(
                    LossKind::ParametricRecordOmitted,
                    format!("{}: suppressed in the source", feature_name(feature)),
                ));
                continue;
            }
            match &feature.definition {
                FeatureDefinition::Sketch { .. } => {
                    // Sketches are written where a profile consumes them, so the
                    // program keeps one construct per solid operation.
                }
                FeatureDefinition::Extrude { .. } => {
                    if self.extrude(feature) {
                        emitted += 1;
                    }
                }
                FeatureDefinition::Fillet { groups } => {
                    let groups = groups.clone();
                    if self.fillet(feature, &groups) {
                        emitted += 1;
                    }
                }
                other => {
                    self.losses.push(LossNote::new(
                        LossKind::ParametricRecordOmitted,
                        format!(
                            "{}: the {} feature family is not expressed by this encoder",
                            feature_name(feature),
                            family_name(other)
                        ),
                    ));
                }
            }
        }
        if emitted == 0 {
            self.push("    pass");
        }
        self.footer();
        (self.lines.join("\n"), self.losses, self.counts)
    }

    fn push(&mut self, line: &str) {
        self.lines.push(line.to_owned());
    }

    fn body(&mut self, line: &str) {
        if line.is_empty() {
            self.lines.push(String::new());
        } else {
            self.lines.push(format!("    {line}"));
        }
    }

    fn count(&mut self, key: &str) {
        *self.counts.entry(key.to_owned()).or_default() += 1;
    }

    fn header(&mut self) {
        let format = self
            .ir
            .source
            .as_ref()
            .map_or_else(|| "unknown".to_owned(), |source| source.format.clone());
        self.push("\"\"\"build123d program generated by cadmpeg.");
        self.push("");
        self.push(&format!("Source format : {format}"));
        self.push(&format!("IR version    : {}", self.ir.ir_version));
        self.push("");
        self.push("This is the feature history, not the solved shape. Values marked");
        self.push("INFERRED were recovered from the solved body because the source stores");
        self.push("that selection as a native identity; the export report gives the rule");
        self.push("used in each case.");
        self.push("\"\"\"");
        self.push("");
        self.push("from build123d import *");
        self.push("");
    }

    fn footer(&mut self) {
        self.push("");
        self.push("result = part.part");
        self.push("if result is None:");
        self.push("    print(\"volume: 0.000000 (empty)\")");
        self.push("else:");
        self.push("    print(\"volume: %.6f\" % result.volume)");
    }

    fn extrude(&mut self, feature: &Feature) -> bool {
        let Some(index) = self
            .planned
            .iter()
            .position(|plan| plan.feature.id == feature.id)
        else {
            return false;
        };
        let Some(profile) = self.planned[index].profile.as_ref() else {
            self.losses.push(LossNote::new(
                LossKind::ParametricRecordOmitted,
                format!("{}: its profile is unresolved", feature_name(feature)),
            ));
            return false;
        };
        let sketch = profile.sketch.clone();
        let regions = profile.regions.clone();
        let inference = profile.inference.clone();
        let op = self.planned[index].op;
        let extent = self.planned[index].extent;
        let Some((back, forward)) = resolve::extrude_travel(extent) else {
            return false;
        };
        let Some(mode) = boolean_mode(op) else {
            self.losses.push(LossNote::new(
                LossKind::ParametricRecordOmitted,
                format!(
                    "{}: its boolean operation is unresolved",
                    feature_name(feature)
                ),
            ));
            return false;
        };
        if matches!(op, BooleanOp::NewBody) {
            self.bodies_started += 1;
            if self.bodies_started > 1 {
                self.losses.push(LossNote::new(
                    LossKind::ParametricRecordOmitted,
                    format!(
                        "{}: starts a further body, and the emitted program merges every \
                         body into one part",
                        feature_name(feature)
                    ),
                ));
            }
        }

        let note = inference.map_or_else(String::new, |rule| format!("  INFERRED ({rule})"));
        self.body("");
        self.body(&format!(
            "# {} - {}, {}{note}",
            feature_name(feature),
            operation_name(op),
            travel_label(back, forward)
        ));
        self.emit_sketch(&sketch, &regions);

        let symmetric = back > 0.0 && (back - forward).abs() < 1e-9;
        let mut arguments = vec![format!("amount={}", geom::number(forward))];
        if symmetric {
            arguments.push("both=True".to_owned());
        }
        if mode != "Mode.ADD" {
            arguments.push(format!("mode={mode}"));
        }
        if back > 0.0 && !symmetric {
            // build123d has no asymmetric two-sided extrude, so the travel is
            // written as its two halves.
            self.losses.push(LossNote::new(
                LossKind::ParametricRecordOmitted,
                format!(
                    "{}: an asymmetric two-sided extent is emitted as two extrusions",
                    feature_name(feature)
                ),
            ));
            let suffix = if mode == "Mode.ADD" {
                String::new()
            } else {
                format!(", mode={mode}")
            };
            self.body(&format!(
                "extrude(amount={}{suffix})",
                geom::number(forward)
            ));
            self.body(&format!(
                "extrude(part.sketches[-1], amount={}, dir=(0, 0, -1){suffix})",
                geom::number(back)
            ));
        } else {
            self.body(&format!("extrude({})", arguments.join(", ")));
        }
        self.count("features");
        self.count("extrudes");
        true
    }

    fn emit_sketch(&mut self, sketch: &str, regions: &[Region]) {
        let Some(frame) = self.frames.get(sketch).copied() else {
            return;
        };
        let name = format!("sketch{}", self.sketch_seq);
        self.sketch_seq += 1;
        self.body(&format!(
            "with BuildSketch(Plane(origin={}, x_dir={}, z_dir={})) as {name}:",
            geom::tuple(frame.origin),
            geom::tuple(frame.x_axis),
            geom::tuple(frame.normal)
        ));
        let rings: Vec<(usize, bool)> = regions
            .iter()
            .flat_map(|region| {
                std::iter::once((region.outer, false))
                    .chain(region.holes.iter().map(|hole| (*hole, true)))
            })
            .collect();
        for (ring, subtract) in rings {
            self.emit_ring(sketch, ring, subtract);
        }
        self.count("sketches");
    }

    fn emit_ring(&mut self, sketch: &str, ring: usize, subtract: bool) {
        let mode = if subtract { ", mode=Mode.SUBTRACT" } else { "" };
        let Some(circle) = self
            .arrangements
            .get(sketch)
            .and_then(|arrangement| arrangement.rings.get(ring))
            .and_then(|ring| ring.circle)
        else {
            self.losses.push(LossNote::new(
                LossKind::ParametricRecordOmitted,
                format!(
                    "sketch {sketch} loop {ring} is not a single circle; this encoder \
                     writes circular profile loops only"
                ),
            ));
            return;
        };
        let ((u, v), radius) = circle;
        self.body(&format!(
            "    with Locations(({}, {})):",
            geom::number(u),
            geom::number(v)
        ));
        self.body(&format!("        Circle({}{mode})", geom::number(radius)));
    }

    fn fillet(&mut self, feature: &Feature, groups: &[cadmpeg_ir::features::FilletGroup]) -> bool {
        let mut emitted = false;
        for (index, group) in groups.iter().enumerate() {
            let RadiusSpec::Constant { radius } = &group.radius else {
                self.losses.push(LossNote::new(
                    LossKind::ParametricRecordOmitted,
                    format!(
                        "{}: group {index} has a non-constant radius law, which this \
                         encoder cannot express",
                        feature_name(feature)
                    ),
                ));
                continue;
            };
            let edges = resolve::blend_edges(self.ir, radius.0, &|face| self.topo.neighbours(face));
            if edges.is_empty() {
                self.losses.push(LossNote::new(
                    LossKind::ParametricRecordOmitted,
                    format!(
                        "{}: no blend face of radius {} was found in the solved body, so \
                         its edge selection could not be recovered",
                        feature_name(feature),
                        geom::number(radius.0)
                    ),
                ));
                continue;
            }
            let variable = format!("edges{}", self.blend_seq);
            self.blend_seq += 1;
            let predicates: Vec<String> = edges
                .iter()
                .map(|edge| {
                    format!(
                        "(abs(e.radius - {}) < 1e-4 and (e.arc_center - Vector{}).length < 1e-4)",
                        geom::number(edge.radius),
                        geom::tuple(edge.center)
                    )
                })
                .collect();
            self.body("");
            self.body(&format!(
                "# {} - radius {} mm  INFERRED (pre-blend edge = intersection of the blend \
                 face's two support carriers)",
                feature_name(feature),
                geom::number(radius.0)
            ));
            self.body(&format!("{variable} = ["));
            self.body("    e");
            self.body("    for e in part.edges()");
            self.body("    if e.geom_type == GeomType.CIRCLE");
            self.body(&format!("    and ({})", predicates.join("\n    or ")));
            self.body("]");
            self.body(&format!("if {variable}:"));
            self.body(&format!(
                "    fillet({variable}, radius={})",
                geom::number(radius.0)
            ));
            self.body("else:");
            self.body(&format!(
                "    print(\"warning: {} matched no edge\")",
                feature_name(feature)
            ));
            self.count("features");
            self.count("fillets");
            emitted = true;
        }
        emitted
    }
}

/// The sketch a profile reference points at, whatever variant it uses.
fn profile_sketch(profile: &ProfileRef) -> Option<&str> {
    match profile {
        ProfileRef::Sketch(sketch)
        | ProfileRef::SketchProfiles { sketch, .. }
        | ProfileRef::SketchRegions { sketch, .. }
        | ProfileRef::SketchEntities { sketch, .. }
        | ProfileRef::SketchSelection { sketch, .. } => Some(sketch.0.as_str()),
        _ => None,
    }
}

fn feature_name(feature: &Feature) -> &str {
    feature.name.as_deref().unwrap_or(feature.id.as_str())
}

const fn boolean_mode(op: BooleanOp) -> Option<&'static str> {
    match op {
        BooleanOp::Join | BooleanOp::NewBody => Some("Mode.ADD"),
        BooleanOp::Cut => Some("Mode.SUBTRACT"),
        BooleanOp::Intersect => Some("Mode.INTERSECT"),
        BooleanOp::Unresolved => None,
    }
}

const fn operation_name(op: BooleanOp) -> &'static str {
    match op {
        BooleanOp::Join => "join",
        BooleanOp::NewBody => "new body",
        BooleanOp::Cut => "cut",
        BooleanOp::Intersect => "intersect",
        BooleanOp::Unresolved => "unresolved",
    }
}

fn travel_label(back: f64, forward: f64) -> String {
    if back > 0.0 && (back - forward).abs() < 1e-9 {
        format!("symmetric {} mm total", geom::number(back + forward))
    } else if back > 0.0 {
        format!(
            "two-sided {} / {} mm",
            geom::number(back),
            geom::number(forward)
        )
    } else {
        format!("blind {} mm", geom::number(forward))
    }
}

fn family_name(definition: &FeatureDefinition) -> &'static str {
    match definition {
        FeatureDefinition::Revolve { .. } => "revolve",
        FeatureDefinition::Chamfer { .. } => "chamfer",
        FeatureDefinition::Hole { .. } => "hole",
        FeatureDefinition::Pattern { .. } => "pattern",
        FeatureDefinition::Shell { .. } => "shell",
        FeatureDefinition::Loft { .. } => "loft",
        FeatureDefinition::Sweep { .. } => "sweep",
        _ => "unsupported",
    }
}
