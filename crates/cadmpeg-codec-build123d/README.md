<!-- SPDX-License-Identifier: Apache-2.0 -->

# cadmpeg-codec-build123d

Writes a cadmpeg document as a [build123d](https://build123d.readthedocs.io) Python program.

build123d is a code-first CAD library, so this target differs from every other encoder in the workspace: the output is source, not a container. The exported model is something a person can read, diff, review, and edit.

```sh
cadmpeg convert part.f3d -f build123d -o part.py
python part.py
```

## What it writes

The solved B-rep. Every face is rebuilt from its surface carrier and its solved boundary edges, and the faces are sewn into solids, so the encoder works for any document that carries topology — with or without a feature history.

Faces are emitted by one of three strategies:

| strategy | when |
| --- | --- |
| parametric band | every boundary edge is a full circle about the carrier axis |
| planar wire | a planar face, built widest ring first with the rest as holes |
| wire on carrier | any other analytic carrier, trimmed by a wire rebuilt from the solved edges |

## What it refuses

Anything it cannot rebuild exactly is reported as a `LossNote` rather than approximated. Two of those refusals are not merely conservative — they are required, because OpenCascade aborts the host process instead of returning an error:

- a carrier the IR cannot supply, which would reach `MakeFace` as null;
- a boundary that does not lie on the carrier it would trim.

Both are decided in closed form before the kernel sees them. The emitted program also guards each face individually, so one face the kernel refuses at run time cannot take the rest of the model down with it.

## Blend bands

A toroidal blend face is bounded by two circles that are equally consistent with the quarter tube of a fillet and with the three-quarter tube around it. STEP importers routinely reconstruct the wrong one, which inflates a filleted part by a full torus per blend. This encoder emits explicit parametric bounds, taking the arc its boundary circles actually describe, so a filleted part keeps its true volume.

## Requirements

The emitted program needs build123d 0.10 or newer. The encoder itself has no Python or geometry-kernel dependency: it only writes source.
