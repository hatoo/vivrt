//! Export-time constant folding for `ColorGraph`.
//!
//! Walks the graph in topological order. A node whose every input resolves to
//! a `ColorNode::Const` is evaluated on the CPU and replaced in-place with a
//! `ColorNode::Const` carrying the result. Graph length and `output` index
//! are preserved, so downstream indices never need remapping and the GPU
//! interpreter (`eval_color_graph` in `devicecode.cu`) needs no change.
//!
//! Op evaluators mirror the GPU code in `devicecode.cu` (`mix_blend_rgb`,
//! `math_apply_rgb`, `rgb_to_hsv_bl`, `hsv_to_rgb_bl`, etc.) branch-for-branch.

use crate::principled::{parse_blend, parse_math_op};
use crate::scene_format::{ColorFactor, ColorGraph, ColorNode};

pub fn fold_constants(graph: &ColorGraph) -> ColorGraph {
    let mut nodes: Vec<ColorNode> = Vec::with_capacity(graph.nodes.len());
    for n in &graph.nodes {
        let folded = try_fold(n, &nodes).unwrap_or_else(|| n.clone());
        nodes.push(folded);
    }
    ColorGraph {
        nodes,
        output: graph.output,
    }
}

fn as_const(prev: &[ColorNode], idx: u32) -> Option<[f32; 3]> {
    match prev.get(idx as usize)? {
        ColorNode::Const { rgb } => Some(*rgb),
        _ => None,
    }
}

fn try_fold(node: &ColorNode, prev: &[ColorNode]) -> Option<ColorNode> {
    match node {
        ColorNode::Const { .. } | ColorNode::ImageTex { .. } | ColorNode::VertexColor {} => None,
        ColorNode::Mix {
            a,
            b,
            fac,
            blend,
            clamp,
        } => {
            let a_rgb = as_const(prev, *a)?;
            let b_rgb = as_const(prev, *b)?;
            let fac_f = match fac {
                ColorFactor::Const(v) => *v,
                ColorFactor::Node { node } => {
                    let f = as_const(prev, *node)?;
                    luminance(f)
                }
            };
            let blend_id = parse_blend(blend).ok()?;
            Some(const_node(eval_mix(a_rgb, b_rgb, fac_f, blend_id, *clamp)))
        }
        ColorNode::Invert { input, fac } => {
            let c = as_const(prev, *input)?;
            Some(const_node(eval_invert(c, *fac)))
        }
        ColorNode::Math {
            input,
            op,
            b,
            c,
            clamp,
            swap,
        } => {
            let inp = as_const(prev, *input)?;
            let op_id = parse_math_op(op).ok()?;
            let mut out = eval_math(inp, op_id, *b, *c, *swap);
            if *clamp {
                out = clamp01(out);
            }
            Some(const_node(out))
        }
        ColorNode::HueSat {
            input,
            hue,
            saturation,
            value,
            fac,
        } => {
            let src = as_const(prev, *input)?;
            Some(const_node(eval_hue_sat(src, *hue, *saturation, *value, *fac)))
        }
        ColorNode::RgbCurve { input, lut } => {
            let src = as_const(prev, *input)?;
            if lut.len() != 768 {
                return None; // validation will fire later with a clearer error
            }
            Some(const_node(eval_rgb_curve(src, lut)))
        }
        ColorNode::BrightContrast {
            input,
            bright,
            contrast,
        } => {
            let c = as_const(prev, *input)?;
            Some(const_node(eval_bright_contrast(c, *bright, *contrast)))
        }
    }
}

fn const_node(rgb: [f32; 3]) -> ColorNode {
    ColorNode::Const { rgb }
}

// Matches the Rec.709 luminance used by the GPU at devicecode.cu:667.
fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn clamp01(c: [f32; 3]) -> [f32; 3] {
    [
        c[0].clamp(0.0, 1.0),
        c[1].clamp(0.0, 1.0),
        c[2].clamp(0.0, 1.0),
    ]
}

// Port of mix_blend_rgb (devicecode.cu:452-528). Keep branches numbered the
// same so parity is easy to audit.
fn eval_mix(a: [f32; 3], b: [f32; 3], fac: f32, blend: u32, clamp_out: bool) -> [f32; 3] {
    let facm = 1.0 - fac;
    let mut out = match blend {
        // MIX
        0 => [a[0] * facm + b[0] * fac, a[1] * facm + b[1] * fac, a[2] * facm + b[2] * fac],
        // MULTIPLY
        1 => [
            a[0] * (facm + b[0] * fac),
            a[1] * (facm + b[1] * fac),
            a[2] * (facm + b[2] * fac),
        ],
        // ADD
        2 => [a[0] + b[0] * fac, a[1] + b[1] * fac, a[2] + b[2] * fac],
        // SUBTRACT
        3 => [a[0] - b[0] * fac, a[1] - b[1] * fac, a[2] - b[2] * fac],
        // SCREEN
        4 => [
            1.0 - (facm + (1.0 - b[0]) * fac) * (1.0 - a[0]),
            1.0 - (facm + (1.0 - b[1]) * fac) * (1.0 - a[1]),
            1.0 - (facm + (1.0 - b[2]) * fac) * (1.0 - a[2]),
        ],
        // DIVIDE — per-channel, guard against div-by-zero
        5 => [
            if b[0] == 0.0 { a[0] } else { a[0] * facm + a[0] / b[0] * fac },
            if b[1] == 0.0 { a[1] } else { a[1] * facm + a[1] / b[1] * fac },
            if b[2] == 0.0 { a[2] } else { a[2] * facm + a[2] / b[2] * fac },
        ],
        // DIFFERENCE
        6 => [
            a[0] * facm + (a[0] - b[0]).abs() * fac,
            a[1] * facm + (a[1] - b[1]).abs() * fac,
            a[2] * facm + (a[2] - b[2]).abs() * fac,
        ],
        // DARKEN
        7 => [
            a[0] * facm + a[0].min(b[0]) * fac,
            a[1] * facm + a[1].min(b[1]) * fac,
            a[2] * facm + a[2].min(b[2]) * fac,
        ],
        // LIGHTEN (Blender's asymmetric form)
        8 => [
            a[0].max(b[0] * fac),
            a[1].max(b[1] * fac),
            a[2].max(b[2] * fac),
        ],
        // OVERLAY per-channel
        9 => {
            let ov = |ax: f32, bx: f32| -> f32 {
                if ax < 0.5 {
                    ax * (facm + 2.0 * fac * bx)
                } else {
                    1.0 - (facm + 2.0 * fac * (1.0 - bx)) * (1.0 - ax)
                }
            };
            [ov(a[0], b[0]), ov(a[1], b[1]), ov(a[2], b[2])]
        }
        // SOFT_LIGHT
        10 => {
            let scr = [
                1.0 - (1.0 - b[0]) * (1.0 - a[0]),
                1.0 - (1.0 - b[1]) * (1.0 - a[1]),
                1.0 - (1.0 - b[2]) * (1.0 - a[2]),
            ];
            let inner = [
                (1.0 - a[0]) * b[0] * a[0] + a[0] * scr[0],
                (1.0 - a[1]) * b[1] * a[1] + a[1] * scr[1],
                (1.0 - a[2]) * b[2] * a[2] + a[2] * scr[2],
            ];
            [
                a[0] * facm + inner[0] * fac,
                a[1] * facm + inner[1] * fac,
                a[2] * facm + inner[2] * fac,
            ]
        }
        // LINEAR_LIGHT
        11 => [
            a[0] + (b[0] * 2.0 - 1.0) * fac,
            a[1] + (b[1] * 2.0 - 1.0) * fac,
            a[2] + (b[2] * 2.0 - 1.0) * fac,
        ],
        _ => [a[0] * facm + b[0] * fac, a[1] * facm + b[1] * fac, a[2] * facm + b[2] * fac],
    };
    if clamp_out {
        out = clamp01(out);
    }
    out
}

// Port of math_apply_rgb (devicecode.cu:530-584). `clamp` is applied by the
// caller (matches the GPU order in eval_color_graph).
fn eval_math(inp: [f32; 3], op: u32, b: f32, c: f32, swap: bool) -> [f32; 3] {
    match op {
        0 => [inp[0] + b, inp[1] + b, inp[2] + b], // ADD
        1 => {
            if swap {
                [b - inp[0], b - inp[1], b - inp[2]]
            } else {
                [inp[0] - b, inp[1] - b, inp[2] - b]
            }
        }
        2 => [inp[0] * b, inp[1] * b, inp[2] * b], // MULTIPLY
        3 => {
            if swap {
                [
                    if inp[0] == 0.0 { 0.0 } else { b / inp[0] },
                    if inp[1] == 0.0 { 0.0 } else { b / inp[1] },
                    if inp[2] == 0.0 { 0.0 } else { b / inp[2] },
                ]
            } else if b == 0.0 {
                [0.0, 0.0, 0.0]
            } else {
                [inp[0] / b, inp[1] / b, inp[2] / b]
            }
        }
        4 => {
            if swap {
                let base = b.max(0.0);
                [base.powf(inp[0]), base.powf(inp[1]), base.powf(inp[2])]
            } else {
                [
                    inp[0].max(0.0).powf(b),
                    inp[1].max(0.0).powf(b),
                    inp[2].max(0.0).powf(b),
                ]
            }
        }
        5 => [inp[0] * b + c, inp[1] * b + c, inp[2] * b + c], // MULTIPLY_ADD
        6 => [inp[0].min(b), inp[1].min(b), inp[2].min(b)],    // MINIMUM
        7 => [inp[0].max(b), inp[1].max(b), inp[2].max(b)],    // MAXIMUM
        _ => [inp[0] + b, inp[1] + b, inp[2] + b],
    }
}

fn eval_invert(c: [f32; 3], fac: f32) -> [f32; 3] {
    let facm = 1.0 - fac;
    [
        c[0] * facm + (1.0 - c[0]) * fac,
        c[1] * facm + (1.0 - c[1]) * fac,
        c[2] * facm + (1.0 - c[2]) * fac,
    ]
}

// Port of rgb_to_hsv_bl (devicecode.cu:587-605). HSV channels in [0,1].
fn rgb_to_hsv_bl(c: [f32; 3]) -> [f32; 3] {
    let cmax = c[0].max(c[1]).max(c[2]);
    let cmin = c[0].min(c[1]).min(c[2]);
    let v = cmax;
    let d = cmax - cmin;
    let s = if cmax > 0.0 { d / cmax } else { 0.0 };
    let mut h = 0.0;
    if d > 0.0 {
        h = if cmax == c[0] {
            (c[1] - c[2]) / d + if c[1] < c[2] { 6.0 } else { 0.0 }
        } else if cmax == c[1] {
            (c[2] - c[0]) / d + 2.0
        } else {
            (c[0] - c[1]) / d + 4.0
        };
        h *= 1.0 / 6.0;
    }
    [h, s, v]
}

// Port of hsv_to_rgb_bl (devicecode.cu:607-626).
fn hsv_to_rgb_bl(c: [f32; 3]) -> [f32; 3] {
    let h = c[0] - c[0].floor(); // wrap to [0,1)
    let s = c[1].clamp(0.0, 1.0);
    let v = c[2];
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let mut ii = (i as i32) % 6;
    if ii < 0 {
        ii += 6;
    }
    match ii {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

// Port of the HueSat block in eval_color_graph (devicecode.cu:720-737).
fn eval_hue_sat(src: [f32; 3], hue: f32, sat: f32, val: f32, fac: f32) -> [f32; 3] {
    let mut hsv = rgb_to_hsv_bl(src);
    hsv[0] += hue - 0.5;
    hsv[1] = (hsv[1] * sat).clamp(0.0, 1.0);
    hsv[2] *= val;
    let shifted = hsv_to_rgb_bl(hsv);
    let facm = 1.0 - fac;
    [
        src[0] * facm + shifted[0] * fac,
        src[1] * facm + shifted[1] * fac,
        src[2] * facm + shifted[2] * fac,
    ]
}

// Port of the RgbCurve fetch/lerp (devicecode.cu:699-707). LUT layout is
// channel-major: R[0..256], G[0..256], B[0..256].
fn eval_rgb_curve(src: [f32; 3], lut: &[f32]) -> [f32; 3] {
    let fetch = |channel: usize, x: f32| -> f32 {
        let xc = x.clamp(0.0, 1.0);
        let fx = xc * 255.0;
        let i0 = fx.floor() as usize;
        let i1 = if i0 < 255 { i0 + 1 } else { 255 };
        let t = fx - i0 as f32;
        let base = channel * 256;
        lut[base + i0] * (1.0 - t) + lut[base + i1] * t
    };
    [fetch(0, src[0]), fetch(1, src[1]), fetch(2, src[2])]
}

// Port of the BrightContrast block (devicecode.cu:709-719). Only bottom-clamps.
fn eval_bright_contrast(c: [f32; 3], bright: f32, contrast: f32) -> [f32; 3] {
    let a = 1.0 + contrast;
    let b = bright - 0.5 * contrast;
    [
        (a * c[0] + b).max(0.0),
        (a * c[1] + b).max(0.0),
        (a * c[2] + b).max(0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cg(nodes: Vec<ColorNode>, output: Option<u32>) -> ColorGraph {
        ColorGraph { nodes, output }
    }

    fn out_rgb(g: &ColorGraph) -> [f32; 3] {
        let idx = g.output.unwrap_or((g.nodes.len() - 1) as u32) as usize;
        match &g.nodes[idx] {
            ColorNode::Const { rgb } => *rgb,
            _ => panic!("output node is not Const"),
        }
    }

    fn approx(a: [f32; 3], b: [f32; 3], eps: f32) {
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < eps,
                "component {} differs: {} vs {}",
                i,
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn fold_invert_full() {
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.25, 0.5, 0.75] },
                ColorNode::Invert { input: 0, fac: 1.0 },
            ],
            None,
        );
        let f = fold_constants(&g);
        assert_eq!(f.nodes.len(), 2);
        approx(out_rgb(&f), [0.75, 0.5, 0.25], 1e-6);
    }

    #[test]
    fn fold_invert_partial() {
        // fac=0.5: out = 0.5*rgb + 0.5*(1-rgb) = 0.5
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.1, 0.9, 0.3] },
                ColorNode::Invert { input: 0, fac: 0.5 },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.5, 0.5, 0.5], 1e-6);
    }

    #[test]
    fn fold_math_multiply() {
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.1, 0.2, 0.3] },
                ColorNode::Math {
                    input: 0,
                    op: "multiply".to_string(),
                    b: 2.0,
                    c: 0.0,
                    clamp: false,
                    swap: false,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.2, 0.4, 0.6], 1e-6);
    }

    #[test]
    fn fold_math_subtract_swap() {
        // swap=true: out = b - input
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.25, 0.5, 0.75] },
                ColorNode::Math {
                    input: 0,
                    op: "subtract".to_string(),
                    b: 1.0,
                    c: 0.0,
                    clamp: false,
                    swap: true,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.75, 0.5, 0.25], 1e-6);
    }

    #[test]
    fn fold_huesat_identity() {
        // hue=0.5, sat=1, val=1, fac=1 is identity.
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.3, 0.6, 0.9] },
                ColorNode::HueSat {
                    input: 0,
                    hue: 0.5,
                    saturation: 1.0,
                    value: 1.0,
                    fac: 1.0,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.3, 0.6, 0.9], 1e-5);
    }

    #[test]
    fn fold_huesat_hue_rotate_red_to_green() {
        // Red (1,0,0) with hue += 1/3 should land on green (0,1,0).
        // Using hue field = 0.5 + 1/3 (since default is 0.5).
        let g = cg(
            vec![
                ColorNode::Const { rgb: [1.0, 0.0, 0.0] },
                ColorNode::HueSat {
                    input: 0,
                    hue: 0.5 + 1.0 / 3.0,
                    saturation: 1.0,
                    value: 1.0,
                    fac: 1.0,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.0, 1.0, 0.0], 1e-5);
    }

    #[test]
    fn fold_bright_contrast_bright_only() {
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.2, 0.4, 0.6] },
                ColorNode::BrightContrast {
                    input: 0,
                    bright: 0.1,
                    contrast: 0.0,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.3, 0.5, 0.7], 1e-6);
    }

    #[test]
    fn fold_bright_contrast_floor_at_zero() {
        // a=1+(-1)=0, b = -0.5 - (-0.5) = 0.5? Let's pick something that goes negative.
        // bright=-0.5, contrast=0: a=1, b=-0.5, out = rgb - 0.5 (clamped >=0).
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.1, 0.4, 0.9] },
                ColorNode::BrightContrast {
                    input: 0,
                    bright: -0.5,
                    contrast: 0.0,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.0, 0.0, 0.4], 1e-6);
    }

    #[test]
    fn fold_mix_midpoint() {
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.0, 0.0, 0.0] },
                ColorNode::Const { rgb: [1.0, 1.0, 1.0] },
                ColorNode::Mix {
                    a: 0,
                    b: 1,
                    fac: ColorFactor::Const(0.5),
                    blend: "mix".to_string(),
                    clamp: false,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.5, 0.5, 0.5], 1e-6);
    }

    #[test]
    fn fold_mix_fac_node_luminance() {
        // fac node is grey (0.5, 0.5, 0.5) → luminance ≈ 0.5.
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.0, 0.0, 0.0] },
                ColorNode::Const { rgb: [1.0, 1.0, 1.0] },
                ColorNode::Const { rgb: [0.5, 0.5, 0.5] },
                ColorNode::Mix {
                    a: 0,
                    b: 1,
                    fac: ColorFactor::Node { node: 2 },
                    blend: "mix".to_string(),
                    clamp: false,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.5, 0.5, 0.5], 1e-6);
    }

    #[test]
    fn fold_rgb_curve_identity() {
        // Identity LUT: lut[channel*256 + i] = i/255.
        let mut lut = Vec::with_capacity(768);
        for _ in 0..3 {
            for i in 0..256 {
                lut.push(i as f32 / 255.0);
            }
        }
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.2, 0.5, 0.8] },
                ColorNode::RgbCurve { input: 0, lut },
            ],
            None,
        );
        let f = fold_constants(&g);
        approx(out_rgb(&f), [0.2, 0.5, 0.8], 1.0 / 255.0);
    }

    #[test]
    fn no_fold_image_tex_chain() {
        let g = cg(
            vec![
                ColorNode::ImageTex {
                    tex: 0,
                    uv: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                },
                ColorNode::Math {
                    input: 0,
                    op: "multiply".to_string(),
                    b: 2.0,
                    c: 0.0,
                    clamp: false,
                    swap: false,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        // Neither node should be rewritten.
        assert!(matches!(f.nodes[0], ColorNode::ImageTex { .. }));
        assert!(matches!(f.nodes[1], ColorNode::Math { .. }));
    }

    #[test]
    fn no_fold_vertex_color_chain() {
        let g = cg(
            vec![
                ColorNode::VertexColor {},
                ColorNode::HueSat {
                    input: 0,
                    hue: 0.5,
                    saturation: 1.0,
                    value: 1.0,
                    fac: 1.0,
                },
            ],
            None,
        );
        let f = fold_constants(&g);
        assert!(matches!(f.nodes[0], ColorNode::VertexColor {}));
        assert!(matches!(f.nodes[1], ColorNode::HueSat { .. }));
    }

    #[test]
    fn fold_deep_chain_collapses() {
        // Const -> Math -> Math -> Mix(_, Const, fac=0.5) → terminal Const
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.1, 0.2, 0.4] },
                ColorNode::Math {
                    input: 0,
                    op: "multiply".to_string(),
                    b: 2.0,
                    c: 0.0,
                    clamp: false,
                    swap: false,
                }, // (0.2, 0.4, 0.8)
                ColorNode::Math {
                    input: 1,
                    op: "add".to_string(),
                    b: 0.1,
                    c: 0.0,
                    clamp: false,
                    swap: false,
                }, // (0.3, 0.5, 0.9)
                ColorNode::Const { rgb: [0.1, 0.1, 0.1] },
                ColorNode::Mix {
                    a: 2,
                    b: 3,
                    fac: ColorFactor::Const(0.5),
                    blend: "mix".to_string(),
                    clamp: false,
                }, // average = (0.2, 0.3, 0.5)
            ],
            None,
        );
        let f = fold_constants(&g);
        for i in 0..5 {
            assert!(
                matches!(f.nodes[i], ColorNode::Const { .. }),
                "node {} should have folded to Const",
                i
            );
        }
        approx(out_rgb(&f), [0.2, 0.3, 0.5], 1e-6);
    }

    #[test]
    fn output_index_preserved() {
        let g = cg(
            vec![
                ColorNode::Const { rgb: [0.1, 0.1, 0.1] },
                ColorNode::Math {
                    input: 0,
                    op: "add".to_string(),
                    b: 0.5,
                    c: 0.0,
                    clamp: false,
                    swap: false,
                },
                ColorNode::Const { rgb: [0.9, 0.9, 0.9] },
            ],
            Some(1),
        );
        let f = fold_constants(&g);
        assert_eq!(f.output, Some(1));
        approx(out_rgb(&f), [0.6, 0.6, 0.6], 1e-6);
    }
}
