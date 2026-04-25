//! Thin CLI front-end for the `vibrt` library.
//!
//! Parses arguments, loads `scene.json` + `scene.bin` from disk, calls
//! `vibrt::render_to_pixels`, and saves the resulting RGBA buffer to PNG/EXR.
//! All real work lives in `vibrt/src/render.rs`; the Python addon talks to
//! the same library directly via PyO3 instead of going through this binary.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

use vibrt_native::{
    image_io, pipeline, render_to_pixels, scene_loader, RenderOptions, StdoutProgress,
};

#[derive(Parser)]
#[command(name = "vibrt", about = "Blender-native OptiX path-tracing renderer")]
struct Args {
    /// Input scene.json (scene.bin must be alongside).
    input: Option<PathBuf>,

    /// Output image (.exr or .png). Default: output.exr
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Override samples per pixel.
    #[arg(short, long)]
    spp: Option<u32>,

    /// Override max ray depth.
    #[arg(short, long)]
    depth: Option<u32>,

    /// Override indirect-contribution luminance clamp. 0 disables.
    #[arg(long = "clamp-indirect")]
    clamp_indirect: Option<f32>,

    /// Override image width.
    #[arg(long)]
    width: Option<u32>,

    /// Override image height.
    #[arg(long)]
    height: Option<u32>,

    /// Only compile the device code.
    #[arg(long)]
    compile_only: bool,

    /// Run the OptiX AI denoiser on the final image.
    #[arg(long)]
    denoise: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.compile_only {
        pipeline::compile_ptx()?;
        println!("Compilation OK.");
        return Ok(());
    }

    let input = args
        .input
        .as_ref()
        .context("input scene.json required (or use --compile-only)")?;
    let t_load = std::time::Instant::now();
    // The bin / json buffers must outlive `scene` because LoadedScene now
    // borrows directly from them (textures, mesh attributes) instead of
    // allocating its own copies.
    let mut json_text = String::new();
    let mut bin = Vec::new();
    let scene = scene_loader::load_scene_from_path(input, &mut json_text, &mut bin)?;
    println!("Scene load: {:.2?}", t_load.elapsed());

    let opts = RenderOptions {
        spp: args.spp,
        max_depth: args.depth,
        clamp_indirect: args.clamp_indirect,
        width: args.width,
        height: args.height,
        denoise: args.denoise,
    };

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("output.exr"));

    let mut progress = StdoutProgress;
    let out = render_to_pixels(&scene, &opts, &mut progress)?;
    image_io::save_image(&output, out.width, out.height, &out.pixels)?;
    println!("Saved {}", output.display());
    Ok(())
}
