//! Output linear RGBA float buffer as EXR or tonemapped PNG.

use anyhow::{Context, Result};
use std::path::Path;

/// Save a linear RGBA float image. Format chosen by extension (.exr, .raw, or .png).
///
/// `.raw` is a bespoke format for the Blender addon: 16-byte header (`"VBLT"`
/// magic + width:u32 + height:u32 + channels:u32=4), then `width*height*4` LE
/// f32 pixels in bottom-left origin (matches Blender's `img.pixels` convention).
pub fn save_image(path: &Path, width: u32, height: u32, rgba: &[f32]) -> Result<()> {
    assert_eq!(rgba.len(), (width * height * 4) as usize);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "exr" => save_exr(path, width, height, rgba),
        "raw" => save_raw(path, width, height, rgba),
        "png" => save_png(path, width, height, rgba),
        other => {
            eprintln!(
                "[vibrt] warn: unrecognized output extension {:?} for {} — defaulting to PNG (supported: exr, raw, png)",
                other,
                path.display()
            );
            save_png(path, width, height, rgba)
        }
    }
}

fn save_raw(path: &Path, width: u32, height: u32, rgba: &[f32]) -> Result<()> {
    use std::io::Write;
    // Our raygen writes row 0 as bottom-of-scene; that already matches Blender's
    // bottom-left origin, so no Y-flip here.
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = std::io::BufWriter::new(file);
    w.write_all(b"VBLT")?;
    w.write_all(&width.to_le_bytes())?;
    w.write_all(&height.to_le_bytes())?;
    w.write_all(&4u32.to_le_bytes())?;
    let bytes = unsafe { std::slice::from_raw_parts(rgba.as_ptr() as *const u8, rgba.len() * 4) };
    w.write_all(bytes)?;
    Ok(())
}

fn save_exr(path: &Path, width: u32, height: u32, rgba: &[f32]) -> Result<()> {
    use image::codecs::openexr::OpenExrEncoder;
    use image::ImageEncoder;
    // Flip Y so image origin matches Blender's "bottom-left = (0,0)" expectation:
    // OptiX writes with y=0 at top, so we flip here before encoding.
    let mut flipped = vec![0.0f32; rgba.len()];
    for y in 0..height {
        let src_row = (height - 1 - y) as usize * width as usize * 4;
        let dst_row = y as usize * width as usize * 4;
        flipped[dst_row..dst_row + (width as usize) * 4]
            .copy_from_slice(&rgba[src_row..src_row + (width as usize) * 4]);
    }
    let byte_len = flipped.len() * 4;
    let bytes = unsafe { std::slice::from_raw_parts(flipped.as_ptr() as *const u8, byte_len) };
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let w = std::io::BufWriter::new(file);
    let encoder = OpenExrEncoder::new(w);
    encoder
        .write_image(bytes, width, height, image::ExtendedColorType::Rgba32F)
        .context("writing EXR")?;
    Ok(())
}

fn save_png(path: &Path, width: u32, height: u32, rgba: &[f32]) -> Result<()> {
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;
    let mut rgb8 = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        let src_y = height - 1 - y;
        for x in 0..width {
            let idx = (src_y * width + x) as usize * 4;
            rgb8.push(linear_to_srgb_u8(rgba[idx]));
            rgb8.push(linear_to_srgb_u8(rgba[idx + 1]));
            rgb8.push(linear_to_srgb_u8(rgba[idx + 2]));
        }
    }
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let w = std::io::BufWriter::new(file);
    let encoder = PngEncoder::new(w);
    encoder
        .write_image(&rgb8, width, height, image::ExtendedColorType::Rgb8)
        .context("writing PNG")?;
    Ok(())
}

fn linear_to_srgb_u8(x: f32) -> u8 {
    // Simple Reinhard-ish compression before sRGB gamma — avoids blown highlights
    // when the renderer outputs HDR values.
    let x = if x.is_finite() { x.max(0.0) } else { 0.0 };
    let tonemapped = x / (1.0 + x);
    let srgb = if tonemapped <= 0.0031308 {
        12.92 * tonemapped
    } else {
        1.055 * tonemapped.powf(1.0 / 2.4) - 0.055
    };
    (srgb.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
