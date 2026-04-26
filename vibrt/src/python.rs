//! PyO3 bindings — gated by the `python` Cargo feature.
//!
//! Exposes a single `render(scene_json, scene_bin, opts, log_cb=None,
//! cancel_cb=None) -> ndarray` function that the Blender addon calls instead
//! of spawning `vibrt.exe`. The scene is handed across the FFI boundary as
//! `(json: &str, bin: &[u8])` — both borrow Python-owned buffers for the
//! duration of the call, so the 11 GB texture blob never leaves Python's
//! heap.
//!
//! All CUDA / OptiX work happens inside `py.allow_threads(...)` so the GIL
//! is released for the entire render. Progress callbacks reacquire the GIL
//! briefly via `Python::with_gil`.

use crate::{
    load_scene_from_bytes, render::RenderOutput, render_to_pixels, Progress, RenderOptions,
};
use numpy::ndarray::Array;
use numpy::{IntoPyArray, PyArray3};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `Progress` impl backed by Python callables. Logs go through `log_cb`
/// (called as `log_cb(msg: str)`) and cancellation polls `cancel_cb` (called
/// as `cancel_cb() -> bool`). Both callbacks are optional.
struct PyProgress {
    log_cb: Option<PyObject>,
    cancel_cb: Option<PyObject>,
    /// Sticky once Python has signalled cancel — avoids re-polling after the
    /// caller has already asked us to stop.
    cancelled: bool,
}

impl Progress for PyProgress {
    fn log(&mut self, msg: &str) {
        match &self.log_cb {
            Some(cb) => Python::with_gil(|py| {
                if let Err(err) = cb.call1(py, (msg,)) {
                    err.write_unraisable_bound(py, Some(&cb.bind(py).clone()));
                }
            }),
            None => println!("{msg}"),
        }
    }

    fn cancelled(&mut self) -> bool {
        if self.cancelled {
            return true;
        }
        let Some(cb) = &self.cancel_cb else {
            return false;
        };
        let result = Python::with_gil(|py| -> PyResult<bool> {
            let r = cb.call0(py)?;
            r.bind(py).is_truthy()
        });
        match result {
            Ok(v) => {
                self.cancelled = v;
                v
            }
            Err(_) => false,
        }
    }
}

/// Pull `RenderOptions` out of a Python dict. Missing or `None` entries leave
/// the field as `None` so the scene's stored defaults survive.
fn parse_options(opts: &Bound<'_, PyDict>) -> PyResult<RenderOptions> {
    fn get_opt<T>(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<T>>
    where
        for<'py> T: FromPyObject<'py>,
    {
        match d.get_item(key)? {
            Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
            _ => Ok(None),
        }
    }

    let denoise = match opts.get_item("denoise")? {
        Some(v) if !v.is_none() => v.extract::<bool>()?,
        _ => false,
    };
    Ok(RenderOptions {
        spp: get_opt(opts, "spp")?,
        max_depth: get_opt(opts, "max_depth")?,
        clamp_indirect: get_opt(opts, "clamp_indirect")?,
        width: get_opt(opts, "width")?,
        height: get_opt(opts, "height")?,
        denoise,
    })
}

/// Render an in-memory scene and return a `(height, width, 4)` float32 numpy
/// array. Equivalent to the CLI's `vibrt scene.json --output out.exr` minus
/// the disk roundtrip.
///
/// `scene_bin` accepts any buffer-protocol object (`bytes`, `bytearray`,
/// `memoryview`, numpy array, ...). Using a `bytearray` from
/// `exporter.export_scene_to_memory` skips the `bytes()` finalisation copy
/// that bouncing through `bytes` would otherwise pay (~12 GB on junk_shop).
///
/// `texture_arrays` is the optional per-texture pixel-array list that the
/// in-process exporter produces. Each entry is a contiguous f32 RGBA
/// buffer; `TextureDesc.array_index` in `scene_json` indexes into it. When
/// the list is empty (or omitted), every texture descriptor must carry a
/// `pixels` BlobRef pointing into `scene_bin` instead — the disk-path
/// fallback. Splitting textures off the bin sidesteps the bytearray-
/// growth memcpy spikes; on junk_shop that's ~6 s saved.
#[pyfunction]
#[pyo3(signature = (
    scene_json, scene_bin, opts, log_cb = None, cancel_cb = None,
    texture_arrays = None,
))]
fn render<'py>(
    py: Python<'py>,
    scene_json: &str,
    scene_bin: PyBuffer<u8>,
    opts: &Bound<'py, PyDict>,
    log_cb: Option<PyObject>,
    cancel_cb: Option<PyObject>,
    texture_arrays: Option<Vec<PyBuffer<f32>>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let ro = parse_options(opts)?;
    let mut progress = PyProgress {
        log_cb,
        cancel_cb,
        cancelled: false,
    };

    if !scene_bin.is_c_contiguous() {
        return Err(PyRuntimeError::new_err(
            "scene_bin must be C-contiguous (got a non-contiguous buffer)",
        ));
    }
    let bin_ptr = scene_bin.buf_ptr() as *const u8;
    let bin_len = scene_bin.item_count();
    // SAFETY: PyBuffer holds a refcounted "buffer view" via the Python
    // buffer protocol — while it's alive, the source object is pinned and
    // cannot be resized or reallocated. The slice we construct lives only
    // inside the closure below, which finishes before `scene_bin` (and its
    // PyBuffer) is dropped at function exit.
    let bin_slice: &[u8] = unsafe { std::slice::from_raw_parts(bin_ptr, bin_len) };

    // Resolve texture arrays the same way: each PyBuffer<f32> is pinned for
    // the duration of this call. Build `&[&[f32]]` so scene_loader can
    // borrow without needing to know about PyO3.
    let arrays = texture_arrays.unwrap_or_default();
    let mut tex_slices: Vec<&[f32]> = Vec::with_capacity(arrays.len());
    for (i, b) in arrays.iter().enumerate() {
        if !b.is_c_contiguous() {
            return Err(PyRuntimeError::new_err(format!(
                "texture_arrays[{}] must be C-contiguous",
                i
            )));
        }
        let p = b.buf_ptr() as *const f32;
        let n = b.item_count();
        // SAFETY: same buffer-protocol pinning argument as `bin_slice`.
        tex_slices.push(unsafe { std::slice::from_raw_parts(p, n) });
    }

    // Run the whole pipeline without the GIL so CUDA driver threads don't
    // deadlock against the Python interpreter. The `&str` borrow is fine
    // across allow_threads because Python-owned strings are immutable; the
    // pinned PyBuffers make `bin_slice` and `tex_slices` similarly safe.
    let render_result: anyhow::Result<RenderOutput> = py.allow_threads(|| {
        let t_load = std::time::Instant::now();
        let scene = load_scene_from_bytes(scene_json, bin_slice, &tex_slices)?;
        progress.log(&format!("Scene load: {:.2?}", t_load.elapsed()));
        render_to_pixels(&scene, &ro, &mut progress)
    });

    // Surface cooperative cancellation as KeyboardInterrupt so Blender's
    // operator machinery picks it up like Ctrl-C / Esc on any other engine.
    if progress.cancelled {
        return Err(PyKeyboardInterrupt::new_err("render cancelled"));
    }

    let out = render_result.map_err(|e| PyRuntimeError::new_err(format!("vibrt: {e:#}")))?;
    let h = out.height as usize;
    let w = out.width as usize;
    let arr = Array::from_shape_vec((h, w, 4), out.pixels).map_err(|e| {
        PyRuntimeError::new_err(format!("pixel buffer reshape failed: {e}"))
    })?;
    Ok(arr.into_pyarray_bound(py))
}

#[pymodule]
fn vibrt_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render, m)?)?;
    Ok(())
}
