//! PyO3 bindings — gated by the `python` Cargo feature.
//!
//! Exposes a single `render(scene_json, mesh_blobs, opts, log_cb=None,
//! cancel_cb=None, texture_arrays=None) -> ndarray` function that the
//! Blender addon calls. Scene data crosses the FFI boundary as a `&str`
//! plus two lists of buffer-protocol-pinned numpy arrays (mesh blobs as
//! bytes, texture pixels as f32) — never copied; Rust borrows for the
//! duration of the call.
//!
//! All CUDA / OptiX work happens inside `py.allow_threads(...)` so the GIL
//! is released for the entire render. Progress callbacks reacquire the GIL
//! briefly via `Python::with_gil`.

use crate::{
    load_scene_from_bytes, render::RenderOutput, render_to_pixels, Progress, RenderOptions,
};
use numpy::ndarray::Array;
use numpy::{IntoPyArray, PyArray2, PyArray3};
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

/// Render an in-memory scene and return a `(pixels, depth)` tuple where
/// `pixels` is a `(height, width, 4)` float32 numpy array (linear RGBA,
/// top-left origin) and `depth` is a `(height, width)` float32 array of
/// per-pixel primary-ray hit distance from the camera (metres; misses
/// store the camera's `clip_end`). The addon turns `depth` into Mist /
/// Z passes so Cycles-authored compositors run on top of vibrt's output.
///
/// `mesh_blobs` is the per-blob byte-buffer list referenced by
/// `MeshDesc.vertices` / `.indices` / etc. — one numpy array per blob,
/// passed across PyO3 as `PyBuffer<u8>` (Python-side `.view(np.uint8)`
/// reinterprets f32 / u32 arrays without copying). The Rust loader picks
/// each entry by index and `bytemuck::cast_slice`s it to the field's type.
///
/// `texture_arrays` is the parallel list for `TextureDesc.array_index` —
/// kept separate because Rust takes texture data as typed
/// `Vec<PyBuffer<f32>>` for the GPU upload path.
#[pyfunction]
#[pyo3(signature = (
    scene_json, mesh_blobs, opts, log_cb = None, cancel_cb = None,
    texture_arrays = None,
))]
fn render<'py>(
    py: Python<'py>,
    scene_json: &str,
    mesh_blobs: Vec<PyBuffer<u8>>,
    opts: &Bound<'py, PyDict>,
    log_cb: Option<PyObject>,
    cancel_cb: Option<PyObject>,
    texture_arrays: Option<Vec<PyBuffer<f32>>>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<f32>>)> {
    let ro = parse_options(opts)?;
    let mut progress = PyProgress {
        log_cb,
        cancel_cb,
        cancelled: false,
    };

    // Build `&[&[u8]]` from the buffer list. Each PyBuffer pins its
    // source numpy array for the duration of this scope, so the slices
    // we hand to the loader stay valid through `allow_threads`.
    let mut blob_slices: Vec<&[u8]> = Vec::with_capacity(mesh_blobs.len());
    for (i, b) in mesh_blobs.iter().enumerate() {
        if !b.is_c_contiguous() {
            return Err(PyRuntimeError::new_err(format!(
                "mesh_blobs[{}] must be C-contiguous",
                i
            )));
        }
        let p = b.buf_ptr() as *const u8;
        let n = b.item_count();
        // SAFETY: PyBuffer holds a buffer-protocol view; the source is
        // pinned and cannot be resized while we hold it.
        blob_slices.push(unsafe { std::slice::from_raw_parts(p, n) });
    }

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
        tex_slices.push(unsafe { std::slice::from_raw_parts(p, n) });
    }

    // Run the whole pipeline without the GIL so CUDA driver threads don't
    // deadlock against the Python interpreter.
    let render_result: anyhow::Result<RenderOutput> = py.allow_threads(|| {
        let t_load = std::time::Instant::now();
        let scene = load_scene_from_bytes(scene_json, &blob_slices, &tex_slices)?;
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
    let depth = Array::from_shape_vec((h, w), out.depth).map_err(|e| {
        PyRuntimeError::new_err(format!("depth buffer reshape failed: {e}"))
    })?;
    Ok((arr.into_pyarray_bound(py), depth.into_pyarray_bound(py)))
}

#[pymodule]
fn vibrt_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render, m)?)?;
    Ok(())
}
