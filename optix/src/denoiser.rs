use crate::context::DeviceContext;
use crate::error::{self, Result};
use crate::sys::FunctionTable;
use crate::types::{DenoiserAlphaMode, DenoiserModelKind, PixelFormat};
use std::ptr;
use std::sync::Arc;

/// Denoiser options.
#[derive(Debug, Clone, Default)]
pub struct DenoiserOptions {
    pub guide_albedo: bool,
    pub guide_normal: bool,
    pub denoise_alpha: DenoiserAlphaMode,
}

/// Denoiser memory requirements.
#[derive(Debug, Clone, Copy)]
pub struct DenoiserSizes {
    pub state_size: usize,
    pub with_overlap_scratch_size: usize,
    pub without_overlap_scratch_size: usize,
    pub overlap_window_size: u32,
    pub compute_average_color_size: usize,
    pub compute_intensity_size: usize,
}

/// A 2D image descriptor for the denoiser.
#[derive(Debug, Clone, Copy)]
pub struct Image2D {
    pub data: optix_sys::CUdeviceptr,
    pub width: u32,
    pub height: u32,
    pub row_stride: u32,
    pub pixel_stride: u32,
    pub format: PixelFormat,
}

impl Image2D {
    pub(crate) fn to_raw(&self) -> optix_sys::OptixImage2D {
        optix_sys::OptixImage2D {
            data: self.data,
            width: self.width,
            height: self.height,
            rowStrideInBytes: self.row_stride,
            pixelStrideInBytes: self.pixel_stride,
            format: self.format.to_raw(),
        }
    }

    fn empty_raw() -> optix_sys::OptixImage2D {
        optix_sys::OptixImage2D {
            data: 0,
            width: 0,
            height: 0,
            rowStrideInBytes: 0,
            pixelStrideInBytes: 0,
            format: optix_sys::OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT3,
        }
    }
}

/// Denoiser guide layer images.
#[derive(Debug, Clone, Default)]
pub struct DenoiserGuideLayer {
    pub albedo: Option<Image2D>,
    pub normal: Option<Image2D>,
    pub flow: Option<Image2D>,
}

/// A single denoiser input/output layer.
#[derive(Debug, Clone, Copy)]
pub struct DenoiserLayer {
    pub input: Image2D,
    pub output: Image2D,
    pub previous_output: Option<Image2D>,
}

/// Denoiser parameters.
#[derive(Debug, Clone, Default)]
pub struct DenoiserParams {
    pub hdr_intensity: optix_sys::CUdeviceptr,
    pub blend_factor: f32,
    pub hdr_average_color: optix_sys::CUdeviceptr,
    pub temporal_mode_use_previous_layers: bool,
}

/// RAII wrapper around an OptiX denoiser.
pub struct Denoiser {
    raw: optix_sys::OptixDenoiser,
    table: Arc<FunctionTable>,
}

unsafe impl Send for Denoiser {}
unsafe impl Sync for Denoiser {}

impl Denoiser {
    /// Create a new denoiser.
    pub fn new(
        ctx: &DeviceContext,
        model_kind: DenoiserModelKind,
        options: &DenoiserOptions,
    ) -> Result<Self> {
        let raw_options = optix_sys::OptixDenoiserOptions {
            guideAlbedo: options.guide_albedo as u32,
            guideNormal: options.guide_normal as u32,
            denoiseAlpha: options.denoise_alpha.to_raw(),
        };

        let mut raw: optix_sys::OptixDenoiser = ptr::null_mut();
        let result = unsafe {
            (ctx.table.raw.optixDenoiserCreate.unwrap())(
                ctx.raw,
                model_kind.to_raw(),
                &raw_options,
                &mut raw,
            )
        };
        error::check(result)?;

        Ok(Self {
            raw,
            table: ctx.table.clone(),
        })
    }

    /// Compute memory resources for the denoiser.
    pub fn compute_memory_resources(
        &self,
        max_input_width: u32,
        max_input_height: u32,
    ) -> Result<DenoiserSizes> {
        let mut sizes = optix_sys::OptixDenoiserSizes::default();
        let result = unsafe {
            (self.table.raw.optixDenoiserComputeMemoryResources.unwrap())(
                self.raw,
                max_input_width,
                max_input_height,
                &mut sizes,
            )
        };
        error::check(result)?;

        Ok(DenoiserSizes {
            state_size: sizes.stateSizeInBytes,
            with_overlap_scratch_size: sizes.withOverlapScratchSizeInBytes,
            without_overlap_scratch_size: sizes.withoutOverlapScratchSizeInBytes,
            overlap_window_size: sizes.overlapWindowSizeInPixels,
            compute_average_color_size: sizes.computeAverageColorSizeInBytes,
            compute_intensity_size: sizes.computeIntensitySizeInBytes,
        })
    }

    /// Set up the denoiser with allocated memory.
    pub fn setup(
        &self,
        stream: optix_sys::CUstream,
        input_width: u32,
        input_height: u32,
        state: optix_sys::CUdeviceptr,
        state_size: usize,
        scratch: optix_sys::CUdeviceptr,
        scratch_size: usize,
    ) -> Result<()> {
        let result = unsafe {
            (self.table.raw.optixDenoiserSetup.unwrap())(
                self.raw,
                stream,
                input_width,
                input_height,
                state,
                state_size,
                scratch,
                scratch_size,
            )
        };
        error::check(result)
    }

    /// Invoke the denoiser.
    pub fn invoke(
        &self,
        stream: optix_sys::CUstream,
        params: &DenoiserParams,
        state: optix_sys::CUdeviceptr,
        state_size: usize,
        guide_layer: &DenoiserGuideLayer,
        layers: &[DenoiserLayer],
        input_offset_x: u32,
        input_offset_y: u32,
        scratch: optix_sys::CUdeviceptr,
        scratch_size: usize,
    ) -> Result<()> {
        let raw_params = optix_sys::OptixDenoiserParams {
            hdrIntensity: params.hdr_intensity,
            blendFactor: params.blend_factor,
            hdrAverageColor: params.hdr_average_color,
            temporalModeUsePreviousLayers: params.temporal_mode_use_previous_layers as u32,
        };

        let raw_guide = optix_sys::OptixDenoiserGuideLayer {
            albedo: guide_layer
                .albedo
                .as_ref()
                .map_or(Image2D::empty_raw(), |i| i.to_raw()),
            normal: guide_layer
                .normal
                .as_ref()
                .map_or(Image2D::empty_raw(), |i| i.to_raw()),
            flow: guide_layer
                .flow
                .as_ref()
                .map_or(Image2D::empty_raw(), |i| i.to_raw()),
            previousOutputInternalGuideLayer: Image2D::empty_raw(),
            outputInternalGuideLayer: Image2D::empty_raw(),
            flowTrustworthiness: Image2D::empty_raw(),
        };

        let raw_layers: Vec<optix_sys::OptixDenoiserLayer> = layers
            .iter()
            .map(|l| optix_sys::OptixDenoiserLayer {
                input: l.input.to_raw(),
                previousOutput: l
                    .previous_output
                    .as_ref()
                    .map_or(Image2D::empty_raw(), |i| i.to_raw()),
                output: l.output.to_raw(),
                type_: optix_sys::OptixDenoiserAOVType::OPTIX_DENOISER_AOV_TYPE_NONE,
            })
            .collect();

        let result = unsafe {
            (self.table.raw.optixDenoiserInvoke.unwrap())(
                self.raw,
                stream,
                &raw_params,
                state,
                state_size,
                &raw_guide,
                raw_layers.as_ptr(),
                raw_layers.len() as u32,
                input_offset_x,
                input_offset_y,
                scratch,
                scratch_size,
            )
        };
        error::check(result)
    }
}

impl Drop for Denoiser {
    fn drop(&mut self) {
        unsafe {
            (self.table.raw.optixDenoiserDestroy.unwrap())(self.raw);
        }
    }
}
