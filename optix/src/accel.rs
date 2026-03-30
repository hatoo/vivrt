use crate::context::DeviceContext;
use crate::error::{self, Result};
use crate::types::{BuildFlags, BuildOperation, GeometryFlags, IndicesFormat, VertexFormat};
use std::mem::MaybeUninit;
use std::ptr;

/// Options for building an acceleration structure.
#[derive(Debug, Clone)]
pub struct AccelBuildOptions {
    pub build_flags: BuildFlags,
    pub operation: BuildOperation,
}

impl Default for AccelBuildOptions {
    fn default() -> Self {
        Self {
            build_flags: BuildFlags::NONE,
            operation: BuildOperation::Build,
        }
    }
}

/// Memory requirements for an acceleration structure build.
#[derive(Debug, Clone, Copy)]
pub struct AccelBufferSizes {
    pub output_size: usize,
    pub temp_size: usize,
    pub temp_update_size: usize,
}

/// Property to emit after an accel build.
#[derive(Debug, Clone, Copy)]
pub enum AccelEmitProperty {
    CompactedSize(optix_sys::CUdeviceptr),
}

/// A build input for acceleration structures.
///
/// This enum replaces the raw `OptixBuildInput` union with a type-safe interface.
pub enum BuildInput<'a> {
    Triangles(TriangleArrayInput<'a>),
    CustomPrimitives(CustomPrimitiveInput<'a>),
    Instances(InstanceArrayInput),
}

/// Triangle geometry build input.
pub struct TriangleArrayInput<'a> {
    pub vertex_buffers: &'a [optix_sys::CUdeviceptr],
    pub num_vertices: u32,
    pub vertex_format: VertexFormat,
    pub vertex_stride: u32,
    pub index_buffer: optix_sys::CUdeviceptr,
    pub num_index_triplets: u32,
    pub index_format: IndicesFormat,
    pub index_stride: u32,
    pub flags: &'a [GeometryFlags],
    pub num_sbt_records: u32,
    pub primitive_index_offset: u32,
}

impl<'a> TriangleArrayInput<'a> {
    /// Create a simple triangle input without indices.
    pub fn new(
        vertex_buffers: &'a [optix_sys::CUdeviceptr],
        num_vertices: u32,
        vertex_format: VertexFormat,
        vertex_stride: u32,
        flags: &'a [GeometryFlags],
    ) -> Self {
        Self {
            vertex_buffers,
            num_vertices,
            vertex_format,
            vertex_stride,
            index_buffer: 0,
            num_index_triplets: 0,
            index_format: IndicesFormat::None,
            index_stride: 0,
            flags,
            num_sbt_records: flags.len() as u32,
            primitive_index_offset: 0,
        }
    }

    /// Set index buffer for indexed triangles.
    pub fn with_indices(
        mut self,
        index_buffer: optix_sys::CUdeviceptr,
        num_index_triplets: u32,
        index_format: IndicesFormat,
        index_stride: u32,
    ) -> Self {
        self.index_buffer = index_buffer;
        self.num_index_triplets = num_index_triplets;
        self.index_format = index_format;
        self.index_stride = index_stride;
        self
    }
}

/// Custom (AABB) primitive build input.
pub struct CustomPrimitiveInput<'a> {
    pub aabb_buffers: &'a [optix_sys::CUdeviceptr],
    pub num_primitives: u32,
    pub stride: u32,
    pub flags: &'a [GeometryFlags],
    pub num_sbt_records: u32,
    pub primitive_index_offset: u32,
}

/// Instance array build input for IAS (Instance Acceleration Structure).
pub struct InstanceArrayInput {
    pub instances: optix_sys::CUdeviceptr,
    pub num_instances: u32,
}

/// Convert safe BuildInput to raw, storing temporary data in the provided buffers.
fn build_input_to_raw(input: &BuildInput, raw_flags: &mut Vec<u32>) -> optix_sys::OptixBuildInput {
    let mut raw: optix_sys::OptixBuildInput = unsafe { MaybeUninit::zeroed().assume_init() };

    match input {
        BuildInput::Triangles(tri) => {
            raw.type_ = optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            let flags_start = raw_flags.len();
            raw_flags.extend(tri.flags.iter().map(|f| f.bits()));

            let ta = optix_sys::OptixBuildInputTriangleArray {
                vertexBuffers: tri.vertex_buffers.as_ptr(),
                numVertices: tri.num_vertices,
                vertexFormat: tri.vertex_format.to_raw(),
                vertexStrideInBytes: tri.vertex_stride,
                indexBuffer: tri.index_buffer,
                numIndexTriplets: tri.num_index_triplets,
                indexFormat: tri.index_format.to_raw(),
                indexStrideInBytes: tri.index_stride,
                flags: raw_flags[flags_start..].as_ptr(),
                numSbtRecords: tri.num_sbt_records,
                primitiveIndexOffset: tri.primitive_index_offset,
                ..Default::default()
            };
            raw.__bindgen_anon_1 = optix_sys::OptixBuildInput__bindgen_ty_1 { triangleArray: ta };
        }
        BuildInput::CustomPrimitives(cp) => {
            raw.type_ = optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            let flags_start = raw_flags.len();
            raw_flags.extend(cp.flags.iter().map(|f| f.bits()));

            let cpa = optix_sys::OptixBuildInputCustomPrimitiveArray {
                aabbBuffers: cp.aabb_buffers.as_ptr(),
                numPrimitives: cp.num_primitives,
                strideInBytes: cp.stride,
                flags: raw_flags[flags_start..].as_ptr(),
                numSbtRecords: cp.num_sbt_records,
                primitiveIndexOffset: cp.primitive_index_offset,
                ..Default::default()
            };
            raw.__bindgen_anon_1 = optix_sys::OptixBuildInput__bindgen_ty_1 {
                customPrimitiveArray: cpa,
            };
        }
        BuildInput::Instances(inst) => {
            raw.type_ = optix_sys::OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            let ia = optix_sys::OptixBuildInputInstanceArray {
                instances: inst.instances,
                numInstances: inst.num_instances,
                ..Default::default()
            };
            raw.__bindgen_anon_1 = optix_sys::OptixBuildInput__bindgen_ty_1 { instanceArray: ia };
        }
    }

    raw
}

/// Compute memory requirements for an acceleration structure build.
pub fn accel_compute_memory_usage(
    ctx: &DeviceContext,
    options: &AccelBuildOptions,
    build_inputs: &[BuildInput],
) -> Result<AccelBufferSizes> {
    let raw_options = optix_sys::OptixAccelBuildOptions {
        buildFlags: options.build_flags.bits(),
        operation: options.operation.to_raw(),
        ..Default::default()
    };

    let mut raw_flags = Vec::new();
    let raw_inputs: Vec<_> = build_inputs
        .iter()
        .map(|bi| build_input_to_raw(bi, &mut raw_flags))
        .collect();

    let mut sizes = optix_sys::OptixAccelBufferSizes::default();
    let result = unsafe {
        (ctx.table.raw.optixAccelComputeMemoryUsage.unwrap())(
            ctx.raw,
            &raw_options,
            raw_inputs.as_ptr(),
            raw_inputs.len() as u32,
            &mut sizes,
        )
    };
    error::check(result)?;

    Ok(AccelBufferSizes {
        output_size: sizes.outputSizeInBytes,
        temp_size: sizes.tempSizeInBytes,
        temp_update_size: sizes.tempUpdateSizeInBytes,
    })
}

/// Build an acceleration structure. Returns the traversable handle.
pub fn accel_build(
    ctx: &DeviceContext,
    stream: optix_sys::CUstream,
    options: &AccelBuildOptions,
    build_inputs: &[BuildInput],
    temp_buffer: optix_sys::CUdeviceptr,
    temp_buffer_size: usize,
    output_buffer: optix_sys::CUdeviceptr,
    output_buffer_size: usize,
) -> Result<optix_sys::OptixTraversableHandle> {
    let raw_options = optix_sys::OptixAccelBuildOptions {
        buildFlags: options.build_flags.bits(),
        operation: options.operation.to_raw(),
        ..Default::default()
    };

    let mut raw_flags = Vec::new();
    let raw_inputs: Vec<_> = build_inputs
        .iter()
        .map(|bi| build_input_to_raw(bi, &mut raw_flags))
        .collect();

    let mut handle: optix_sys::OptixTraversableHandle = 0;
    let result = unsafe {
        (ctx.table.raw.optixAccelBuild.unwrap())(
            ctx.raw,
            stream,
            &raw_options,
            raw_inputs.as_ptr(),
            raw_inputs.len() as u32,
            temp_buffer,
            temp_buffer_size,
            output_buffer,
            output_buffer_size,
            &mut handle,
            ptr::null(),
            0,
        )
    };
    error::check(result)?;

    Ok(handle)
}

/// Compact an acceleration structure.
pub fn accel_compact(
    ctx: &DeviceContext,
    stream: optix_sys::CUstream,
    input_handle: optix_sys::OptixTraversableHandle,
    output_buffer: optix_sys::CUdeviceptr,
    output_buffer_size: usize,
) -> Result<optix_sys::OptixTraversableHandle> {
    let mut handle: optix_sys::OptixTraversableHandle = 0;
    let result = unsafe {
        (ctx.table.raw.optixAccelCompact.unwrap())(
            ctx.raw,
            stream,
            input_handle,
            output_buffer,
            output_buffer_size,
            &mut handle,
        )
    };
    error::check(result)?;
    Ok(handle)
}
