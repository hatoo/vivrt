use crate::context::DeviceContext;
use crate::error::{self, Result, WithLog};
use crate::program_group::ProgramGroup;
use crate::sys::FunctionTable;
use crate::types::{ExceptionFlags, PrimitiveTypeFlags, TraversableGraphFlags};
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

/// Options for pipeline compilation.
///
/// This struct owns the launch params variable name string.
pub struct PipelineCompileOptions {
    pub uses_motion_blur: bool,
    pub traversable_graph_flags: TraversableGraphFlags,
    pub num_payload_values: i32,
    pub num_attribute_values: i32,
    pub exception_flags: ExceptionFlags,
    pub uses_primitive_type_flags: PrimitiveTypeFlags,
    pub allow_opacity_micromaps: bool,
    launch_params_name: CString,
}

impl PipelineCompileOptions {
    /// Create pipeline compile options with the given launch params variable name.
    ///
    /// The name must match the `extern "C" { __constant__ Params params; }` declaration
    /// in your device code.
    pub fn new(launch_params_name: &str) -> Self {
        Self {
            uses_motion_blur: false,
            traversable_graph_flags: TraversableGraphFlags::ALLOW_SINGLE_GAS,
            num_payload_values: 3,
            num_attribute_values: 3,
            exception_flags: ExceptionFlags::NONE,
            uses_primitive_type_flags: PrimitiveTypeFlags::TRIANGLE,
            allow_opacity_micromaps: false,
            launch_params_name: CString::new(launch_params_name)
                .expect("invalid launch params name"),
        }
    }

    pub fn uses_motion_blur(mut self, v: bool) -> Self {
        self.uses_motion_blur = v;
        self
    }
    pub fn traversable_graph_flags(mut self, f: TraversableGraphFlags) -> Self {
        self.traversable_graph_flags = f;
        self
    }
    pub fn num_payload_values(mut self, n: i32) -> Self {
        self.num_payload_values = n;
        self
    }
    pub fn num_attribute_values(mut self, n: i32) -> Self {
        self.num_attribute_values = n;
        self
    }
    pub fn exception_flags(mut self, f: ExceptionFlags) -> Self {
        self.exception_flags = f;
        self
    }
    pub fn uses_primitive_type_flags(mut self, f: PrimitiveTypeFlags) -> Self {
        self.uses_primitive_type_flags = f;
        self
    }

    pub(crate) fn to_raw(&self) -> optix_sys::OptixPipelineCompileOptions {
        optix_sys::OptixPipelineCompileOptions {
            usesMotionBlur: self.uses_motion_blur as i32,
            traversableGraphFlags: self.traversable_graph_flags.bits(),
            numPayloadValues: self.num_payload_values,
            numAttributeValues: self.num_attribute_values,
            exceptionFlags: self.exception_flags.bits(),
            pipelineLaunchParamsVariableName: self.launch_params_name.as_ptr(),
            usesPrimitiveTypeFlags: self.uses_primitive_type_flags.bits(),
            allowOpacityMicromaps: self.allow_opacity_micromaps as i32,
            ..Default::default()
        }
    }
}

/// Options for pipeline linking.
#[derive(Debug, Clone)]
pub struct PipelineLinkOptions {
    pub max_trace_depth: u32,
}

impl Default for PipelineLinkOptions {
    fn default() -> Self {
        Self { max_trace_depth: 1 }
    }
}

/// RAII wrapper around an OptiX pipeline.
pub struct Pipeline {
    pub(crate) raw: optix_sys::OptixPipeline,
    pub(crate) table: Arc<FunctionTable>,
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl Pipeline {
    /// Create a new pipeline from program groups.
    pub fn new(
        ctx: &DeviceContext,
        compile_options: &PipelineCompileOptions,
        link_options: &PipelineLinkOptions,
        program_groups: &[&ProgramGroup],
    ) -> Result<WithLog<Self>> {
        let raw_compile = compile_options.to_raw();
        let raw_link = optix_sys::OptixPipelineLinkOptions {
            maxTraceDepth: link_options.max_trace_depth,
        };

        let raw_pgs: Vec<optix_sys::OptixProgramGroup> =
            program_groups.iter().map(|pg| pg.raw).collect();

        let mut raw: optix_sys::OptixPipeline = ptr::null_mut();
        let mut log = [0u8; 2048];
        let mut log_size = log.len();

        let result = unsafe {
            (ctx.table.raw.optixPipelineCreate.unwrap())(
                ctx.raw,
                &raw_compile,
                &raw_link,
                raw_pgs.as_ptr(),
                raw_pgs.len() as u32,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut raw,
            )
        };
        error::check(result)?;

        Ok(WithLog {
            value: Self {
                raw,
                table: ctx.table.clone(),
            },
            log: crate::error::extract_log(&log, log_size),
        })
    }

    /// Set the stack sizes for the pipeline.
    pub fn set_stack_size(
        &self,
        direct_callable_from_traversal: u32,
        direct_callable_from_state: u32,
        continuation: u32,
        max_traversable_graph_depth: u32,
    ) -> Result<()> {
        let result = unsafe {
            (self.table.raw.optixPipelineSetStackSize.unwrap())(
                self.raw,
                direct_callable_from_traversal,
                direct_callable_from_state,
                continuation,
                max_traversable_graph_depth,
            )
        };
        error::check(result)
    }

    /// Launch the pipeline.
    pub fn launch(
        &self,
        stream: optix_sys::CUstream,
        params_device_ptr: optix_sys::CUdeviceptr,
        params_size: usize,
        sbt: &optix_sys::OptixShaderBindingTable,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<()> {
        let result = unsafe {
            (self.table.raw.optixLaunch.unwrap())(
                self.raw,
                stream,
                params_device_ptr,
                params_size,
                sbt,
                width,
                height,
                depth,
            )
        };
        error::check(result)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            (self.table.raw.optixPipelineDestroy.unwrap())(self.raw);
        }
    }
}
