use crate::context::DeviceContext;
use crate::error::{self, Result, WithLog};
use crate::pipeline::PipelineCompileOptions;
use crate::sys::FunctionTable;
use crate::types::{CompileDebugLevel, CompileOptimizationLevel};
use std::ptr;
use std::sync::Arc;

/// Options for module compilation.
#[derive(Debug, Clone)]
pub struct ModuleCompileOptions {
    pub max_register_count: i32,
    pub opt_level: CompileOptimizationLevel,
    pub debug_level: CompileDebugLevel,
}

impl Default for ModuleCompileOptions {
    fn default() -> Self {
        Self {
            max_register_count: 0,
            opt_level: CompileOptimizationLevel::Default,
            debug_level: CompileDebugLevel::None,
        }
    }
}

impl ModuleCompileOptions {
    pub(crate) fn to_raw(&self) -> optix_sys::OptixModuleCompileOptions {
        optix_sys::OptixModuleCompileOptions {
            maxRegisterCount: self.max_register_count,
            optLevel: self.opt_level.to_raw(),
            debugLevel: self.debug_level.to_raw(),
            boundValues: ptr::null(),
            numBoundValues: 0,
            numPayloadTypes: 0,
            payloadTypes: ptr::null(),
        }
    }
}

/// RAII wrapper around an OptiX module.
///
/// Created from PTX or OptiX IR source code.
pub struct Module {
    pub(crate) raw: optix_sys::OptixModule,
    pub(crate) table: Arc<FunctionTable>,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    /// Create a module from PTX or OptiX IR source.
    pub fn new(
        ctx: &DeviceContext,
        module_options: &ModuleCompileOptions,
        pipeline_options: &PipelineCompileOptions,
        source: &[u8],
    ) -> Result<WithLog<Self>> {
        let raw_module_opts = module_options.to_raw();
        let raw_pipeline_opts = pipeline_options.to_raw();

        let mut raw: optix_sys::OptixModule = ptr::null_mut();
        let mut log = [0u8; 2048];
        let mut log_size = log.len();

        let result = unsafe {
            (ctx.table.raw.optixModuleCreate.unwrap())(
                ctx.raw,
                &raw_module_opts,
                &raw_pipeline_opts,
                source.as_ptr() as *const i8,
                source.len(),
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

    /// Get a built-in intersection module for a primitive type (e.g., spheres, curves).
    pub fn builtin_is(
        ctx: &DeviceContext,
        module_options: &ModuleCompileOptions,
        pipeline_options: &PipelineCompileOptions,
        primitive_type: crate::types::PrimitiveType,
    ) -> Result<Self> {
        let raw_module_opts = module_options.to_raw();
        let raw_pipeline_opts = pipeline_options.to_raw();

        let builtin_opts = optix_sys::OptixBuiltinISOptions {
            builtinISModuleType: primitive_type.to_raw(),
            usesMotionBlur: 0,
            buildFlags: 0,
            curveEndcapFlags: 0,
        };

        let mut raw: optix_sys::OptixModule = ptr::null_mut();
        let result = unsafe {
            (ctx.table.raw.optixBuiltinISModuleGet.unwrap())(
                ctx.raw,
                &raw_module_opts,
                &raw_pipeline_opts,
                &builtin_opts,
                &mut raw,
            )
        };
        error::check(result)?;

        Ok(Self {
            raw,
            table: ctx.table.clone(),
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            (self.table.raw.optixModuleDestroy.unwrap())(self.raw);
        }
    }
}
