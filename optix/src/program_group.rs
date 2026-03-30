use crate::context::DeviceContext;
use crate::error::{self, Result, WithLog};
use crate::module::Module;
use crate::sbt::SbtRecordHeader;
use crate::sys::FunctionTable;
use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::Arc;

/// RAII wrapper around an OptiX program group.
pub struct ProgramGroup {
    pub(crate) raw: optix_sys::OptixProgramGroup,
    pub(crate) table: Arc<FunctionTable>,
}

unsafe impl Send for ProgramGroup {}
unsafe impl Sync for ProgramGroup {}

impl ProgramGroup {
    /// Create a raygen program group.
    pub fn raygen(
        ctx: &DeviceContext,
        module: &Module,
        entry_function: &str,
    ) -> Result<WithLog<Self>> {
        let entry =
            CString::new(entry_function).map_err(|_| crate::error::OptixError::InvalidValue)?;
        let desc = optix_sys::OptixProgramGroupDesc {
            kind: optix_sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            __bindgen_anon_1: optix_sys::OptixProgramGroupDesc__bindgen_ty_1 {
                raygen: optix_sys::OptixProgramGroupSingleModule {
                    module: module.raw,
                    entryFunctionName: entry.as_ptr(),
                },
            },
            flags: 0,
        };
        Self::create_single(ctx, &desc)
    }

    /// Create a miss program group.
    pub fn miss(
        ctx: &DeviceContext,
        module: &Module,
        entry_function: &str,
    ) -> Result<WithLog<Self>> {
        let entry =
            CString::new(entry_function).map_err(|_| crate::error::OptixError::InvalidValue)?;
        let desc = optix_sys::OptixProgramGroupDesc {
            kind: optix_sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
            __bindgen_anon_1: optix_sys::OptixProgramGroupDesc__bindgen_ty_1 {
                miss: optix_sys::OptixProgramGroupSingleModule {
                    module: module.raw,
                    entryFunctionName: entry.as_ptr(),
                },
            },
            flags: 0,
        };
        Self::create_single(ctx, &desc)
    }

    /// Create an exception program group.
    pub fn exception(
        ctx: &DeviceContext,
        module: &Module,
        entry_function: &str,
    ) -> Result<WithLog<Self>> {
        let entry =
            CString::new(entry_function).map_err(|_| crate::error::OptixError::InvalidValue)?;
        let desc = optix_sys::OptixProgramGroupDesc {
            kind: optix_sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
            __bindgen_anon_1: optix_sys::OptixProgramGroupDesc__bindgen_ty_1 {
                exception: optix_sys::OptixProgramGroupSingleModule {
                    module: module.raw,
                    entryFunctionName: entry.as_ptr(),
                },
            },
            flags: 0,
        };
        Self::create_single(ctx, &desc)
    }

    /// Start building a hitgroup program group.
    pub fn hitgroup(ctx: &DeviceContext) -> HitgroupBuilder<'_> {
        HitgroupBuilder {
            ctx,
            closest_hit: None,
            any_hit: None,
            intersection: None,
        }
    }

    /// Start building a callables program group.
    pub fn callables(ctx: &DeviceContext) -> CallablesBuilder<'_> {
        CallablesBuilder {
            ctx,
            direct_callable: None,
            continuation_callable: None,
        }
    }

    /// Pack the SBT record header for this program group.
    pub fn pack_header(&self) -> Result<SbtRecordHeader> {
        let mut header = SbtRecordHeader::default();
        let result = unsafe {
            (self.table.raw.optixSbtRecordPackHeader.unwrap())(
                self.raw,
                header.data.as_mut_ptr() as *mut c_void,
            )
        };
        crate::error::check(result)?;
        Ok(header)
    }

    fn create_single(
        ctx: &DeviceContext,
        desc: &optix_sys::OptixProgramGroupDesc,
    ) -> Result<WithLog<Self>> {
        let options = optix_sys::OptixProgramGroupOptions {
            payloadType: ptr::null(),
        };

        let mut raw: optix_sys::OptixProgramGroup = ptr::null_mut();
        let mut log = [0u8; 2048];
        let mut log_size = log.len();

        let result = unsafe {
            (ctx.table.raw.optixProgramGroupCreate.unwrap())(
                ctx.raw,
                desc,
                1,
                &options,
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
}

impl Drop for ProgramGroup {
    fn drop(&mut self) {
        unsafe {
            (self.table.raw.optixProgramGroupDestroy.unwrap())(self.raw);
        }
    }
}

/// Builder for hitgroup program groups.
pub struct HitgroupBuilder<'a> {
    ctx: &'a DeviceContext,
    closest_hit: Option<(&'a Module, String)>,
    any_hit: Option<(&'a Module, String)>,
    intersection: Option<(&'a Module, String)>,
}

impl<'a> HitgroupBuilder<'a> {
    pub fn closest_hit(mut self, module: &'a Module, entry: &str) -> Self {
        self.closest_hit = Some((module, entry.to_string()));
        self
    }

    pub fn any_hit(mut self, module: &'a Module, entry: &str) -> Self {
        self.any_hit = Some((module, entry.to_string()));
        self
    }

    pub fn intersection(mut self, module: &'a Module, entry: &str) -> Self {
        self.intersection = Some((module, entry.to_string()));
        self
    }

    pub fn build(self) -> Result<WithLog<ProgramGroup>> {
        let ch_entry = self
            .closest_hit
            .as_ref()
            .map(|(_, e)| CString::new(e.as_str()).unwrap());
        let ah_entry = self
            .any_hit
            .as_ref()
            .map(|(_, e)| CString::new(e.as_str()).unwrap());
        let is_entry = self
            .intersection
            .as_ref()
            .filter(|(_, e)| !e.is_empty())
            .map(|(_, e)| CString::new(e.as_str()).unwrap());

        let hitgroup = optix_sys::OptixProgramGroupHitgroup {
            moduleCH: self
                .closest_hit
                .as_ref()
                .map_or(ptr::null_mut(), |(m, _)| m.raw),
            entryFunctionNameCH: ch_entry.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            moduleAH: self
                .any_hit
                .as_ref()
                .map_or(ptr::null_mut(), |(m, _)| m.raw),
            entryFunctionNameAH: ah_entry.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            moduleIS: self
                .intersection
                .as_ref()
                .map_or(ptr::null_mut(), |(m, _)| m.raw),
            entryFunctionNameIS: is_entry.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        };

        let desc = optix_sys::OptixProgramGroupDesc {
            kind: optix_sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            __bindgen_anon_1: optix_sys::OptixProgramGroupDesc__bindgen_ty_1 { hitgroup },
            flags: 0,
        };

        ProgramGroup::create_single(self.ctx, &desc)
    }
}

/// Builder for callables program groups.
pub struct CallablesBuilder<'a> {
    ctx: &'a DeviceContext,
    direct_callable: Option<(&'a Module, String)>,
    continuation_callable: Option<(&'a Module, String)>,
}

impl<'a> CallablesBuilder<'a> {
    pub fn direct_callable(mut self, module: &'a Module, entry: &str) -> Self {
        self.direct_callable = Some((module, entry.to_string()));
        self
    }

    pub fn continuation_callable(mut self, module: &'a Module, entry: &str) -> Self {
        self.continuation_callable = Some((module, entry.to_string()));
        self
    }

    pub fn build(self) -> Result<WithLog<ProgramGroup>> {
        let dc_entry = self
            .direct_callable
            .as_ref()
            .map(|(_, e)| CString::new(e.as_str()).unwrap());
        let cc_entry = self
            .continuation_callable
            .as_ref()
            .map(|(_, e)| CString::new(e.as_str()).unwrap());

        let callables = optix_sys::OptixProgramGroupCallables {
            moduleDC: self
                .direct_callable
                .as_ref()
                .map_or(ptr::null_mut(), |(m, _)| m.raw),
            entryFunctionNameDC: dc_entry.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            moduleCC: self
                .continuation_callable
                .as_ref()
                .map_or(ptr::null_mut(), |(m, _)| m.raw),
            entryFunctionNameCC: cc_entry.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        };

        let desc = optix_sys::OptixProgramGroupDesc {
            kind: optix_sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
            __bindgen_anon_1: optix_sys::OptixProgramGroupDesc__bindgen_ty_1 { callables },
            flags: 0,
        };

        ProgramGroup::create_single(self.ctx, &desc)
    }
}
