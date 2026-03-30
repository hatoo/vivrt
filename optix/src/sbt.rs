use crate::error::{OptixError, Result};
use crate::program_group::ProgramGroup;

/// The 32-byte SBT record header, packed by OptiX.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct SbtRecordHeader {
    pub(crate) data: [u8; optix_sys::OPTIX_SBT_RECORD_HEADER_SIZE],
}

impl Default for SbtRecordHeader {
    fn default() -> Self {
        Self {
            data: [0u8; optix_sys::OPTIX_SBT_RECORD_HEADER_SIZE],
        }
    }
}

/// An SBT record combining a 32-byte OptiX header with user data.
///
/// The struct is 16-byte aligned as required by OptiX.
#[repr(C, align(16))]
pub struct SbtRecord<T: Copy> {
    pub header: SbtRecordHeader,
    pub data: T,
}

impl<T: Copy> SbtRecord<T> {
    /// Create an SBT record by packing the header from a program group.
    pub fn new(program_group: &ProgramGroup, data: T) -> Result<Self> {
        let header = program_group.pack_header()?;
        Ok(Self { header, data })
    }
}

/// Builder for creating a validated `OptixShaderBindingTable`.
pub struct ShaderBindingTableBuilder {
    raygen_record: optix_sys::CUdeviceptr,
    exception_record: optix_sys::CUdeviceptr,
    miss_record_base: optix_sys::CUdeviceptr,
    miss_record_stride: u32,
    miss_record_count: u32,
    hitgroup_record_base: optix_sys::CUdeviceptr,
    hitgroup_record_stride: u32,
    hitgroup_record_count: u32,
    callables_record_base: optix_sys::CUdeviceptr,
    callables_record_stride: u32,
    callables_record_count: u32,
}

impl ShaderBindingTableBuilder {
    /// Start building an SBT with the raygen record device pointer.
    pub fn new(raygen_record: optix_sys::CUdeviceptr) -> Self {
        Self {
            raygen_record,
            exception_record: 0,
            miss_record_base: 0,
            miss_record_stride: 0,
            miss_record_count: 0,
            hitgroup_record_base: 0,
            hitgroup_record_stride: 0,
            hitgroup_record_count: 0,
            callables_record_base: 0,
            callables_record_stride: 0,
            callables_record_count: 0,
        }
    }

    pub fn exception_record(mut self, record: optix_sys::CUdeviceptr) -> Self {
        self.exception_record = record;
        self
    }

    pub fn miss_records(mut self, base: optix_sys::CUdeviceptr, stride: u32, count: u32) -> Self {
        self.miss_record_base = base;
        self.miss_record_stride = stride;
        self.miss_record_count = count;
        self
    }

    pub fn hitgroup_records(
        mut self,
        base: optix_sys::CUdeviceptr,
        stride: u32,
        count: u32,
    ) -> Self {
        self.hitgroup_record_base = base;
        self.hitgroup_record_stride = stride;
        self.hitgroup_record_count = count;
        self
    }

    pub fn callables_records(
        mut self,
        base: optix_sys::CUdeviceptr,
        stride: u32,
        count: u32,
    ) -> Self {
        self.callables_record_base = base;
        self.callables_record_stride = stride;
        self.callables_record_count = count;
        self
    }

    /// Build the SBT, validating alignment requirements.
    pub fn build(self) -> Result<optix_sys::OptixShaderBindingTable> {
        const ALIGN: u64 = optix_sys::OPTIX_SBT_RECORD_ALIGNMENT as u64;

        if self.raygen_record % ALIGN != 0 {
            return Err(OptixError::InvalidValue);
        }
        if self.miss_record_base != 0 && self.miss_record_base % ALIGN != 0 {
            return Err(OptixError::InvalidValue);
        }
        if self.hitgroup_record_base != 0 && self.hitgroup_record_base % ALIGN != 0 {
            return Err(OptixError::InvalidValue);
        }

        Ok(optix_sys::OptixShaderBindingTable {
            raygenRecord: self.raygen_record,
            exceptionRecord: self.exception_record,
            missRecordBase: self.miss_record_base,
            missRecordStrideInBytes: self.miss_record_stride,
            missRecordCount: self.miss_record_count,
            hitgroupRecordBase: self.hitgroup_record_base,
            hitgroupRecordStrideInBytes: self.hitgroup_record_stride,
            hitgroupRecordCount: self.hitgroup_record_count,
            callablesRecordBase: self.callables_record_base,
            callablesRecordStrideInBytes: self.callables_record_stride,
            callablesRecordCount: self.callables_record_count,
        })
    }
}
