#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod init;

// Generated FFI bindings from OptiX headers
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub use init::{optix_init, optix_init_with_handle, optix_uninit_with_handle};

// OPTIX_SBT_RECORD_HEADER_SIZE uses size_t in C, but bindgen emits it as u32.
// Re-export as usize for ergonomic Rust usage.
pub const OPTIX_SBT_RECORD_HEADER_SIZE: usize = 32;
