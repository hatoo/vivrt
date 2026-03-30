use crate::{OptixFunctionTable, OptixResult};
use std::mem;
use std::ptr;

const ABI_VERSION: u32 = crate::OPTIX_ABI_VERSION;

/// Type signature for `optixQueryFunctionTable` exported by the OptiX library.
type OptixQueryFunctionTableFn = unsafe extern "C" fn(
    abiId: u32,
    numOptions: u32,
    optionKeys: *const u32,
    optionValues: *const u32,
    functionTable: *mut ::std::os::raw::c_void,
    sizeOfTable: usize,
) -> OptixResult;

/// Initializes OptiX by dynamically loading the library and populating a function table.
///
/// Returns the populated `OptixFunctionTable` on success, or an `OptixResult` error code.
/// The library handle is intentionally leaked so that the library stays loaded for the
/// lifetime of the process.
pub fn optix_init() -> Result<OptixFunctionTable, OptixResult> {
    let (_handle, table) = optix_init_with_handle()?;
    mem::forget(_handle);
    Ok(table)
}

/// Initializes OptiX and returns both the library handle and function table.
///
/// The caller is responsible for keeping the `libloading::Library` alive for the
/// duration of OptiX usage. Drop it (or call [`optix_uninit_with_handle`]) to unload.
pub fn optix_init_with_handle() -> Result<(libloading::Library, OptixFunctionTable), OptixResult> {
    let lib = load_optix_library()?;

    let query_fn: OptixQueryFunctionTableFn = unsafe {
        let sym = lib
            .get::<OptixQueryFunctionTableFn>(b"optixQueryFunctionTable\0")
            .map_err(|_| OptixResult::OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND)?;
        *sym
    };

    let mut table: OptixFunctionTable = unsafe { mem::zeroed() };
    let result = unsafe {
        query_fn(
            ABI_VERSION,
            0,
            ptr::null(),
            ptr::null(),
            &mut table as *mut OptixFunctionTable as *mut ::std::os::raw::c_void,
            mem::size_of::<OptixFunctionTable>(),
        )
    };

    if result != OptixResult::OPTIX_SUCCESS {
        return Err(result);
    }

    Ok((lib, table))
}

/// Unloads the OptiX library and zeros the function table.
///
/// All `OptixDeviceContext` objects must be destroyed before calling this.
pub fn optix_uninit_with_handle(
    handle: libloading::Library,
    table: &mut OptixFunctionTable,
) -> Result<(), OptixResult> {
    *table = unsafe { mem::zeroed() };
    drop(handle);
    Ok(())
}

// ---------------------------------------------------------------------------
// Platform-specific library loading
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn load_optix_library() -> Result<libloading::Library, OptixResult> {
    use std::path::PathBuf;

    const DLL_NAME: &str = "nvoptix.dll";

    // Try 1: System directory (GetSystemDirectoryA + nvoptix.dll)
    {
        let mut buf = [0u8; 260];
        let len = unsafe { GetSystemDirectoryA(buf.as_mut_ptr(), buf.len() as u32) };
        if len > 0 {
            if let Ok(system_dir) = std::str::from_utf8(&buf[..len as usize]) {
                let path = PathBuf::from(system_dir).join(DLL_NAME);
                if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
                    return Ok(lib);
                }
            }
        }
    }

    // Try 2: Scan GPU registry entries for OpenGLDriverName, look for nvoptix.dll
    // next to the OpenGL driver. Matches the fallback in optix_stubs.h.
    {
        let display_class =
            "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\0";

        let mut hkey: HKEY = ptr::null_mut();
        let status = unsafe {
            RegOpenKeyExA(
                HKEY_LOCAL_MACHINE,
                display_class.as_ptr(),
                0,
                KEY_READ,
                &mut hkey,
            )
        };

        if status == 0 {
            let mut index: u32 = 0;
            loop {
                let mut subkey_name = [0u8; 256];
                let mut subkey_len = subkey_name.len() as u32;
                let status = unsafe {
                    RegEnumKeyExA(
                        hkey,
                        index,
                        subkey_name.as_mut_ptr(),
                        &mut subkey_len,
                        ptr::null_mut(),
                        ptr::null_mut(),
                        ptr::null_mut(),
                        ptr::null_mut(),
                    )
                };
                if status != 0 {
                    break;
                }
                index += 1;

                // Open the subkey
                let mut sub_hkey: HKEY = ptr::null_mut();
                let status = unsafe {
                    RegOpenKeyExA(hkey, subkey_name.as_ptr(), 0, KEY_READ, &mut sub_hkey)
                };
                if status != 0 {
                    continue;
                }

                // Read OpenGLDriverName value
                let value_name = b"OpenGLDriverName\0";
                let mut data = [0u8; 1024];
                let mut data_len = data.len() as u32;
                let mut value_type: u32 = 0;
                let status = unsafe {
                    RegQueryValueExA(
                        sub_hkey,
                        value_name.as_ptr(),
                        ptr::null_mut(),
                        &mut value_type,
                        data.as_mut_ptr(),
                        &mut data_len,
                    )
                };
                unsafe { RegCloseKey(sub_hkey) };

                if status != 0 || data_len == 0 {
                    continue;
                }

                // Extract the directory from the driver path
                let driver_str = std::str::from_utf8(&data[..data_len as usize])
                    .unwrap_or("")
                    .trim_end_matches('\0');
                let driver_str = driver_str.split('\0').next().unwrap_or(driver_str);

                if let Some(dir) = PathBuf::from(driver_str).parent() {
                    let candidate = dir.join(DLL_NAME);
                    if let Ok(lib) = unsafe { libloading::Library::new(&candidate) } {
                        unsafe { RegCloseKey(hkey) };
                        return Ok(lib);
                    }
                }
            }
            unsafe { RegCloseKey(hkey) };
        }
    }

    Err(OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND)
}

#[cfg(target_os = "linux")]
fn load_optix_library() -> Result<libloading::Library, OptixResult> {
    unsafe {
        libloading::Library::new("libnvoptix.so.1")
            .map_err(|_| OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND)
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn load_optix_library() -> Result<libloading::Library, OptixResult> {
    Err(OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND)
}

// ---------------------------------------------------------------------------
// Minimal Win32 FFI declarations (avoids windows-sys dependency)
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
type HKEY = *mut ::std::os::raw::c_void;

#[cfg(target_os = "windows")]
const HKEY_LOCAL_MACHINE: HKEY = 0x80000002u32 as usize as HKEY;

#[cfg(target_os = "windows")]
const KEY_READ: u32 = 0x20019;

#[cfg(target_os = "windows")]
#[link(name = "advapi32")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetSystemDirectoryA(lpBuffer: *mut u8, uSize: u32) -> u32;
    fn RegOpenKeyExA(
        hKey: HKEY,
        lpSubKey: *const u8,
        ulOptions: u32,
        samDesired: u32,
        phkResult: *mut HKEY,
    ) -> i32;
    fn RegEnumKeyExA(
        hKey: HKEY,
        dwIndex: u32,
        lpName: *mut u8,
        lpcchName: *mut u32,
        lpReserved: *mut u32,
        lpClass: *mut u8,
        lpcchClass: *mut u32,
        lpftLastWriteTime: *mut u64,
    ) -> i32;
    fn RegQueryValueExA(
        hKey: HKEY,
        lpValueName: *const u8,
        lpReserved: *mut u32,
        lpType: *mut u32,
        lpData: *mut u8,
        lpcbData: *mut u32,
    ) -> i32;
    fn RegCloseKey(hKey: HKEY) -> i32;
}
