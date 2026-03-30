use optix_sys;

// --- Simple enums ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompileOptimizationLevel {
    #[default]
    Default,
    Level0,
    Level1,
    Level2,
    Level3,
}

impl CompileOptimizationLevel {
    pub(crate) fn to_raw(self) -> optix_sys::OptixCompileOptimizationLevel {
        use optix_sys::OptixCompileOptimizationLevel as R;
        match self {
            Self::Default => R::OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            Self::Level0 => R::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
            Self::Level1 => R::OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
            Self::Level2 => R::OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
            Self::Level3 => R::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompileDebugLevel {
    #[default]
    Default,
    None,
    Minimal,
    Moderate,
    Full,
}

impl CompileDebugLevel {
    pub(crate) fn to_raw(self) -> optix_sys::OptixCompileDebugLevel {
        use optix_sys::OptixCompileDebugLevel as R;
        match self {
            Self::Default => R::OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
            Self::None => R::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            Self::Minimal => R::OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL,
            Self::Moderate => R::OPTIX_COMPILE_DEBUG_LEVEL_MODERATE,
            Self::Full => R::OPTIX_COMPILE_DEBUG_LEVEL_FULL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    #[default]
    Off,
    All,
}

impl ValidationMode {
    pub(crate) fn to_raw(self) -> optix_sys::OptixDeviceContextValidationMode {
        use optix_sys::OptixDeviceContextValidationMode as R;
        match self {
            Self::Off => R::OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF,
            Self::All => R::OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    Float3,
    Float2,
    Half3,
    Half2,
    Snorm16_3,
    Snorm16_2,
}

impl VertexFormat {
    pub(crate) fn to_raw(self) -> optix_sys::OptixVertexFormat {
        use optix_sys::OptixVertexFormat as R;
        match self {
            Self::Float3 => R::OPTIX_VERTEX_FORMAT_FLOAT3,
            Self::Float2 => R::OPTIX_VERTEX_FORMAT_FLOAT2,
            Self::Half3 => R::OPTIX_VERTEX_FORMAT_HALF3,
            Self::Half2 => R::OPTIX_VERTEX_FORMAT_HALF2,
            Self::Snorm16_3 => R::OPTIX_VERTEX_FORMAT_SNORM16_3,
            Self::Snorm16_2 => R::OPTIX_VERTEX_FORMAT_SNORM16_2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndicesFormat {
    #[default]
    None,
    UnsignedByte3,
    UnsignedShort3,
    UnsignedInt3,
}

impl IndicesFormat {
    pub(crate) fn to_raw(self) -> optix_sys::OptixIndicesFormat {
        use optix_sys::OptixIndicesFormat as R;
        match self {
            Self::None => R::OPTIX_INDICES_FORMAT_NONE,
            Self::UnsignedByte3 => R::OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3,
            Self::UnsignedShort3 => R::OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3,
            Self::UnsignedInt3 => R::OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildOperation {
    Build,
    Update,
}

impl BuildOperation {
    pub(crate) fn to_raw(self) -> optix_sys::OptixBuildOperation {
        use optix_sys::OptixBuildOperation as R;
        match self {
            Self::Build => R::OPTIX_BUILD_OPERATION_BUILD,
            Self::Update => R::OPTIX_BUILD_OPERATION_UPDATE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceProperty {
    LimitMaxTraceDepth,
    LimitMaxTraversableGraphDepth,
    LimitMaxPrimitivesPerGas,
    LimitMaxInstancesPerIas,
    RtcoreVersion,
    LimitMaxInstanceId,
    LimitNumBitsInstanceVisibilityMask,
    LimitMaxSbtRecordsPerGas,
    LimitMaxSbtOffset,
    ShaderExecutionReordering,
}

impl DeviceProperty {
    pub(crate) fn to_raw(self) -> optix_sys::OptixDeviceProperty {
        use optix_sys::OptixDeviceProperty as R;
        match self {
            Self::LimitMaxTraceDepth => R::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,
            Self::LimitMaxTraversableGraphDepth => {
                R::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH
            }
            Self::LimitMaxPrimitivesPerGas => R::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
            Self::LimitMaxInstancesPerIas => R::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
            Self::RtcoreVersion => R::OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
            Self::LimitMaxInstanceId => R::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
            Self::LimitNumBitsInstanceVisibilityMask => {
                R::OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK
            }
            Self::LimitMaxSbtRecordsPerGas => {
                R::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS
            }
            Self::LimitMaxSbtOffset => R::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET,
            Self::ShaderExecutionReordering => R::OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiserModelKind {
    Aov,
    TemporalAov,
    Upscale2X,
    TemporalUpscale2X,
}

impl DenoiserModelKind {
    pub(crate) fn to_raw(self) -> optix_sys::OptixDenoiserModelKind {
        use optix_sys::OptixDenoiserModelKind as R;
        match self {
            Self::Aov => R::OPTIX_DENOISER_MODEL_KIND_AOV,
            Self::TemporalAov => R::OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV,
            Self::Upscale2X => R::OPTIX_DENOISER_MODEL_KIND_UPSCALE2X,
            Self::TemporalUpscale2X => R::OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenoiserAlphaMode {
    #[default]
    Copy,
    Denoise,
}

impl DenoiserAlphaMode {
    pub(crate) fn to_raw(self) -> optix_sys::OptixDenoiserAlphaMode {
        use optix_sys::OptixDenoiserAlphaMode as R;
        match self {
            Self::Copy => R::OPTIX_DENOISER_ALPHA_MODE_COPY,
            Self::Denoise => R::OPTIX_DENOISER_ALPHA_MODE_DENOISE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Half1,
    Half2,
    Half3,
    Half4,
    Float1,
    Float2,
    Float3,
    Float4,
    Uchar3,
    Uchar4,
}

impl PixelFormat {
    pub(crate) fn to_raw(self) -> optix_sys::OptixPixelFormat {
        use optix_sys::OptixPixelFormat as R;
        match self {
            Self::Half1 => R::OPTIX_PIXEL_FORMAT_HALF1,
            Self::Half2 => R::OPTIX_PIXEL_FORMAT_HALF2,
            Self::Half3 => R::OPTIX_PIXEL_FORMAT_HALF3,
            Self::Half4 => R::OPTIX_PIXEL_FORMAT_HALF4,
            Self::Float1 => R::OPTIX_PIXEL_FORMAT_FLOAT1,
            Self::Float2 => R::OPTIX_PIXEL_FORMAT_FLOAT2,
            Self::Float3 => R::OPTIX_PIXEL_FORMAT_FLOAT3,
            Self::Float4 => R::OPTIX_PIXEL_FORMAT_FLOAT4,
            Self::Uchar3 => R::OPTIX_PIXEL_FORMAT_UCHAR3,
            Self::Uchar4 => R::OPTIX_PIXEL_FORMAT_UCHAR4,
        }
    }
}

// --- Bitflags ---

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BuildFlags: u32 {
        const NONE = 0;
        const ALLOW_UPDATE = 1 << 0;
        const ALLOW_COMPACTION = 1 << 1;
        const PREFER_FAST_TRACE = 1 << 2;
        const PREFER_FAST_BUILD = 1 << 3;
        const ALLOW_RANDOM_VERTEX_ACCESS = 1 << 4;
        const ALLOW_RANDOM_INSTANCE_ACCESS = 1 << 5;
        const ALLOW_OPACITY_MICROMAP_UPDATE = 1 << 6;
        const ALLOW_DISABLE_OPACITY_MICROMAPS = 1 << 7;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct GeometryFlags: u32 {
        const NONE = 0;
        const DISABLE_ANYHIT = 1 << 0;
        const REQUIRE_SINGLE_ANYHIT_CALL = 1 << 1;
        const DISABLE_TRIANGLE_FACE_CULLING = 1 << 2;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct InstanceFlags: u32 {
        const NONE = 0;
        const DISABLE_TRIANGLE_FACE_CULLING = 1 << 0;
        const FLIP_TRIANGLE_FACING = 1 << 1;
        const DISABLE_ANYHIT = 1 << 2;
        const ENFORCE_ANYHIT = 1 << 3;
        const FORCE_OPACITY_MICROMAP_2_STATE = 1 << 4;
        const DISABLE_OPACITY_MICROMAPS = 1 << 5;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ExceptionFlags: u32 {
        const NONE = 0;
        const STACK_OVERFLOW = 1 << 0;
        const TRACE_DEPTH = 1 << 1;
        const USER = 1 << 2;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TraversableGraphFlags: u32 {
        const ALLOW_ANY = 0;
        const ALLOW_SINGLE_GAS = 1 << 0;
        const ALLOW_SINGLE_LEVEL_INSTANCING = 1 << 1;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PrimitiveTypeFlags: u32 {
        const CUSTOM = 1 << 0;
        const ROUND_QUADRATIC_BSPLINE = 1 << 1;
        const ROUND_CUBIC_BSPLINE = 1 << 2;
        const ROUND_LINEAR = 1 << 3;
        const ROUND_CATMULLROM = 1 << 4;
        const FLAT_QUADRATIC_BSPLINE = 1 << 5;
        const SPHERE = 1 << 6;
        const ROUND_CUBIC_BEZIER = 1 << 7;
        const TRIANGLE = 1 << 31;
    }
}
