/// A parsed PBRT v4 scene.
#[derive(Debug, Clone)]
pub struct Scene {
    pub directives: Vec<Directive>,
}

/// A single directive in the scene file.
#[derive(Debug, Clone)]
pub enum Directive {
    // Scene-wide (pre-WorldBegin)
    Camera { ty: String, params: ParamList },
    Film { ty: String, params: ParamList },
    Sampler { ty: String, params: ParamList },
    Integrator { ty: String, params: ParamList },
    PixelFilter { ty: String, params: ParamList },
    Accelerator { ty: String, params: ParamList },
    ColorSpace(String),
    Option { name: String, value: ParamValue },

    // Transforms
    Identity,
    Translate { v: [f64; 3] },
    Scale { v: [f64; 3] },
    Rotate { angle: f64, axis: [f64; 3] },
    LookAt { eye: [f64; 3], look: [f64; 3], up: [f64; 3] },
    Transform { m: [f64; 16] },
    ConcatTransform { m: [f64; 16] },
    CoordinateSystem(String),
    CoordSysTransform(String),
    TransformTimes { start: f64, end: f64 },
    ActiveTransform(ActiveTransformType),
    ReverseOrientation,

    // World block
    WorldBegin,
    AttributeBegin,
    AttributeEnd,
    Attribute { target: String, params: ParamList },

    // Shapes, materials, lights
    Shape { ty: String, params: ParamList },
    Material { ty: String, params: ParamList },
    MakeNamedMaterial { name: String, params: ParamList },
    NamedMaterial(String),
    Texture { name: String, ty: String, class: String, params: ParamList },
    LightSource { ty: String, params: ParamList },
    AreaLightSource { ty: String, params: ParamList },

    // Media
    MakeNamedMedium { name: String, params: ParamList },
    MediumInterface { exterior: String, interior: String },

    // Instancing
    ObjectBegin(String),
    ObjectEnd,
    ObjectInstance(String),

    // File inclusion
    Include(String),
    Import(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTransformType {
    StartTime,
    EndTime,
    All,
}

pub type ParamList = Vec<Param>;

/// A single named, typed parameter.
#[derive(Debug, Clone)]
pub struct Param {
    pub ty: ParamType,
    pub name: String,
    pub value: ParamValue,
}

/// Parameter type tag parsed from `"type name"` strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Integer,
    Float,
    Point2,
    Vector2,
    Point3,
    Vector3,
    Normal3,
    Bool,
    String,
    Rgb,
    Spectrum,
    Blackbody,
    Texture,
}

/// The value(s) of a parameter.
#[derive(Debug, Clone)]
pub enum ParamValue {
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
    Bools(Vec<bool>),
}
