use crate::error::{ParseError, Result};
use crate::lexer::{Lexer, Located, Token};
use crate::types::*;

struct Parser {
    tokens: Vec<Located<Token>>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Located<Token>>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Located<Token>> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Located<Token>> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn loc(&self) -> (usize, usize) {
        self.peek().map(|t| (t.line, t.column)).unwrap_or_else(|| {
            self.tokens
                .last()
                .map(|t| (t.line, t.column))
                .unwrap_or((1, 1))
        })
    }

    fn err(&self, msg: impl Into<String>) -> ParseError {
        let (line, col) = self.loc();
        ParseError::new(msg, line, col)
    }

    fn expect_string(&mut self) -> Result<String> {
        match self.advance() {
            Some(Located {
                value: Token::String(s),
                ..
            }) => Ok(s.clone()),
            _ => Err(self.err("expected quoted string")),
        }
    }

    fn expect_number(&mut self) -> Result<f64> {
        match self.advance() {
            Some(Located {
                value: Token::Number(n),
                ..
            }) => Ok(*n),
            _ => Err(self.err("expected number")),
        }
    }

    fn expect_identifier(&mut self) -> Result<String> {
        match self.advance() {
            Some(Located {
                value: Token::Identifier(s),
                ..
            }) => Ok(s.clone()),
            _ => Err(self.err("expected identifier")),
        }
    }

    fn read_n_numbers<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut arr = [0.0; N];
        for v in arr.iter_mut() {
            *v = self.expect_number()?;
        }
        Ok(arr)
    }

    fn parse(mut self) -> Result<Scene> {
        let mut directives = Vec::new();
        while self.peek().is_some() {
            directives.push(self.parse_directive()?);
        }
        Ok(Scene { directives })
    }

    fn parse_directive(&mut self) -> Result<Directive> {
        let ident = self.expect_identifier()?;
        match ident.as_str() {
            // No-arg
            "WorldBegin" => Ok(Directive::WorldBegin),
            "AttributeBegin" => Ok(Directive::AttributeBegin),
            "AttributeEnd" => Ok(Directive::AttributeEnd),
            "ObjectEnd" => Ok(Directive::ObjectEnd),
            "ReverseOrientation" => Ok(Directive::ReverseOrientation),
            "Identity" => Ok(Directive::Identity),

            // Fixed-arg transforms
            "Translate" => Ok(Directive::Translate {
                v: self.read_n_numbers()?,
            }),
            "Scale" => Ok(Directive::Scale {
                v: self.read_n_numbers()?,
            }),
            "Rotate" => {
                let angle = self.expect_number()?;
                let axis = self.read_n_numbers()?;
                Ok(Directive::Rotate { angle, axis })
            }
            "LookAt" => {
                let eye = self.read_n_numbers()?;
                let look = self.read_n_numbers()?;
                let up = self.read_n_numbers()?;
                Ok(Directive::LookAt { eye, look, up })
            }
            "Transform" => Ok(Directive::Transform {
                m: self.read_n_numbers()?,
            }),
            "ConcatTransform" => Ok(Directive::ConcatTransform {
                m: self.read_n_numbers()?,
            }),
            "TransformTimes" => {
                let start = self.expect_number()?;
                let end = self.expect_number()?;
                Ok(Directive::TransformTimes { start, end })
            }
            "ActiveTransform" => {
                let mode = self.expect_identifier()?;
                let ty = match mode.as_str() {
                    "StartTime" => ActiveTransformType::StartTime,
                    "EndTime" => ActiveTransformType::EndTime,
                    "All" => ActiveTransformType::All,
                    _ => return Err(self.err(format!("unknown ActiveTransform mode '{mode}'"))),
                };
                Ok(Directive::ActiveTransform(ty))
            }

            // Single string arg
            "CoordinateSystem" => Ok(Directive::CoordinateSystem(self.expect_string()?)),
            "CoordSysTransform" => Ok(Directive::CoordSysTransform(self.expect_string()?)),
            "ColorSpace" => Ok(Directive::ColorSpace(self.expect_string()?)),
            "Include" => Ok(Directive::Include(self.expect_string()?)),
            "Import" => Ok(Directive::Import(self.expect_string()?)),
            "NamedMaterial" => Ok(Directive::NamedMaterial(self.expect_string()?)),
            "ObjectBegin" => Ok(Directive::ObjectBegin(self.expect_string()?)),
            "ObjectInstance" => Ok(Directive::ObjectInstance(self.expect_string()?)),

            // Two string args
            "MediumInterface" => {
                let exterior = self.expect_string()?;
                let interior = self.expect_string()?;
                Ok(Directive::MediumInterface { exterior, interior })
            }

            // Type + param list
            "Camera" => self.parse_typed_directive(|ty, params| Directive::Camera { ty, params }),
            "Film" => self.parse_typed_directive(|ty, params| Directive::Film { ty, params }),
            "Sampler" => self.parse_typed_directive(|ty, params| Directive::Sampler { ty, params }),
            "Integrator" => {
                self.parse_typed_directive(|ty, params| Directive::Integrator { ty, params })
            }
            "PixelFilter" => {
                self.parse_typed_directive(|ty, params| Directive::PixelFilter { ty, params })
            }
            "Accelerator" => {
                self.parse_typed_directive(|ty, params| Directive::Accelerator { ty, params })
            }
            "Shape" => self.parse_typed_directive(|ty, params| Directive::Shape { ty, params }),
            "Material" => {
                self.parse_typed_directive(|ty, params| Directive::Material { ty, params })
            }
            "LightSource" => {
                self.parse_typed_directive(|ty, params| Directive::LightSource { ty, params })
            }
            "AreaLightSource" => {
                self.parse_typed_directive(|ty, params| Directive::AreaLightSource { ty, params })
            }

            // Attribute "target" param-list
            "Attribute" => {
                let target = self.expect_string()?;
                let params = self.parse_param_list()?;
                Ok(Directive::Attribute { target, params })
            }

            // Texture "name" "type" "class" param-list
            "Texture" => {
                let name = self.expect_string()?;
                let ty = self.expect_string()?;
                let class = self.expect_string()?;
                let params = self.parse_param_list()?;
                Ok(Directive::Texture {
                    name,
                    ty,
                    class,
                    params,
                })
            }

            // MakeNamedMaterial "name" param-list
            "MakeNamedMaterial" => {
                let name = self.expect_string()?;
                let params = self.parse_param_list()?;
                Ok(Directive::MakeNamedMaterial { name, params })
            }

            // MakeNamedMedium "name" param-list
            "MakeNamedMedium" => {
                let name = self.expect_string()?;
                let params = self.parse_param_list()?;
                Ok(Directive::MakeNamedMedium { name, params })
            }

            // Option "name" value
            "Option" => {
                let name = self.expect_string()?;
                let value = self.parse_single_value()?;
                Ok(Directive::Option { name, value })
            }

            _ => Err(self.err(format!("unknown directive '{ident}'"))),
        }
    }

    fn parse_typed_directive<F>(&mut self, f: F) -> Result<Directive>
    where
        F: FnOnce(String, ParamList) -> Directive,
    {
        let ty = self.expect_string()?;
        let params = self.parse_param_list()?;
        Ok(f(ty, params))
    }

    /// Parse a parameter list. Stops when the next token is an identifier (next directive) or EOF.
    fn parse_param_list(&mut self) -> Result<ParamList> {
        let mut params = Vec::new();
        while let Some(tok) = self.peek() {
            match &tok.value {
                Token::String(_) => {
                    params.push(self.parse_param()?);
                }
                Token::Identifier(_) => break, // next directive
                _ => break,
            }
        }
        Ok(params)
    }

    /// Parse a single parameter: `"type name" value-or-[values]`
    fn parse_param(&mut self) -> Result<Param> {
        let type_name = self.expect_string()?;
        let (ty, name) = parse_type_name(&type_name)
            .ok_or_else(|| self.err(format!("invalid parameter type/name '{type_name}'")))?;

        let value = self.parse_param_value(&ty)?;
        Ok(Param { ty, name, value })
    }

    /// Parse the value of a parameter, with or without brackets.
    fn parse_param_value(&mut self, ty: &ParamType) -> Result<ParamValue> {
        let bracketed = matches!(
            self.peek(),
            Some(Located {
                value: Token::LBracket,
                ..
            })
        );
        if bracketed {
            self.advance(); // consume [
        }

        let value = match ty {
            ParamType::Integer => {
                if bracketed {
                    let mut vals = Vec::new();
                    while self.is_number_next() {
                        vals.push(self.expect_number()? as i64);
                    }
                    ParamValue::Ints(vals)
                } else {
                    ParamValue::Ints(vec![self.expect_number()? as i64])
                }
            }
            ParamType::Float | ParamType::Blackbody => {
                if bracketed {
                    let mut vals = Vec::new();
                    while self.is_number_next() {
                        vals.push(self.expect_number()?);
                    }
                    ParamValue::Floats(vals)
                } else {
                    ParamValue::Floats(vec![self.expect_number()?])
                }
            }
            ParamType::Point2 | ParamType::Vector2 => {
                if bracketed {
                    let mut vals = Vec::new();
                    while self.is_number_next() {
                        vals.push(self.expect_number()?);
                    }
                    ParamValue::Floats(vals)
                } else {
                    let a = self.expect_number()?;
                    let b = self.expect_number()?;
                    ParamValue::Floats(vec![a, b])
                }
            }
            ParamType::Point3 | ParamType::Vector3 | ParamType::Normal3 | ParamType::Rgb => {
                if bracketed {
                    let mut vals = Vec::new();
                    while self.is_number_next() {
                        vals.push(self.expect_number()?);
                    }
                    ParamValue::Floats(vals)
                } else {
                    let a = self.expect_number()?;
                    let b = self.expect_number()?;
                    let c = self.expect_number()?;
                    ParamValue::Floats(vec![a, b, c])
                }
            }
            ParamType::Spectrum => {
                // Can be floats (inline wavelength/value pairs) or a single string (filename or named spectrum)
                if self.is_string_next() {
                    ParamValue::Strings(vec![self.expect_string()?])
                } else {
                    let mut vals = Vec::new();
                    while self.is_number_next() {
                        vals.push(self.expect_number()?);
                    }
                    ParamValue::Floats(vals)
                }
            }
            ParamType::Bool => {
                if bracketed {
                    let mut vals = Vec::new();
                    while self.is_bool_next() {
                        match self.advance() {
                            Some(Located {
                                value: Token::Bool(b),
                                ..
                            }) => vals.push(*b),
                            _ => break,
                        }
                    }
                    ParamValue::Bools(vals)
                } else {
                    match self.advance() {
                        Some(Located {
                            value: Token::Bool(b),
                            ..
                        }) => ParamValue::Bools(vec![*b]),
                        _ => return Err(self.err("expected bool value")),
                    }
                }
            }
            ParamType::String | ParamType::Texture => {
                if bracketed {
                    let mut vals = Vec::new();
                    while self.is_string_next() {
                        vals.push(self.expect_string()?);
                    }
                    ParamValue::Strings(vals)
                } else {
                    ParamValue::Strings(vec![self.expect_string()?])
                }
            }
        };

        if bracketed {
            match self.peek() {
                Some(Located {
                    value: Token::RBracket,
                    ..
                }) => {
                    self.advance();
                }
                _ => return Err(self.err("expected ']'")),
            }
        }

        Ok(value)
    }

    fn parse_single_value(&mut self) -> Result<ParamValue> {
        match self.peek() {
            Some(Located {
                value: Token::Number(n),
                ..
            }) => {
                let n = *n;
                self.advance();
                Ok(ParamValue::Floats(vec![n]))
            }
            Some(Located {
                value: Token::String(s),
                ..
            }) => {
                let s = s.clone();
                self.advance();
                Ok(ParamValue::Strings(vec![s]))
            }
            Some(Located {
                value: Token::Bool(b),
                ..
            }) => {
                let b = *b;
                self.advance();
                Ok(ParamValue::Bools(vec![b]))
            }
            _ => Err(self.err("expected a value")),
        }
    }

    fn is_number_next(&self) -> bool {
        matches!(
            self.peek(),
            Some(Located {
                value: Token::Number(_),
                ..
            })
        )
    }

    fn is_string_next(&self) -> bool {
        matches!(
            self.peek(),
            Some(Located {
                value: Token::String(_),
                ..
            })
        )
    }

    fn is_bool_next(&self) -> bool {
        matches!(
            self.peek(),
            Some(Located {
                value: Token::Bool(_),
                ..
            })
        )
    }
}

/// Parse `"type name"` into (ParamType, name).
fn parse_type_name(s: &str) -> Option<(ParamType, String)> {
    let mut parts = s.splitn(2, ' ');
    let ty_str = parts.next()?;
    let name = parts.next()?.to_string();
    if name.is_empty() {
        return None;
    }

    let ty = match ty_str {
        "integer" => ParamType::Integer,
        "float" => ParamType::Float,
        "point2" => ParamType::Point2,
        "vector2" => ParamType::Vector2,
        "point3" => ParamType::Point3,
        "vector3" => ParamType::Vector3,
        "normal3" => ParamType::Normal3,
        "bool" => ParamType::Bool,
        "string" => ParamType::String,
        "rgb" => ParamType::Rgb,
        "spectrum" => ParamType::Spectrum,
        "blackbody" => ParamType::Blackbody,
        "texture" => ParamType::Texture,
        _ => return None,
    };

    Some((ty, name))
}

/// Parse a PBRT v4 scene from source text.
pub fn parse(input: &str) -> Result<Scene> {
    let tokens = Lexer::new(input).tokenize()?;
    Parser::new(tokens).parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let scene = parse("").unwrap();
        assert!(scene.directives.is_empty());
    }

    #[test]
    fn test_minimal_scene() {
        let scene = parse(
            r#"
            LookAt 3 4 1.5  .5 .5 0  0 0 1
            Camera "perspective" "float fov" 45
            Film "rgb" "integer xresolution" [1920] "integer yresolution" [1080]
            WorldBegin
            Shape "sphere" "float radius" 1
        "#,
        )
        .unwrap();

        assert_eq!(scene.directives.len(), 5);

        match &scene.directives[0] {
            Directive::LookAt { eye, look, up } => {
                assert_eq!(eye, &[3.0, 4.0, 1.5]);
                assert_eq!(look, &[0.5, 0.5, 0.0]);
                assert_eq!(up, &[0.0, 0.0, 1.0]);
            }
            _ => panic!("expected LookAt"),
        }

        match &scene.directives[1] {
            Directive::Camera { ty, params } => {
                assert_eq!(ty, "perspective");
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name, "fov");
            }
            _ => panic!("expected Camera"),
        }
    }

    #[test]
    fn test_texture_directive() {
        let scene = parse(
            r#"
            Texture "checks" "spectrum" "checkerboard"
                "float uscale" [16] "float vscale" [16]
        "#,
        )
        .unwrap();

        match &scene.directives[0] {
            Directive::Texture {
                name,
                ty,
                class,
                params,
            } => {
                assert_eq!(name, "checks");
                assert_eq!(ty, "spectrum");
                assert_eq!(class, "checkerboard");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected Texture"),
        }
    }

    #[test]
    fn test_attribute_blocks() {
        let scene = parse(
            r#"
            WorldBegin
            AttributeBegin
                Material "diffuse" "rgb reflectance" [.5 .5 .5]
                Shape "sphere"
            AttributeEnd
        "#,
        )
        .unwrap();

        assert_eq!(scene.directives.len(), 5);
        matches!(&scene.directives[1], Directive::AttributeBegin);
        matches!(&scene.directives[4], Directive::AttributeEnd);
    }

    #[test]
    fn test_transforms() {
        let scene = parse(
            r#"
            Identity
            Translate 1 2 3
            Rotate 90 0 1 0
            Scale 2 2 2
        "#,
        )
        .unwrap();

        assert_eq!(scene.directives.len(), 4);
        match &scene.directives[1] {
            Directive::Translate { v } => assert_eq!(v, &[1.0, 2.0, 3.0]),
            _ => panic!("expected Translate"),
        }
        match &scene.directives[2] {
            Directive::Rotate { angle, axis } => {
                assert_eq!(*angle, 90.0);
                assert_eq!(axis, &[0.0, 1.0, 0.0]);
            }
            _ => panic!("expected Rotate"),
        }
    }

    #[test]
    fn test_triangle_mesh() {
        let scene = parse(
            r#"
            Shape "trianglemesh"
                "point3 P" [ -1 0 0  1 0 0  0 1 0 ]
                "integer indices" [ 0 1 2 ]
        "#,
        )
        .unwrap();

        match &scene.directives[0] {
            Directive::Shape { ty, params } => {
                assert_eq!(ty, "trianglemesh");
                assert_eq!(params.len(), 2);
                match &params[0].value {
                    ParamValue::Floats(v) => assert_eq!(v.len(), 9),
                    _ => panic!("expected floats"),
                }
                match &params[1].value {
                    ParamValue::Ints(v) => assert_eq!(v, &[0, 1, 2]),
                    _ => panic!("expected ints"),
                }
            }
            _ => panic!("expected Shape"),
        }
    }

    #[test]
    fn test_make_named_material() {
        let scene = parse(
            r#"
            MakeNamedMaterial "mymtl" "string type" "diffuse" "rgb reflectance" [.8 .2 .1]
        "#,
        )
        .unwrap();

        match &scene.directives[0] {
            Directive::MakeNamedMaterial { name, params } => {
                assert_eq!(name, "mymtl");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected MakeNamedMaterial"),
        }
    }

    #[test]
    fn test_spectrum_string() {
        let scene = parse(
            r#"
            Material "conductor" "spectrum eta" "metal-Cu-eta"
        "#,
        )
        .unwrap();

        match &scene.directives[0] {
            Directive::Material { params, .. } => match &params[0].value {
                ParamValue::Strings(v) => assert_eq!(v[0], "metal-Cu-eta"),
                _ => panic!("expected string spectrum"),
            },
            _ => panic!("expected Material"),
        }
    }

    #[test]
    fn test_comments() {
        let scene = parse(
            r#"
            # This is a comment
            WorldBegin  # inline comment
            # Another comment
            Shape "sphere"
        "#,
        )
        .unwrap();

        assert_eq!(scene.directives.len(), 2);
    }
}
