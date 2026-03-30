use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file.pbrt>", args[0]);
        std::process::exit(1);
    }

    let input = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", args[1]);
        std::process::exit(1);
    });

    match pbrt_parser::parse(&input) {
        Ok(scene) => {
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for d in &scene.directives {
                let name = match d {
                    pbrt_parser::Directive::Camera { .. } => "Camera",
                    pbrt_parser::Directive::Film { .. } => "Film",
                    pbrt_parser::Directive::Sampler { .. } => "Sampler",
                    pbrt_parser::Directive::Integrator { .. } => "Integrator",
                    pbrt_parser::Directive::PixelFilter { .. } => "PixelFilter",
                    pbrt_parser::Directive::Accelerator { .. } => "Accelerator",
                    pbrt_parser::Directive::WorldBegin => "WorldBegin",
                    pbrt_parser::Directive::Shape { .. } => "Shape",
                    pbrt_parser::Directive::Material { .. } => "Material",
                    pbrt_parser::Directive::MakeNamedMaterial { .. } => "MakeNamedMaterial",
                    pbrt_parser::Directive::NamedMaterial(_) => "NamedMaterial",
                    pbrt_parser::Directive::Texture { .. } => "Texture",
                    pbrt_parser::Directive::LightSource { .. } => "LightSource",
                    pbrt_parser::Directive::AreaLightSource { .. } => "AreaLightSource",
                    pbrt_parser::Directive::AttributeBegin => "AttributeBegin",
                    pbrt_parser::Directive::AttributeEnd => "AttributeEnd",
                    pbrt_parser::Directive::Translate { .. } => "Translate",
                    pbrt_parser::Directive::Rotate { .. } => "Rotate",
                    pbrt_parser::Directive::Scale { .. } => "Scale",
                    pbrt_parser::Directive::LookAt { .. } => "LookAt",
                    pbrt_parser::Directive::Transform { .. } => "Transform",
                    pbrt_parser::Directive::ConcatTransform { .. } => "ConcatTransform",
                    pbrt_parser::Directive::Include(_) => "Include",
                    pbrt_parser::Directive::Import(_) => "Import",
                    pbrt_parser::Directive::Identity => "Identity",
                    pbrt_parser::Directive::ObjectBegin(_) => "ObjectBegin",
                    pbrt_parser::Directive::ObjectEnd => "ObjectEnd",
                    pbrt_parser::Directive::ObjectInstance(_) => "ObjectInstance",
                    pbrt_parser::Directive::MakeNamedMedium { .. } => "MakeNamedMedium",
                    pbrt_parser::Directive::MediumInterface { .. } => "MediumInterface",
                    _ => "other",
                };
                *counts.entry(name).or_default() += 1;
            }

            println!("Parsed {} directives:", scene.directives.len());
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            for (name, count) in sorted {
                println!("  {name}: {count}");
            }
        }
        Err(e) => {
            eprintln!("Parse error: {e}");
            std::process::exit(1);
        }
    }
}
