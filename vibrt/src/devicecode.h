#pragma once

#include <optix_types.h>

enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_DIELECTRIC = 1,
    MAT_COATED_DIFFUSE = 2,
    MAT_CONDUCTOR = 3,
    MAT_COATED_CONDUCTOR = 4,
};

struct DistantLight {
    float direction[3]; // normalized, pointing TO the light
    float emission[3];
};

struct SphereLight {
    float center[3];
    float radius;
    float emission[3];
    float _pad;
};

struct TriangleLight {
    float v0[3];
    float v1[3];
    float v2[3];
    float emission[3];
    float normal[3];
    float area;
    float _pad;
};

struct DiffuseParams {
    int   has_checkerboard;
    float checker_scale_u;
    float checker_scale_v;
    float checker_color1[3];
    float checker_color2[3];
};

struct DielectricParams {
    float eta;
    float tint[3]; // absorption tint from participating medium
};

struct ConductorParams {
    float eta[3];
    float k[3];
};

struct HitGroupData {
    // Material
    int           material_type;
    float         albedo[3];
    float         emission[3];
    float         roughness;
    float         roughness_v;
    float         coat_roughness;
    float         coat_eta;
    union {
        DiffuseParams       diffuse;
        DielectricParams    dielectric;
        ConductorParams     conductor;
    };
    // Geometry
    float*        vertices;
    float*        normals;
    int*          indices;
    float*        texcoords;
    int           num_vertices;
    // Texture maps
    float*        texture_data;
    int           texture_width;
    int           texture_height;
    float*        bump_data;
    int           bump_width;
    int           bump_height;
    float*        alpha_data;
    int           alpha_width;
    int           alpha_height;
    float*        roughness_data;
    int           roughness_width;
    int           roughness_height;
};

struct LaunchParams {
    unsigned int* image;
    unsigned int  width;
    unsigned int  height;
    unsigned int  samples_per_pixel;
    unsigned int  max_depth;
    float         cam_eye[3];
    float         cam_u[3];
    float         cam_v[3];
    float         cam_w[3];
    OptixTraversableHandle traversable;

    float         ambient_light[3];
    int           num_distant_lights;
    DistantLight* distant_lights;
    int           num_sphere_lights;
    SphereLight*  sphere_lights;
    int           num_triangle_lights;
    TriangleLight* triangle_lights;

    // Environment map (IBL)
    float*        envmap_data;
    int           envmap_width;
    int           envmap_height;
    float*        envmap_marginal_cdf;     // float[height+1]
    float*        envmap_conditional_cdf;  // float[height*(width+1)]
    float         envmap_integral;

    // GGX energy compensation LUT (Kulla-Conty)
    float*        ggx_e_lut;       // E(cosTheta, alpha), 32x32
    float*        ggx_e_avg_lut;   // E_avg(alpha), 32
};

struct RayGenData {};

struct MissData {
    float bg_color[3];
};
