#pragma once

#include <optix_types.h>

enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_DIELECTRIC = 1,
    MAT_COATED_DIFFUSE = 2,
    MAT_CONDUCTOR = 3,
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
};

struct CoatedDiffuseParams {
    float roughness;
};

struct HitGroupData {
    int           material_type;
    float         albedo[3];
    float         emission[3];
    // Material-specific params (union)
    union {
        DiffuseParams       diffuse;
        DielectricParams    dielectric;
        CoatedDiffuseParams coated;
    };
    // Image texture (NULL if no texture)
    float*        texture_data;  // RGB float, width*height*3
    int           texture_width;
    int           texture_height;
    // Geometry data
    float*        texcoords;
    float*        normals;
    int*          indices;
    float*        vertices;
    int           num_vertices;
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
};

struct RayGenData {};

struct MissData {
    float bg_color[3];
};
