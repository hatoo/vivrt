#pragma once

#include <optix_types.h>

enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_DIELECTRIC = 1,
};

struct DistantLight {
    float direction[3]; // normalized, pointing TO the light
    float emission[3];
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
};

struct RayGenData {};

struct MissData {
    float bg_color[3];
};

struct HitGroupData {
    int           material_type;
    float         albedo[3];
    float         eta;
    // Procedural checkerboard
    int           has_checkerboard;
    float         checker_scale_u;
    float         checker_scale_v;
    float         checker_color1[3];
    float         checker_color2[3];
    // Geometry data (triangles)
    float*        texcoords; // 2 floats per vertex, or NULL
    float*        normals;   // 3 floats per vertex, or NULL (smooth shading)
    int*          indices;   // 3 ints per triangle, or NULL
    float*        vertices;  // 3 floats per vertex
    int           num_vertices;
};
