#pragma once

#include <optix_types.h>

/// Single node in a material's colour graph. Layout mirrors
/// `gpu_types::ColorGraphNode` on the Rust side.
struct ColorGraphNode {
  unsigned int tag;
  unsigned int payload[15];
};

#define COLOR_NODE_CONST 0u
#define COLOR_NODE_IMAGE_TEX 1u
#define COLOR_NODE_MIX 2u
#define COLOR_NODE_INVERT 3u
#define COLOR_NODE_MATH 4u
#define COLOR_NODE_HUE_SAT 5u
#define COLOR_NODE_RGB_CURVE 6u
#define COLOR_NODE_BRIGHT_CONTRAST 7u
#define COLOR_NODE_VERTEX_COLOR 8u

struct PrincipledGpu {
  float base_color[3];
  float metallic;
  float roughness;
  float ior;
  float transmission;
  float emission[3];
  float *base_color_tex;
  int base_color_tex_w;
  int base_color_tex_h;
  int base_color_tex_channels;
  float *normal_tex;
  int normal_tex_w;
  int normal_tex_h;
  int normal_tex_channels;
  float *roughness_tex;
  int roughness_tex_w;
  int roughness_tex_h;
  int roughness_tex_channels;
  float *metallic_tex;
  int metallic_tex_w;
  int metallic_tex_h;
  int metallic_tex_channels;
  float *transmission_tex;
  int transmission_tex_w;
  int transmission_tex_h;
  int transmission_tex_channels;
  float *emission_tex;
  int emission_tex_w;
  int emission_tex_h;
  int emission_tex_channels;
  float uv_transform[6];
  float normal_strength;
  float bump_strength;
  float *bump_tex;
  int bump_tex_w;
  int bump_tex_h;
  int bump_tex_channels;
  float alpha_threshold;
  float anisotropy;
  float tangent_rotation;
  float coat_weight;
  float coat_roughness;
  float coat_ior;
  float sheen_weight;
  float sheen_roughness;
  float sheen_tint[3];
  float sss_weight;
  float sss_radius[3];
  float hair_weight;
  float hair_offset;
  float hair_roughness_u;
  float hair_roughness_v;
  int use_vertex_color;
  ColorGraphNode *color_graph_nodes;
  int color_graph_len;
  int color_graph_output;
};

struct HitGroupData {
  PrincipledGpu *mat;
  float *vertices;
  float *normals;
  int *indices;
  float *uvs;
  int num_vertices;
  int area_light_group;
  unsigned int *material_indices; // nullptr => use `mat`
  PrincipledGpu **materials;
  int num_materials;
  float *vertex_colors; // f32x3 per vertex, nullptr if absent
};

struct PointLight {
  float position[3];
  float radius;
  float emission[3];
  float _pad;
};

struct SunLight {
  float direction[3]; // points TO the light (opposite of "sun direction")
  float cos_angle;
  float emission[3];
  float _pad;
};

struct SpotLight {
  float position[3];
  float _pad0;
  float direction[3]; // emission direction
  float cos_outer;
  float emission[3];
  float cos_inner;
};

struct AreaRectLight {
  float corner[3];
  float size_u;
  float u_axis[3];
  float size_v;
  float v_axis[3];
  unsigned int two_sided;
  float normal[3];
  unsigned int camera_visible;
  float emission[3];
  float power;
};

struct LaunchParams {
  float *image; // float4 per pixel
  unsigned int width;
  unsigned int height;
  unsigned int samples_per_pixel;
  unsigned int max_depth;

  float cam_eye[3];
  float cam_u[3];
  float cam_v[3];
  float cam_w[3];
  float cam_lens_radius;
  float cam_focal_distance;

  OptixTraversableHandle traversable;

  int num_point_lights;
  PointLight *point_lights;
  int num_sun_lights;
  SunLight *sun_lights;
  int num_spot_lights;
  SpotLight *spot_lights;
  int num_rect_lights;
  AreaRectLight *rect_lights;
  float *rect_light_cdf;

  int world_type; // 0=constant, 1=envmap
  float world_color[3];
  float world_strength;
  float *envmap_data;
  int envmap_width;
  int envmap_height;
  float *envmap_marginal_cdf;
  float *envmap_conditional_cdf;
  float envmap_integral;
  float envmap_rotation_z_rad;

  float *ggx_e_lut;
  float *ggx_e_avg_lut;

  // Clamp for indirect (bounce>=1) contributions. <=0 disables.
  float clamp_indirect;

  // Denoiser guide AOVs (optional; null when denoising is off). Each is one
  // float3 per pixel, captured from a single un-jittered primary ray.
  // `normal_aov` is in camera space: X=right, Y=up, Z=forward.
  float *albedo_aov;
  float *normal_aov;
};

struct RayGenData {};

struct MissData {
  int _unused;
};
