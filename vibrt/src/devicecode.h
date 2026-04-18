#pragma once

#include <optix_types.h>

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
  float sss_anisotropy;
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
  int _pad_hg;
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
  float _pad0;
  float normal[3];
  float _pad1;
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
};

struct RayGenData {};

struct MissData {
  int _unused;
};
