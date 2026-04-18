"""Map Blender Principled BSDF + Image Texture nodes to vibrt-blender JSON."""

from __future__ import annotations

import struct

import bpy


def _pack_f32(xs) -> bytes:
    return struct.pack(f"<{len(xs)}f", *xs)


def _write_blob(buf: bytearray, data: bytes) -> dict:
    off = len(buf)
    buf.extend(data)
    pad = (-len(buf)) & 15
    buf.extend(b"\x00" * pad)
    return {"offset": off, "len": len(data)}


def export_image_texture(
    image: bpy.types.Image,
    buf: bytearray,
    textures: list,
    colorspace: str | None = None,
) -> int:
    """Serialize an image into scene.bin and register a TextureDesc entry.

    Returns the texture index. Reuses existing entries by image name.
    """
    key = f"__image__{image.name}"
    for i, t in enumerate(textures):
        if t.get("_key") == key:
            return i
    if image.size[0] == 0 or image.size[1] == 0:
        image.update()
    width, height = image.size[0], image.size[1]
    if width == 0 or height == 0:
        # Fallback 1x1 white
        width, height = 1, 1
        pixels = [1.0, 1.0, 1.0, 1.0]
    else:
        pixels = list(image.pixels[:])  # RGBA f32
    if colorspace is None:
        colorspace = (
            "srgb"
            if image.colorspace_settings.name.lower().startswith("srgb")
            else "linear"
        )
    desc = {
        "width": int(width),
        "height": int(height),
        "channels": 4,
        "colorspace": colorspace,
        "pixels": _write_blob(buf, _pack_f32(pixels)),
        "_key": key,
    }
    textures.append(desc)
    return len(textures) - 1


def _socket_default(sock):
    return sock.default_value


def _socket_linked_image(sock) -> bpy.types.Image | None:
    if not sock.is_linked:
        return None
    src = sock.links[0].from_node
    if src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
        return src.image
    return None


def export_material(
    mat: bpy.types.Material, buf: bytearray, textures: list
) -> dict:
    """Extract Principled BSDF parameters from `mat`. Falls back to base color."""
    base_color = [0.8, 0.8, 0.8]
    metallic = 0.0
    roughness = 0.5
    ior = 1.45
    transmission = 0.0
    emission = [0.0, 0.0, 0.0]
    base_tex = None
    normal_tex = None
    rough_tex = None
    metal_tex = None

    if mat.use_nodes:
        out = next(
            (
                n
                for n in mat.node_tree.nodes
                if n.bl_idname == "ShaderNodeOutputMaterial"
                and n.is_active_output
            ),
            None,
        )
        if out is None:
            out = next(
                (
                    n
                    for n in mat.node_tree.nodes
                    if n.bl_idname == "ShaderNodeOutputMaterial"
                ),
                None,
            )
        if out is not None and out.inputs["Surface"].is_linked:
            src = out.inputs["Surface"].links[0].from_node
            if src.bl_idname == "ShaderNodeBsdfPrincipled":
                bc = src.inputs["Base Color"]
                base_color = list(_socket_default(bc))[:3]
                img = _socket_linked_image(bc)
                if img is not None:
                    base_tex = export_image_texture(img, buf, textures, "srgb")

                metallic = float(_socket_default(src.inputs["Metallic"]))
                img = _socket_linked_image(src.inputs["Metallic"])
                if img is not None:
                    metal_tex = export_image_texture(img, buf, textures, "linear")

                roughness = float(_socket_default(src.inputs["Roughness"]))
                img = _socket_linked_image(src.inputs["Roughness"])
                if img is not None:
                    rough_tex = export_image_texture(img, buf, textures, "linear")

                if "IOR" in src.inputs:
                    ior = float(_socket_default(src.inputs["IOR"]))
                # Blender 4.x: "Transmission Weight"; older: "Transmission"
                trans_name = (
                    "Transmission Weight"
                    if "Transmission Weight" in src.inputs
                    else "Transmission"
                    if "Transmission" in src.inputs
                    else None
                )
                if trans_name:
                    transmission = float(_socket_default(src.inputs[trans_name]))

                if "Emission" in src.inputs:
                    emission = list(_socket_default(src.inputs["Emission"]))[:3]
                elif "Emission Color" in src.inputs:
                    em_color = list(_socket_default(src.inputs["Emission Color"]))[:3]
                    em_strength = float(
                        _socket_default(src.inputs["Emission Strength"])
                    ) if "Emission Strength" in src.inputs else 1.0
                    emission = [c * em_strength for c in em_color]

                # Normal input
                norm_sock = src.inputs.get("Normal")
                if norm_sock is not None and norm_sock.is_linked:
                    link_src = norm_sock.links[0].from_node
                    img = None
                    if link_src.bl_idname == "ShaderNodeNormalMap":
                        img = _socket_linked_image(link_src.inputs["Color"])
                    elif link_src.bl_idname == "ShaderNodeTexImage":
                        img = link_src.image
                    if img is not None:
                        normal_tex = export_image_texture(img, buf, textures, "linear")
    else:
        # Fallback: diffuse color from mat.diffuse_color
        base_color = list(mat.diffuse_color)[:3]

    out: dict = {
        "base_color": base_color,
        "metallic": metallic,
        "roughness": roughness,
        "ior": ior,
        "transmission": transmission,
        "emission": emission,
    }
    if base_tex is not None:
        out["base_color_tex"] = base_tex
    if normal_tex is not None:
        out["normal_tex"] = normal_tex
    if rough_tex is not None:
        out["roughness_tex"] = rough_tex
    if metal_tex is not None:
        out["metallic_tex"] = metal_tex
    return out
