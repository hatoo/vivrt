import bpy


class VibrtBlenderPrefs(bpy.types.AddonPreferences):
    bl_idname = __package__

    vibrt_executable: bpy.props.StringProperty(
        name="vibrt-blender executable",
        subtype="FILE_PATH",
        description="Path to the vibrt-blender binary. If empty, $VIBRT_BLENDER_EXECUTABLE or PATH is used.",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "vibrt_executable")
        layout.label(
            text="Lookup order: addon preference > $VIBRT_BLENDER_EXECUTABLE > PATH",
            icon="INFO",
        )


def register():
    bpy.utils.register_class(VibrtBlenderPrefs)


def unregister():
    bpy.utils.unregister_class(VibrtBlenderPrefs)
