import bpy


class VIBRT_PT_sampling(bpy.types.Panel):
    bl_label = "Sampling"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {"VIBRT"}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(context.scene, "vibrt_spp")


def register():
    bpy.types.Scene.vibrt_spp = bpy.props.IntProperty(
        name="Samples",
        description="Samples per pixel for vibrt rendering",
        default=64,
        min=1,
        soft_max=4096,
    )
    bpy.utils.register_class(VIBRT_PT_sampling)


def unregister():
    bpy.utils.unregister_class(VIBRT_PT_sampling)
    del bpy.types.Scene.vibrt_spp
