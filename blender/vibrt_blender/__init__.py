bl_info = {
    "name": "vibrt",
    "author": "vibrt",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "Render > Render Engine",
    "description": "OptiX path-tracing renderer via vibrt",
    "category": "Render",
}

import bpy

from . import preferences
from . import properties
from . import engine


def register():
    preferences.register()
    properties.register()
    engine.register()


def unregister():
    engine.unregister()
    properties.unregister()
    preferences.unregister()
