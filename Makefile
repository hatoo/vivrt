# Regenerate test scenes (and optionally render previews) from their
# procedural generators or from .blend files.
#
# Usage:
#   make                      # regenerate every test_scenes/*/scene.json
#   make veach_mis            # regenerate one scene by directory name
#   make previews             # regenerate scenes and render preview.png for each
#   make veach_mis-preview    # render preview for one scene
#   make classroom-preview    # export + render a .blend-based scene
#   make addon                # rebuild blender/vibrt_blender.zip
#   make dev-install          # junction the addon into Blender's user addons dir
#   make clean                # remove generated scene.json, scene.bin, preview.png
#
# Overridable:
#   PYTHON  (default: py)
#   VIBRT   (default: ./target/release/vibrt.exe)
#   SPP     (default: 128)
#   PCT     (unset): render resolution percentage for .blend scenes, e.g. PCT=25

PYTHON ?= py
VIBRT  ?= ./target/release/vibrt.exe
SPP    ?= 128
PCT    ?=

SCENE_SCRIPTS := $(wildcard test_scenes/*/make_scene.py)
SCENES        := $(patsubst test_scenes/%/make_scene.py,%,$(SCENE_SCRIPTS))
SCENE_JSONS   := $(SCENE_SCRIPTS:make_scene.py=scene.json)
SCENE_BINS    := $(SCENE_SCRIPTS:make_scene.py=scene.bin)
PREVIEW_PNGS  := $(SCENE_SCRIPTS:make_scene.py=preview.png)

PREVIEW_TARGETS := $(addsuffix -preview,$(SCENES))

# .blend-based scenes: one entry per test_scenes/<name>/ directory that is
# driven by a .blend file. Listed explicitly so a missing entry is obvious.
BLEND_SCENE_classroom := test_scenes/classroom/classroom/classroom.blend

BLEND_SCENES          := classroom
BLEND_PREVIEW_PNGS    := $(foreach s,$(BLEND_SCENES),test_scenes/$(s)/preview.png)
BLEND_PREVIEW_TARGETS := $(addsuffix -preview,$(BLEND_SCENES))

PCT_FLAG := $(if $(strip $(PCT)),--percentage $(PCT))

ADDON_ZIP     := blender/vibrt_blender.zip
ADDON_SOURCES := $(wildcard blender/vibrt_blender/*.py)

.PHONY: all scenes previews addon dev-install clean $(SCENES) $(PREVIEW_TARGETS) $(BLEND_PREVIEW_TARGETS)

all: scenes

scenes: $(SCENE_JSONS)
previews: $(PREVIEW_PNGS) $(BLEND_PREVIEW_PNGS)
addon: $(ADDON_ZIP)

# Shorthand: `make <scene>` regenerates scene.json;
#            `make <scene>-preview` renders preview.png.
$(SCENES): %: test_scenes/%/scene.json
$(PREVIEW_TARGETS): %-preview: test_scenes/%/preview.png
$(BLEND_PREVIEW_TARGETS): %-preview: test_scenes/%/preview.png

# Running make_scene.py writes scene.json and scene.bin side-by-side.
test_scenes/%/scene.json: test_scenes/%/make_scene.py
	cd $(dir $<) && $(PYTHON) make_scene.py

test_scenes/%/preview.png: test_scenes/%/scene.json
	$(VIBRT) $< --spp $(SPP) --output $@

# .blend scenes go via scripts/render_blend.py, which spawns Blender to export
# then runs vibrt. Rebuilds when the .blend or the addon's Python sources
# change (the latter because material_export.py changes the exported scene).
define BLEND_PREVIEW_RULE
test_scenes/$(1)/preview.png: $$(BLEND_SCENE_$(1)) $$(ADDON_SOURCES) scripts/render_blend.py scripts/_blender_export.py
	$$(PYTHON) scripts/render_blend.py $$< --output $$@ --spp $$(SPP) --vibrt $$(VIBRT) $$(PCT_FLAG)
endef
$(foreach s,$(BLEND_SCENES),$(eval $(call BLEND_PREVIEW_RULE,$(s))))

$(ADDON_ZIP): $(ADDON_SOURCES) blender/build_addon.py
	$(PYTHON) blender/build_addon.py

dev-install:
	$(PYTHON) blender/dev_install.py

clean:
	rm -f $(SCENE_JSONS) $(SCENE_BINS) $(PREVIEW_PNGS) $(BLEND_PREVIEW_PNGS) $(ADDON_ZIP)
