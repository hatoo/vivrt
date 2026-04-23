# Regenerate test scenes (and optionally render previews) from their
# procedural generators or from .blend files.
#
# Usage:
#   make                      # regenerate every test_scenes/*/scene.json
#   make veach_mis            # regenerate one scene by directory name
#   make previews             # regenerate scenes and render preview.png for each
#   make veach_mis-preview    # render preview for one scene
#   make classroom-preview    # export + render a .blend-based scene
#   make classroom-cycles     # render a .blend-based scene with Cycles (reference)
#   make cycles-previews      # render every .blend scene with Cycles
#   make addon                # rebuild blender/vibrt_blender.zip
#   make dev-install          # junction the addon into Blender's user addons dir
#   make clean                # remove generated scene.json, scene.bin, preview.png
#
# Overridable:
#   PYTHON  (default: py)
#   VIBRT   (default: ./target/release/vibrt.exe)
#   SPP     (default: 128)
#   PCT     (unset): render resolution percentage for .blend scenes, e.g. PCT=25
#   DENOISE (unset): set to 1 to run the OptiX AI denoiser on the output

PYTHON  ?= py
VIBRT   ?= ./target/release/vibrt.exe
SPP     ?= 128
PCT     ?=
DENOISE ?=

SCENE_SCRIPTS := $(wildcard test_scenes/*/make_scene.py)
SCENES        := $(patsubst test_scenes/%/make_scene.py,%,$(SCENE_SCRIPTS))
SCENE_JSONS   := $(SCENE_SCRIPTS:make_scene.py=scene.json)
SCENE_BINS    := $(SCENE_SCRIPTS:make_scene.py=scene.bin)
PREVIEW_PNGS  := $(SCENE_SCRIPTS:make_scene.py=preview.png)

PREVIEW_TARGETS := $(addsuffix -preview,$(SCENES))

# .blend-based scenes: one entry per test_scenes/<name>/ directory that is
# driven by a .blend file. Listed explicitly so a missing entry is obvious.
BLEND_SCENE_classroom := test_scenes/classroom/classroom/classroom.blend
BLEND_SCENE_bmw27     := test_scenes/bmw27/bmw27/bmw27_gpu.blend
BLEND_SCENE_junk_shop := test_scenes/junk_shop/junk_shop/junk_shop.blend

BLEND_SCENES          := classroom bmw27 junk_shop
BLEND_PREVIEW_PNGS    := $(foreach s,$(BLEND_SCENES),test_scenes/$(s)/preview.png)
BLEND_PREVIEW_TARGETS := $(addsuffix -preview,$(BLEND_SCENES))
BLEND_CYCLES_PNGS     := $(foreach s,$(BLEND_SCENES),test_scenes/$(s)/preview_cycles.png)
BLEND_CYCLES_TARGETS  := $(addsuffix -cycles,$(BLEND_SCENES))

PCT_FLAG     := $(if $(strip $(PCT)),--percentage $(PCT))
DENOISE_FLAG := $(if $(strip $(DENOISE)),--denoise)

ADDON_ZIP     := blender/vibrt_blender.zip
ADDON_SOURCES := $(wildcard blender/vibrt_blender/*.py)

.PHONY: all scenes previews cycles-previews addon dev-install clean FORCE $(SCENES) $(PREVIEW_TARGETS) $(BLEND_PREVIEW_TARGETS) $(BLEND_CYCLES_TARGETS)

FORCE:

all: scenes

scenes: $(SCENE_JSONS)
previews: $(PREVIEW_PNGS) $(BLEND_PREVIEW_PNGS)
cycles-previews: $(BLEND_CYCLES_PNGS)
addon: $(ADDON_ZIP)

# Shorthand: `make <scene>` regenerates scene.json;
#            `make <scene>-preview` renders preview.png.
$(SCENES): %: test_scenes/%/scene.json
$(PREVIEW_TARGETS): %-preview: test_scenes/%/preview.png
$(BLEND_PREVIEW_TARGETS): %-preview: test_scenes/%/preview.png
$(BLEND_CYCLES_TARGETS): %-cycles: test_scenes/%/preview_cycles.png

# Running make_scene.py writes scene.json and scene.bin side-by-side.
test_scenes/%/scene.json: test_scenes/%/make_scene.py
	cd $(dir $<) && $(PYTHON) make_scene.py

test_scenes/%/preview.png: test_scenes/%/scene.json FORCE
	$(VIBRT) $< --spp $(SPP) --output $@ $(DENOISE_FLAG)

# .blend scenes go via scripts/render_blend.py, which spawns Blender to export
# then runs vibrt. Rebuilds when the .blend or the addon's Python sources
# change (the latter because material_export.py changes the exported scene).
define BLEND_PREVIEW_RULE
test_scenes/$(1)/preview.png: $$(BLEND_SCENE_$(1)) $$(ADDON_SOURCES) scripts/render_blend.py scripts/_blender_export.py FORCE
	$$(PYTHON) scripts/render_blend.py $$< --output $$@ --spp $$(SPP) --vibrt $$(VIBRT) $$(PCT_FLAG) $$(DENOISE_FLAG)
endef
$(foreach s,$(BLEND_SCENES),$(eval $(call BLEND_PREVIEW_RULE,$(s))))

# Reference render via Cycles, for side-by-side comparison with vibrt output.
define BLEND_CYCLES_RULE
test_scenes/$(1)/preview_cycles.png: $$(BLEND_SCENE_$(1)) scripts/render_cycles.py scripts/_blender_cycles.py FORCE
	$$(PYTHON) scripts/render_cycles.py $$< --output $$@ --spp $$(SPP) $$(PCT_FLAG)
endef
$(foreach s,$(BLEND_SCENES),$(eval $(call BLEND_CYCLES_RULE,$(s))))

$(ADDON_ZIP): $(ADDON_SOURCES) blender/build_addon.py
	$(PYTHON) blender/build_addon.py

dev-install:
	$(PYTHON) blender/dev_install.py

clean:
	rm -f $(SCENE_JSONS) $(SCENE_BINS) $(PREVIEW_PNGS) $(BLEND_PREVIEW_PNGS) $(BLEND_CYCLES_PNGS) $(ADDON_ZIP)
