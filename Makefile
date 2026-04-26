# Render previews from .blend test scenes via the in-process addon path,
# and build / install the addon itself. The standalone vibrt CLI binary
# is gone — everything goes through Blender + the bundled
# vibrt_native.pyd extension.
#
# Usage:
#   make previews             # render preview.png for each .blend scene
#   make junk_shop-preview    # render one scene
#   make junk_shop-cycles     # Cycles reference render of the same scene
#   make cycles-previews      # all .blend scenes via Cycles
#   make addon                # blender/vibrt_blender.zip (Python only — won't render)
#   make addon-with-native    # also build + bundle vibrt_native.pyd
#   make dev-install          # build .pyd, stage, junction into Blender's user-addons
#   make clean                # remove preview.png, addon zip, staged .pyd
#
# Overridable:
#   PYTHON  (default: py)
#   SPP     (default: 128)
#   PCT     (unset): render resolution percentage, e.g. PCT=25
#   DENOISE (unset): set to 1 to run the OptiX AI denoiser on the output

PYTHON  ?= py
SPP     ?= 128
PCT     ?=
DENOISE ?=

# .blend-based scenes: one entry per test_scenes/<name>/ directory.
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

.PHONY: all previews cycles-previews addon addon-with-native dev-install clean native-build FORCE \
        $(BLEND_PREVIEW_TARGETS) $(BLEND_CYCLES_TARGETS)

FORCE:

# Build the PyO3 extension and stage it next to the addon source. Cargo is
# incremental, so this is a no-op when nothing changed. `make dev-install`
# does the full pipeline (build + stage + junction); this target is the
# build-only step the preview rules depend on.
native-build:
	$(PYTHON) blender/build_addon.py --with-native --stage-only

all: previews

previews: $(BLEND_PREVIEW_PNGS)
cycles-previews: $(BLEND_CYCLES_PNGS)
addon: $(ADDON_ZIP)

$(BLEND_PREVIEW_TARGETS): %-preview: test_scenes/%/preview.png
$(BLEND_CYCLES_TARGETS): %-cycles: test_scenes/%/preview_cycles.png

# Each .blend scene's preview goes through scripts/render_blend.py, which
# spawns Blender headless, force-loads the working-tree addon, and
# triggers a single F12 render via vibrt_native. Depends on the addon
# Python sources (so material/exporter changes invalidate previews) and
# the staged .pyd (built via native-build).
define BLEND_PREVIEW_RULE
test_scenes/$(1)/preview.png: $$(BLEND_SCENE_$(1)) $$(ADDON_SOURCES) scripts/render_blend.py scripts/_blender_render.py native-build FORCE
	$$(PYTHON) scripts/render_blend.py $$< --output $$@ --spp $$(SPP) $$(PCT_FLAG) $$(DENOISE_FLAG)
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

# Same zip target with `--with-native`: invokes `cargo build --features python`
# and copies `vibrt_native.dll` (or platform equivalent) into the zip so the
# addon can render out of the box. The bare `addon` target produces a zip
# that won't render — use this one for distribution.
addon-with-native: $(ADDON_SOURCES) blender/build_addon.py
	$(PYTHON) blender/build_addon.py --with-native

# `make dev-install` is the one-shot for fresh checkouts:
#   1. Build the Rust crate with the `python` feature so vibrt_native.dll
#      lands in target/release/.
#   2. Stage it as `blender/vibrt_blender/vibrt_native.pyd` (the dev-junction
#      target). Without this the addon refuses to render.
#   3. Junction the addon dir into Blender's user-addons folder.
dev-install:
	$(PYTHON) blender/build_addon.py --with-native --stage-only
	$(PYTHON) blender/dev_install.py

clean:
	rm -f $(BLEND_PREVIEW_PNGS) $(BLEND_CYCLES_PNGS) $(ADDON_ZIP) \
	      blender/vibrt_blender/vibrt_native.pyd \
	      blender/vibrt_blender/vibrt_native.so \
	      blender/vibrt_blender/vibrt_native.dylib
