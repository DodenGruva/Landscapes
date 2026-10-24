
// Fog application and far sky-fade structure adapted from Farseer's region.fsh
// (github.com/ViciousBadger/VSMod-Farseer, MIT, (c) Badgerson). Unlike Farseer we
// have real per-vertex surface colors, shaded with screen-space-derivative normals.

in vec4 worldPos;
in vec4 vertexColor;
in float yLevel;
in vec4 rgbaFog;
in float dist;
in float radialDistance;
in float fogAmount;
in float edgeFade;
in vec3 tint;
in float tintAlpha;
in float snowLineAtVertex;
in float frostAtVertex;
in vec3 terrainPos;
in vec3 sectionLocal;

uniform float fogDensityIn;
uniform float fogMinIn;
uniform float horizonFog;
uniform vec3 sunPosition;
uniform vec3 sunColor;
uniform float dayLight;
uniform float cacheHandoffDistance;

// Which quarters of this section a coarse stand-in covers: one bit per child section,
// qz * 2 + qx. Zero is the whole section - every ordinary draw, and what a program that never
// sets it reads. Only the passes drawn section by section set it; opaque ground is limited by
// selecting its clusters instead. See LodStandInPolicy and lodownershiptest.fsh.
uniform int vhStandInQuarters;

// Per-cell vanilla ownership. One texel per 32x32x32 vanilla render chunk, stored as Y
// slices stacked down a single 2D texture because the public client API binds 2D textures
// only. maskEnabled stays 0 until the chunk mask is both requested and uploaded, so the
// default path is exactly the distance handoff above. Addressing must match
// VanillaReadinessMask.TexelIndex; a static check holds the two together.
uniform sampler2D readinessMask;
uniform int maskEnabled;
uniform int maskMinX;
uniform int maskMinZ;
uniform int maskWidth;
uniform int maskDepth;
uniform int maskCapacity;
uniform int maskVerticalChunks;

// 1 paints hidden pixels red instead of hiding them. Diagnostic only.
uniform int maskDebug;

// ---- Cached water ------------------------------------------------------------
//
// An optional near-field cutoff for cached WATER alone, measured in whole vanilla CHUNKS
// around the camera's own chunk, and OFF by default.
//
// No radius. A per-pixel distance test is a circle, a circle is a shape the world does not
// contain, and it reads as an effect painted over the terrain rather than as terrain - the
// same rule the blend band is built on. 0.3.141 used a radius and drew a hard ring across
// every lake; 0.3.144 shrank it and drew a smaller one. So this is a Chebyshev distance in
// 32-block chunk columns: its boundary can only ever run along chunk edges, in the game's own
// grid, and at the default of zero it does not exist at all.
//
// The boundary between cached water and vanilla's is the ownership mask below - vanilla's
// real chunk footprint - exactly as it is for solid ground. This is only a fallback for the
// near field, where the mask can be slow to commit and the engine's own water is transparent
// enough for a cached sheet under it to tint it. See .dlwater.
uniform int waterNearChunks;

// The camera's own vanilla chunk column, as two integer scalars. Never a Vec2i: the client's
// only integer-pair setter reaches glUniform2f, which an integer uniform rejects outright, so
// the value would silently stay at zero. See G42.
uniform int camChunkX;
uniform int camChunkZ;

// Cached water obeys the ownership mask exactly as solid ground does, at every distance.
// There is no outward allowance and no radius: an allowance measured from the camera is a
// circle, and a circle drew cached water straight into chunks vanilla was already drawing.
// The boundary is the mask's own 32-block cells, so it falls on vanilla chunk edges and
// nowhere else.

// 1 gives cached water the engine's own opacity instead of a flat constant. chunkliquid.vsh
// grades water by view angle - about 16% opaque looking straight down at a pond at your feet,
// clamped fully opaque by roughly 25 blocks out - so a flat 0.66 is simultaneously too opaque
// up close and too transparent at the seam, where it let the cached lakebed show through a
// surface vanilla draws solid. This is a property of the surface, not a correction, so it is
// applied everywhere rather than on the band: band-weighting it would create a fresh seam of
// its own.
uniform int waterFresnel;

// The two surface terms vanilla's water has and cached water did not, both weighted by the
// SAME blend band the ground uses - full at the vanilla seam, relaxing outward in the band's
// own 8-block steps.
//
// glint is the specular sun reflection on a noise-perturbed surface normal. It is the loudest
// single difference on a sunny day: the game's water is shiny to the horizon and ours went
// matte at the line. foam is vanilla's open-water surface noise, which is the same argument
// bandSpeckle makes for ground - vanilla's water is never one flat colour.
//
// Band-weighted rather than global because these are the expensive per-pixel terms, because
// past the band fog dominates and the detail is sub-pixel anyway, and because the whole point
// is to match vanilla where vanilla is standing next to us.
//
// What is NOT reproduced here is vanilla's shoreline rim, which it derives by sampling the
// depth texture during OIT. This shader runs in the opaque stage where that same depth buffer
// is the live render target, and sampling an attachment while it is bound is undefined - so
// shore foam has to come from mesh data instead, and does not exist yet.
uniform float waterGlint;
uniform float waterFoam;

// How shallow this shader should assume cached water is, 0 deep to 1 at the waterline.
//
// Vanilla's foam term is a SHALLOW-water effect and nothing else. It reads the depth buffer to
// find how far the bed is behind the surface, caps that at 0.05 and multiplies by 1500, so any
// water more than a block or two deep drives the term to exactly zero: `clamp(1 - diffTotal *
// 1500 - ...)`. Foam is what vanilla draws at a shoreline, not on open water.
//
// This shader has no depth buffer to read, and for three builds it substituted ZERO for that
// term - "infinitely shallow" - which pinned the effect at its maximum over every lake and
// ocean in the world. It adds about `bright * 0.3`, so cached water was carrying roughly a
// third of its own brightness as permanent shoreline foam, which is most of why it measured
// 1.7x vanilla's across the seam and read as flat bright paint.
//
// Zero is vanilla's answer for the water this actually draws: 60% of submerged ground in the
// owner's cache stands under 8 blocks or more and 38% under 24 or more. Real shoreline foam
// needs the depth per fragment, which the water mesh does not carry yet; until it does, being
// right about the open water is better than being wrong about all of it.
uniform float waterShallowness;

// The highest PRE-TEXTURE Fresnel alpha cached water may reach, 0.05 to 2. Vanilla clamps
// this intermediate value to 2 before climate-mapped texture alpha, foam and atmospheric fog
// turn it into the final 0..1 material alpha.
//
// Vanilla's own fresnel saturates: past roughly 200 blocks its water is fully opaque, and ours
// agrees with it because it is the same formula. Vanilla gets away with that because it stops
// drawing water at 80% of the view distance and you never see the far end - this mod draws it
// all the way out, so this cap preserves some bed visibility before atmospheric fog takes over.
//
// The bed beneath is now shaded by the water standing over it, one light level per block, so
// the depth information the eye reads in vanilla's water is already sitting under ours. Holding
// a little material transparency back is what lets it through, and it costs no geometry:
// painting the same gradient onto the surface instead breaks a lake into one quad per depth step and
// measured +75% of the water buffer.
uniform float waterMaxAlpha;

// Match chunkopaque's ground treatment beneath the engine's liquid-depth image.
// Independent of mesh depth shading and surface effects; .dlwater murk is a live A/B.
uniform int waterGroundMurk;
uniform int waterSurfaceVisible;

// The standard water tile from the game's live terrain atlas. Cached water previously used
// only the tile's average colour and omitted its alpha entirely. The dedicated surface shader
// applies the same stable block-tiled detail and alpha modulation as vanilla. The tint-slot
// check prevents the shared translucent buffer from putting this texture on lava or ice.
uniform int waterTextureEnabled;
uniform int waterTextureSlot;
uniform int waterTextureTintSlot;

// Engine state the liquid shader uses, uploaded under mod-owned names so nothing here can
// collide with a uniform the engine manages on its own programs. All are public API:
// ShaderUniforms.SunPosition3D / SunSpecularIntensity / WaterWaveCounter / WindWaveCounter /
// WaterFlowCounter / WindSpeed / PlayerPosForFoam, and Calendar.ReflectColor.
uniform vec3 vhSunPosRel;
uniform vec3 vhReflectColor;
uniform float vhSunSpecular;
uniform float vhWaterWave;
uniform float vhWaterStill;
uniform float vhWindWave;
uniform float vhWaterFlow;
uniform float vhWindSpeed;

// The engine's own foam anchor: an absolute world position re-based to a reference point that
// only moves when the player leaves a 200-block radius. Using it rather than a raw world
// coordinate is what keeps the noise both stable under the player and precise at map edges -
// and it puts our noise in exactly the phase vanilla's water is in.
uniform vec3 vhFoamOrigin;
// -------------------------------------------------------------------------------

// ---- Blend band --------------------------------------------------------------
//
// Cached terrain immediately outside the vanilla seam is made to resemble vanilla, relaxing
// back to its own appearance over a band, so the eye has no single line to find.
//
// The difference is a TEXTURE difference, not a colour one, and two attempts to treat it as a
// colour failed in play before that stuck. Vanilla terrain is never a single flat colour:
// every block shows a texture tile. Cached terrain is one averaged colour per block. Matching
// the average cannot close a difference that is not in the average, which is why a per-channel
// gain and then a darken/saturate pair each looked right on the terrain they were tuned on and
// worse than nothing on the next one. bandSpeckle below is the term that addresses the actual
// defect; darken and saturate remain, defaulting to neutral, because the average may still be
// slightly off once the variation is there.
//
// The band spreads what remains over many small steps; it does not delete it.
//
// One texel per 8x8 block cell, holding distance from the nearest vanilla-owned ground in
// cells. 8 blocks is a quarter of a vanilla chunk: fine enough to read as a smaller
// continuation of the game's blocky idiom, and never finer than one world block, which is
// the rule an earlier smooth radial transition broke - a gradient below the world's own grid
// reads as a GPU effect painted over the terrain rather than as terrain. Nothing here is a
// function of distance from the camera. Addressing must match
// VanillaSeamDistanceField.TexelIndex.
uniform sampler2D seamField;
uniform int bandEnabled;
uniform int bandMinCellX;
uniform int bandMinCellZ;
uniform int bandWidthCells;
uniform int bandDepthCells;
uniform int bandCapacity;

// Band reach in blocks, the ramp's step size in blocks, and how much of the correction to
// apply at the seam. Step is quantised no finer than the stored cell.
uniform float bandWidthBlocks;
uniform float bandStepBlocks;
uniform float bandStrength;

// How far the measured seam sits inside the seam a player can actually see, in blocks.
// Ownership is deliberately denied for the outermost chunk and a half of live vanilla
// terrain - the engine range-culls whole meshes and can stop drawing a chunk it still
// reports - so the field's zero is that much short of where vanilla's ground really ends.
// Without this the full correction would land underneath live vanilla, where nothing can
// see it, and the visible seam would get only part of it.
uniform float bandInnerBlocks;

// Two operators applied at full weight, both functions of the pixel rather than assertions
// about it. Darken is the measured luminance ratio at the seam; saturate is how much further
// from grey vanilla sits once both sides are normalised by their own luminance. A per-channel
// red/green/blue gain was tried first and failed in play: three multipliers tuned on grass
// encode grassland's own colour, so carrying them to sand applied grassland's correction to a
// colour that was never grassland.
uniform float bandDarken;
uniform float bandSaturate;

// The one that matters. Vanilla terrain is never a single flat colour: every block shows a
// texture tile, and at seam distances that reads as fine per-block speckle. Cached terrain is
// one averaged colour per block with nothing on top of it, so the transition is speckled to
// perfectly flat, and no adjustment to the average can close a difference that is not in the
// average. This puts the variation back - not by sampling vanilla's textures, which cached
// meshes carry no identity to look up, but by reproducing their statistics: one random value
// per world block, faded out across the band so it hands over to flat cached terrain
// gradually instead of at a line.
//
// Per BLOCK, never finer. Vanilla's own texture detail is below a pixel at this range anyway,
// and anything varying below the world's grid reads as an effect painted over the terrain.
uniform float bandSpeckle;

// 1 paints the band's weight instead of correcting it, so its shape can be judged from above
// before any colour decision is made. Diagnostic only.
uniform int bandDebug;

// The game's own block atlas, sampled directly. Not a recreation of vanilla's textures and
// not another approximation of their average - the real texels, on cached ground, faded in
// across the seam band by the same weight above.
//
// The rectangle arrives flat from the vertex stage; it is constant across a quad, and a quad
// can span forty blocks, so texture coordinates are derived from world position here rather
// than stored. That is also why the wrap needs explicit gradients: fract() tears once per
// block, and a GPU left to infer the derivative from it picks the coarsest mip on every
// block boundary and draws a grid of grey lines.
uniform sampler2D blockAtlas;
// Width and height in pixels. Not one number: a fresh install's block atlas is 4096 x 2048.
uniform vec2 blockAtlasSize;
uniform sampler2D blockTable;

uniform int blockTextureEnabled;
uniform float blockTextureStrength;
uniform float blockTextureMipLimit;

// How many mip levels SHARPER than the distance warrants to sample.
//
// Correct minification is why a faithful texture does nothing here: shrink a 32-pixel tile
// into three screen pixels and the hardware hands back that tile's average, which - since the
// texture is applied as variation around the block's colour - multiplies by one and changes
// nothing at all. The detail is thrown away before it can be seen, and this reaches back up
// the mip chain for it. Zero is physically correct and invisible; higher is deliberately
// wrong, trading aliasing for detail the eye can actually resolve.
uniform float blockTextureDetail;

// The block table, read here rather than in the vertex stage. Its rows are needed only inside
// the blend band, and the band is a ring of the screen against millions of vertices, so the
// fetches belong on the side that actually asks. -1 in the slot means the quad carries a
// plain colour and there is nothing to read.
flat in int vhBlockSlot;
// The quad's plane where the vertex stage knows it (see lodvertexinputs.vsh), 0 where it does not.
flat in int vhFaceAxis;
// ------------------------------------------------------------------------------

// 1 applies vanilla's own rule that an up-facing surface never darkens with the sun.
uniform int flatTopLight;

// The four terms below each close one measured gap between this shader and vanilla's
// chunkopaque pair, and each has its own switch so a single dusk-to-night sweep can
// attribute what it sees. See .dllight. All default to on.
//
// 1 shades by the engine's own light vector, which follows the sun by day and swings
// to the moon once moonlight beats sunlight; 0 keeps shading by the sun all night.
// lightPosition is declared by fogandlight.fsh and uploaded by the engine on every
// Use(), so it costs nothing to read here.
uniform int lightMoonDir;

// 1 uses vanilla's ramp and floor, max(0.45, 0.5 + 0.5 * dot); 0 keeps the mod's
// 0.55 + 0.45 * max(0, dot), which is 22% too bright on every away-facing slope.
uniform int lightVanillaRamp;

// 1 lights by the engine's blended ambient colour, which is what vanilla's terrain
// actually multiplies by; 0 keeps sunColor * dayLight. This is the dominant term:
// vanilla builds its ambient from ReflectColor, which is floored at a blue night
// colour once the sun is down, while sunColor stays on the orange end of the sunlight
// ramp. Measured offline, the two hues invert after sundown - vanilla goes blue-grey
// and this shader goes orange.
uniform int lightAmbientColor;
uniform vec3 rgbaAmbientIn;

// 1 applies vanilla's daylight brightening, 1 + max(0, shadowIntensity * 2 - 1.66)/1.5.
// shadowIntensity is the engine's DropShadowIntensity, which is 1 while the sun is high
// and falls to 0 as it sets, so this is worth 22.7% extra brightness at midday and
// nothing at dusk. Also declared and uploaded by fogandlight.fsh.
uniform int lightDayBoost;

// Vanilla's baked SUN LIGHT LEVEL, and the largest single error this shader has carried.
//
// applyLight multiplies by `lightColor.a`, the sun light level baked into every terrain
// vertex. The derivation that produced the ambient term below read the LOD case as "full
// sky light" and cancelled that factor as if it were 1. It is not: `bMax` renormalisation
// cancels the ambient colour's own mean, not the sun level, so the level survives and
// multiplies the pixel. Open sky is not level 31 either - it is the world's `sunBrightness`,
// 22 in the base game, which is 0.806 on the engine's ramp and 206/255 = 0.808 once the
// engine has quantised it into the vertex byte the shader actually reads.
//
// So every cached surface was 1/0.808 = 1.24x too bright, at every hour, on every block.
// Measured on a flat two-block test patch at the seam: cached granite gravel came out 1.39x
// vanilla's, of which 1.07x was the stored colour and the remaining 1.29x was this.
//
// Uploaded as 1.0 when `.dllight sun` is off, which is exactly the old behaviour and costs
// no branch. Only the vanilla-ambient path takes it: the sunColor fallback beside it is the
// pre-0.3.51 model kept for comparison, and scaling that would make it a third thing that
// was never either.
uniform float sunLightLevel;

// The client's minbrightness setting, used as a floor for the lighting term below.
uniform float minBrightness;

// 1 mixes the far dissolve band toward the sky the engine actually draws. Every engine
// shader that calls getSkyColorAt passes SkyDaylight, not DayLightStrength; the two
// differ by a factor of 1.25 and by a night floor, so the band this shader fades terrain
// into did not match the sky behind it, and the mismatch moved with the time of day.
// SkyDaylight is internal to the engine, but every input to it is public, so the renderer
// reconstructs it. See .dllight sky.
uniform int lightSkyDayLight;
uniform float skyDayLight;

// P4 horizon handoff. Zero preserves the established sky fade byte-for-byte; one changes only
// its target to the exact colour function used by the silhouette pass.
uniform int silhouetteEnabled;
uniform float silhouetteSkyTint;
uniform vec4 silhouetteColorTint;
uniform float silhouetteLightLevelBias;
uniform int silhouetteSeaLevel;

// ---- Per-section facts ------------------------------------------------------
//
// The two values below describe the section rather than the fragment. The established
// path sets each as a uniform before that section's draw; the indirect variant has no
// per-draw uniforms and receives them from the vertex stage as flat varyings, filled
// from LodGpuSectionRecord. The #defines keep the body below identical in both
// variants. See the matching block in lodterrainbody.vsh.
#ifdef VH_INDIRECT
flat in float vhColumnBlocks;
flat in ivec2 vhMaskSectionOrigin;

#define columnBlocks vhColumnBlocks
#define maskSectionOriginX vhMaskSectionOrigin.x
#define maskSectionOriginZ vhMaskSectionOrigin.y
#else
// This section's origin in whole vanilla chunks. Section origins are multiples of the
// chunk size, so this is exact, and adding a small local offset to it cannot round.
//
// Two scalars rather than one ivec2 on purpose. The client's only vector setter for a
// pair of integers is Uniform(name, Vec2i), which calls glUniform2f; against an integer
// uniform that is GL_INVALID_OPERATION and the value never arrives at all. See G42.
uniform int maskSectionOriginX;
uniform int maskSectionOriginZ;

// Blocks per column in the section being drawn (1 at level 0, doubling per level).
// Coarse sections merge whole neighbourhoods into one colour, and greedy meshing
// then fuses them into large single-colour quads; a little world-space variation
// scaled to the column size breaks those plates up without inventing detail.
// Scaling by column size is what keeps the pattern roughly constant on screen
// instead of aliasing into shimmer at distance.
uniform float columnBlocks;
#endif
// -----------------------------------------------------------------------------

// Live tint table. The alpha byte carries a tint SLOT plus a blend band:
//   0..63    opaque,     slot = alpha
//   64..127  water,      slot = alpha - 64
//   128..191 thin plant, slot = alpha - 128
// Slot 0 is the identity tint. One slot per distinct (climate map, season map) pair,
// because leaves pick a seasonal map per species and water has its own -- a single
// shared foliage tint left every tree the same colour and water untinted grey.
// Sampled at two heights and blended by vertex height: the climate maps are indexed
// by temperature, which drops with altitude, so one sample at the player's feet gave
// mountaintops the same lush green as the valley floor.
// Must equal LodTintRegistry.MaxSlots; LoadShader logs an error if it does not.
const int TINT_SLOTS = 64;

// Blend factors per band, now that alpha carries the slot instead of an opacity.
// Flowers are crossed quads in vanilla; as a solid cube they read as a grey blob, so
// they are drawn mostly see-through and the ground shows through them.
const float WATER_ALPHA = 0.66;
const float THIN_ALPHA = 0.50;

#ifdef VH_WATER_OIT
// Resolve the same shaded result through the engine's six-attachment contract.
// The macro follows the include so OIT() still writes the real glow attachment.
#include oit.fsh
vec4 outColor;
vec4 vhWaterGlow;
#define outGlow vhWaterGlow
#if SSAOLEVEL > 0
vec4 outGNormal;
vec4 outGPosition;
#endif
#else
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outGlow;
#if SSAOLEVEL > 0
layout(location = 2) out vec4 outGNormal;
layout(location = 3) out vec4 outGPosition;
#endif
#endif

#include noise3d.ash
#include dither.fsh
#include fogandlight.fsh
#include skycolor.fsh
#include underwatereffects.fsh
#include lodsilhouettecolor.fsh

// Immutable vanilla-only depth, never the attached liquid-depth destination. The depth
// and OIT surface passes both read the same snapshot so a clear boundary pixel cannot get
// cached murk coverage without the cached surface that belongs over it.
uniform sampler2D vanillaWaterDepth;
uniform int waterJunctionDepth;

// Full-resolution OIT revealage copied after vanilla liquid and before cached water. The
// copy is immutable while this shader draws, so it is safe to sample even though the live
// revealage texture remains attached to the transparent framebuffer. It closes only actual
// surface-coverage holes in an owned boundary cell; ordinary vanilla pixels still win.
uniform sampler2D vanillaOitRevealage;
uniform int waterJunctionSurface;
