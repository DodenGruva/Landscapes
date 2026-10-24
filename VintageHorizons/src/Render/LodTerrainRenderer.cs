using System.Diagnostics;
using System.Text;
using OpenTK.Graphics.OpenGL4;
using Vintagestory.API.Client;
using Vintagestory.API.Common;
using Vintagestory.API.Common.Entities;
using Vintagestory.API.Datastructures;
using Vintagestory.API.MathTools;
using Vintagestory.Client.NoObf;

namespace VintageHorizons;

/// <summary>
/// Renders the LodWorld section pyramid beyond the vanilla view distance. Meshes are
/// built off-thread by the LodWorker from section snapshots; this class schedules
/// mesh jobs (nearest-first), uploads finished vertex data on the render thread, and
/// walks the quadtree each frame picking detail by distance - a parent renders until
/// all four child slots are covered, so level swaps never open holes (DH's rule).
///
/// Rendering techniques (render order/stage, ZFar extension, camera-relative model
/// matrices, fog + transition handling in the shaders) adapted from Farseer
/// (https://github.com/ViciousBadger/VSMod-Farseer, MIT, (c) Badgerson).
/// </summary>
public partial class LodTerrainRenderer : IRenderer
{
    // Development contract: the shipping candidate must exercise the renderer being built.
    // A missing prerequisite produces missing cached geometry and an error instead of quietly
    // substituting the per-section renderer and making a broken path look healthy.
    internal const bool RequirePackedTerrainRenderer = true;

    internal const double PreVanillaRenderOrder = 0.36;
    internal const double PostVanillaRenderOrder = 0.38;
    public double RenderOrder => postVanillaDepthCulling
        ? PostVanillaRenderOrder
        : PreVanillaRenderOrder;
    public int RenderRange => 9999;

    const int MeshSchedulesPerFrame = 4;
    const int MeshUploadsPerFrame = 4;
    // Initial frame-local ceilings. Item caps remain as a final safety rail, while live
    // telemetry records enough time/byte/backlog evidence to tune these without guessing.
    internal const long MeshSnapshotMaxBytesPerFrame = 2 * 1024 * 1024;
    internal const double MeshSnapshotMaxMillisecondsPerFrame = 1.0;
    internal const long MeshUploadMaxBytesPerFrame = 4 * 1024 * 1024;
    internal const double MeshUploadMaxMillisecondsPerFrame = 2.0;
    internal const double SafeTemporalTranslationBlocks = 0.25;
    internal const float SafeTemporalRotationMatrix = 0.0005f;
    internal const int ReadinessProbeMaxItemsPerFrame = 256;
    internal const double ReadinessProbeMaxMillisecondsPerFrame = 0.25;

    /// <summary>
    /// Backlog ceilings. A settled view empties its queue every frame and never reaches
    /// the ordinary limits, so those limits only ever bind while the window is moving -
    /// which is precisely when unowned ground shows cached terrain over real terrain. A
    /// probe costs well under a microsecond, so the item cap, not the clock, was ending
    /// ordinary frames early. When the queue is genuinely deep the budget opens up and the
    /// elapsed-time ceiling becomes the real guard.
    /// </summary>
    internal const int ReadinessProbeCatchUpItemsPerFrame = 1024;
    internal const double ReadinessProbeCatchUpMillisecondsPerFrame = 1.0;
    internal const int ReadinessProbeCatchUpQueueDepth = 512;
    internal const int ReadinessSeedCellsPerFrame = 256;
    internal const int ReadinessBoundaryCellsPerFrame = 128;

    /// <summary>
    /// Cells examined in the loss-detection shell during the frame the camera crosses a
    /// chunk boundary, which is the moment vanilla's own unload radius has just swept past
    /// ground we may still be marking as vanilla-owned.
    ///
    /// The ordinary per-frame scan walks a cursor and takes tens of frames to circle the
    /// camera. Flying backwards outruns it: chunks unload in front of the eyes while their
    /// cells still read owned, so the mask keeps suppressing cached terrain that nothing
    /// else is drawing and a band of world goes missing. A crossing happens once every 32
    /// blocks of travel, so completing the sweep then is affordable, and it costs a queue
    /// of probes rather than the probes themselves.
    /// </summary>
    internal const int ReadinessBoundarySweepOnCrossing = 4096;

    /// <summary>
    /// How often every committed cell is re-confirmed, in milliseconds. This is the upper
    /// bound on how long the mask can hide cached terrain that vanilla has stopped drawing,
    /// and it holds however the camera moved. Cursor sweeps could not provide that bound:
    /// each one restarted or fell behind under movement, and holes outlived the movement.
    /// </summary>
    internal const long ReadinessFullRevalidateMilliseconds = 1000;
    internal const int ReadinessInteriorCellsPerFrame = 32;
    internal const int ReadinessGuardChunks = 2;

    /// <summary>
    /// How long a chunk the engine has just announced is rechecked every frame, and how many of
    /// those rechecks one frame may start. See <see cref="VanillaRenderReadiness.Watch"/>.
    ///
    /// Ten seconds covers a tesselation queue backed up by exploring - most chunks are handed
    /// over within a few frames of their mesh reaching the pool, and a buried chunk leaves the
    /// list at its first probe after upload - and anything still waiting past it returns to the
    /// ordinary sweep. Sixty-four rechecks cost a few tens of microseconds and go first in the
    /// probe budget, so the sweeps behind them keep most of it.
    /// </summary>
    internal const long FreshChunkWatchMilliseconds = 10_000;
    internal const int FreshChunkWatchesPerFrame = 64;

    /// <summary>
    /// Pulled back from the nearest column that is not wholly owned. A cell becomes ready
    /// when two render-frame-separated probes agree, which is a tessellation signal rather
    /// than proof that vanilla's replacement mesh is on screen, so the handoff stops one
    /// vanilla chunk short of the boundary it measured.
    /// </summary>
    internal const float ReadinessHandoffMarginBlocks = 32;

    /// <summary>
    /// Camera movement that re-measures owned ownership. Well under one vanilla chunk, so
    /// the radius cannot lag the camera by a whole ownership cell.
    /// </summary>
    internal const float ReadinessHandoffRecheckBlocks = 4;

    /// <summary>
    /// Cached terrain stays suppressed this close to the camera even under the per-cell
    /// mask, but only while the camera's own cell is committed vanilla ready.
    ///
    /// The mask decides ownership per cell, so a cell vanilla has not proven - usually an
    /// underground one whose chunk never reports rendered - keeps drawing cached terrain.
    /// That is correct at distance and wrong in the player's face: coarse cached geometry
    /// at arm's length reads as a wall through the world. The old radius hid this by
    /// suppressing everything near the camera unconditionally, which is also what made it
    /// open holes. Requiring the camera's own cell to be owned keeps the near-field clean
    /// without reintroducing that: if vanilla is not drawing where the player stands,
    /// nothing is suppressed at all.
    /// </summary>
    internal const float MaskNearFloorBlocks = 48;

    /// <summary>
    /// What the floor above defaults to now: nothing.
    ///
    /// The owner asked for it removed. It exists to stop coarse cached geometry appearing at
    /// arm's length in cells vanilla has never proven - typically underground, where a chunk
    /// never reports itself rendered - but it also hides cached terrain from anyone who turns
    /// the vanilla view distance down far enough to want to look at it, which is the whole
    /// direction this mod exists to enable. Kept as a knob rather than deleted so the older
    /// behaviour is one command away if the near-field artefact it guarded against returns.
    /// </summary>
    internal const float MaskNearFloorDefaultBlocks = 0;

    /// <summary>
    /// About twenty 8-block steps. Sized from the measured seam step of roughly 21 colour
    /// levels against a flat-field detection threshold of one to two levels per step.
    /// </summary>
    internal const float BandDefaultWidthBlocks = 128;

    /// <summary>
    /// How often the seam field may be recomputed while ownership churns. A moved window
    /// rebuilds immediately regardless - the wrapped ring hands a reused slot to different
    /// world coordinates, so a stale window would address another place's distance - but
    /// ordinary ownership traffic can wait. A tenth of a second of lag on a colour ramp is
    /// well under a walking pace, and it cannot open a hole because the band decides no
    /// pixel's visibility.
    /// </summary>
    internal const long BandRebuildIntervalMilliseconds = 100;

    /// <summary>
    /// The widest seam field allowed, in 8-block cells per axis. 512 is a 1 MiB texture and
    /// a quarter-million-cell sweep; beyond that the band turns itself off rather than
    /// quietly spending a frame budget nobody approved. Reached only at vanilla view
    /// distances far past where the band's imitation still works at all.
    /// </summary>
    internal const int BandMaxCapacityCells = 512;
    /// <summary>
    /// Queue depth allowed at the mesh workers. Per thread, not absolute: a fixed 12 was
    /// sized for one builder and would leave a four-thread pool idling three quarters of
    /// the time. Deep enough that a thread finishing a job always has another waiting,
    /// shallow enough that the queue does not outlive the view that asked for it.
    /// </summary>
    const int MeshBacklogPerThread = 4;
    int maxWorkerMeshBacklog;

    /// <summary>Reload requests per frame; only enqueues a key, so it can far exceed the mesh budget.</summary>
    const int MeshLoadRequestsPerFrame = 32;

    readonly ICoreClientAPI capi;
    readonly LodWorld world;
    readonly LodWorker worker;
    readonly LodGpuTelemetry gpuTelemetry;
    readonly Func<long> currentWorldEpoch;
    readonly string? gpuRendererPreference;
    readonly LodGpuFailureInjection gpuFailureInjection;
    readonly LodGpuShadowRenderPath gpuShadow;
    readonly LodRenderPathCoordinator renderPaths;
    bool renderPathConfigured;
    string? requiredTerrainCapabilityFailure;
    bool reportedGpuFrameSizeDisagreement;
    float renderCullDistanceSquared = float.MaxValue;
    /// <summary>
    /// What the renderer's own frame costs, by phase, since the last stats report. Each
    /// of these is tens of microseconds against a ten-millisecond frame, which is well
    /// under what any frame-rate comparison can resolve, so they are timed rather than
    /// inferred. Reported by .dlinfo and by the periodic stats line.
    /// </summary>
    public LodPhaseCost PruneCost, ScheduleCost, UploadCost, GlUploadCost, MeshDisposeCost,
        EvictCost, SeasonalCost, FarDistanceCost, ReadinessCost, WalkCost, DrawCost;
    public bool TrackPhaseAllocations { get; set; }

    public int ProjectionResetCount { get; private set; }
    public long MeshUploadBytes { get; private set; }
    public long MeshSnapshotBytes { get; private set; }
    public int MeshSnapshotItems { get; private set; }
    public int MeshUploadItems { get; private set; }
    public int StaleMeshResultsRejected { get; private set; }
    public long OpaqueDrawCalls { get; private set; }
    public long WaterDrawCalls { get; private set; }
    public long OpaqueDrawVertices { get; private set; }
    public long OpaqueDrawIndices { get; private set; }
    public long WaterDrawVertices { get; private set; }
    public long WaterDrawIndices { get; private set; }
    public long LiveGpuMeshBytes { get; private set; }
    public long LiveOpaqueVertices { get; private set; }
    public long LiveOpaqueIndices { get; private set; }
    public long LiveWaterVertices { get; private set; }
    public long LiveWaterIndices { get; private set; }
    public bool GpuTimingActive => gpuTelemetry.TimingActive;
    public bool GpuTimingRequested => gpuTelemetry.TimingRequested;

    /// <summary>
    /// Arms the delayed GPU timers for a session that did not start under the benchmark
    /// harness. The depth phases are gated on GPU time and nothing else, so a report that
    /// prints 0.0us because nobody set an environment variable is indistinguishable from a
    /// pass. Returns whether timing is on now; see LodGpuTelemetry.DescribeTiming for why
    /// not.
    /// </summary>
    public bool EnableGpuTiming() => gpuTelemetry.RequestTiming();

    /// <summary>Plain-language state of the GPU timers, for any report that prints their figures.</summary>
    public string DescribeGpuTiming() => gpuTelemetry.DescribeTiming();

    public LodPhaseCost GpuOpaqueCost => gpuTelemetry.OpaqueCost;
    public LodPhaseCost GpuSplitNearCost => gpuTelemetry.SplitNearCost;
    public LodPhaseCost GpuSplitFarCost => gpuTelemetry.SplitFarCost;

    /// <summary>GPU time to copy the depth buffer and reduce every pyramid level.</summary>
    public LodPhaseCost GpuHzbCost => gpuTelemetry.HzbGpuCost;
    public LodPhaseCost GpuSplitHzbCost => gpuTelemetry.SplitHzbGpuCost;

    /// <summary>
    /// GPU time for the classification dispatch alone. Separate from the build because
    /// only this one scales with the sampling width, and the width was widened ninefold
    /// in fetch count; a combined figure could hide that completely.
    /// </summary>
    public LodPhaseCost GpuClassifyCost => gpuTelemetry.ClassifyGpuCost;
    public LodPhaseCost GpuOrdinaryCullCost => gpuTelemetry.OrdinaryCullGpuCost;
    public LodPhaseCost GpuSplitNearCullCost => gpuTelemetry.SplitNearCullGpuCost;
    public LodPhaseCost GpuSplitFarCullCost => gpuTelemetry.SplitFarCullGpuCost;
    public LodPhaseCost GpuWaterCost => gpuTelemetry.WaterCost;
    public int GpuTimerPendingResults => gpuTelemetry.PendingResults;
    public int GpuTimerUnavailableSlots => gpuTelemetry.UnavailableSlots;
    public int GpuTimerTargetBusy => gpuTelemetry.TargetBusy;
    public long TemporalOcclusionQueries { get; private set; }
    public long TemporalOcclusionResults { get; private set; }
    public long TemporalOcclusionHiddenResults { get; private set; }
    public long TemporalOcclusionStaleResults { get; private set; }
    public long TemporalOcclusionGlobalInvalidations { get; private set; }
    public long TemporalOcclusionDrawsSkipped { get; private set; }
    public int LastTemporalOcclusionDrawsSkipped { get; private set; }
    public int LastTemporalOcclusionSeamDraws { get; private set; }
    public int LastTemporalOcclusionEdgeDraws { get; private set; }
    public long ReadinessProbes { get; private set; }
    public long ReadinessTrueResults { get; private set; }
    public long ReadinessFalseResults { get; private set; }
    public long ReadinessReadyTransitions { get; private set; }
    public long ReadinessLostTransitions { get; private set; }
    public long ReadinessProbeErrors { get; private set; }
    public long ReadinessWindowChanges { get; private set; }
    public long ReadinessResizes { get; private set; }
    public long ReadinessFullRevalidations { get; private set; }
    public long ReadinessAggregateRepairs { get; private set; }
    public long ReadinessStaleCommittedFound { get; private set; }
    public long ReadinessEmptyChunksSeen { get; private set; }
    public long ReadinessDrawnWithoutGeometry { get; private set; }
    public long ReadinessOwnedWithoutGeometry { get; private set; }
    public long ReadinessOwnershipDeniedNoGeometry { get; private set; }

    /// <summary>
    /// Probes refused ownership because the cell sits outside the engine's own draw range.
    /// A permanently non-zero figure here while standing still is the trailing annulus:
    /// chunks the player walked away from, still loaded and still reporting every "drawing"
    /// signal, that the engine range-culls at draw time.
    /// </summary>
    public long ReadinessOwnershipDeniedBeyondViewDistance { get; private set; }
    public long ReadinessMaskResyncs { get; private set; }

    /// <summary>
    /// Ownership decisions taken with the engine's own bounding sphere in hand, against those
    /// taken from the conservative column bound. Together they say whether the exact rule is
    /// actually carrying the boundary or is quietly unavailable: if `exact` stays at zero in a
    /// normal view, the reflection binding or the culler read has failed and the overlap is
    /// still the old 62 blocks wide, which nothing else on screen would reveal.
    /// </summary>
    public long ReadinessExactRangeDecisions { get; private set; }
    public long ReadinessBoundedRangeDecisions { get; private set; }

    /// <summary>
    /// Cells demoted by the per-frame range sweep rather than by a probe. This is the number
    /// that replaces <see cref="OverlapMarginBlocks"/>: each one is ground handed back to the
    /// cache in the same frame the engine stopped drawing it, which is the work the margin
    /// used to do in advance and approximately.
    /// </summary>
    public long ReadinessRangeSweepLosses { get; private set; }
    public long ReadinessRangeSweeps { get; private set; }

    /// <summary>Worst observed cost of one range sweep, in milliseconds.</summary>
    public double ReadinessRangeSweepPeakMs { get; private set; }

    /// <summary>
    /// Whether a chunk the engine claims must hold an uploaded terrain mesh before it may
    /// first own its cell. On by default and only meaningful with the chunk mask enabled.
    /// This is deliberately a gain gate, never a loss signal: mesh/cull state can be
    /// transient while vanilla replaces terrain, whereas a false IsChunkRendered result
    /// and the exact range clause are the tracker's reliable loss paths.
    /// </summary>
    public bool GeometryOwnershipRule { get; set; } = true;

    /// <summary>
    /// Whether ownership uses the engine's own bounding sphere for the range cull, instead of
    /// bounding where that sphere might be.
    ///
    /// <para>
    /// On is the exact rule and no overlap ring; off is 0.3.152's arithmetic bound and its
    /// 46-block sphere-slack ring, plus whatever overlap is set. A live toggle because the two
    /// produce a visibly different seam and the
    /// only way to compare them honestly is to stand at one and switch, and because it is the
    /// single switch that reverts this whole feature - the sweep below is gated on it too.
    /// </para>
    /// </summary>
    public bool ExactDrawRange { get; set; } = true;

    /// <summary>
    /// Whether a wholly owned cached section may be dropped before it is drawn at all.
    ///
    /// Ownership has two consumers and they fail differently. The shader discards single
    /// fragments in owned cells; this drops an entire cached section on the CPU. Both are
    /// live only with the mask on, so a hole that appears with the mask could come from
    /// either, and no amount of reasoning about ownership separates them once ownership
    /// itself is known good. Turning this off leaves the per-pixel mask working alone.
    /// </summary>
    public bool WholeMeshSkip { get; set; } = true;

    int waterNearChunks = LodNearHandoff.WaterNearChunks;

    /// <summary>
    /// How many vanilla chunk columns around the camera's own hide cached water outright.
    /// Zero, the default, is no near rule at all.
    ///
    /// Measured in CHUNKS and never in blocks, because the two builds that measured it in
    /// blocks both drew a circle. A per-pixel distance test is a circle by construction, and
    /// a circle is a shape the world does not contain: it reads as an effect painted over the
    /// terrain rather than as terrain, which is the exact rule the blend band was built on and
    /// the exact complaint that came back from play, twice. A Chebyshev count of 32-block
    /// columns can only ever put its boundary on a chunk edge.
    ///
    /// Off by default because the ownership mask is the real boundary and cached water no
    /// longer writes depth, so an uncommitted cell near the player costs a faint tint under
    /// the engine's own water rather than a flat sheet in front of it. Turn it on only if
    /// shallow water close to you looks murky, and then it is the smallest count that fixes
    /// it - each one is a whole 32-block ring.
    /// </summary>
    public int WaterNearChunks
    {
        get => waterNearChunks;
        set => waterNearChunks = Math.Clamp(value, 0, 16);
    }

    /// <summary>
    /// Give cached water the engine's own view-angle opacity instead of a flat constant.
    ///
    /// Cached water was 0.66 opaque at every distance and from every angle. Vanilla's is a
    /// grazing-angle term, transcribed in the fragment shader from `chunkliquid.vsh`, which
    /// is nearly transparent underfoot and fully opaque by a few tens of blocks. The flat
    /// constant was therefore wrong in both directions, and at the seam it was the visible
    /// one: the cached lakebed showed through a surface vanilla draws solid.
    ///
    /// Applied everywhere, deliberately NOT on the blend band. It describes the surface
    /// rather than correcting it, and fading it in across a band would put back a seam of
    /// exactly the kind this work removes. `.dlwater fresnel off` restores the constant.
    /// </summary>
    public bool WaterFresnel { get; set; } = true;



    float waterGlint = 1f;
    float waterFoam = 1.6f;

    /// <summary>
    /// How much of vanilla's specular sun reflection cached water carries, 0 to 1.
    ///
    /// The single loudest difference between the two waters on a sunny day: the game's water
    /// is shiny out to the horizon and cached water went matte at the line.
    ///
    /// Carried on EVERY cached water fragment. It was weighted by the blend band for one
    /// build, to bound the per-pixel cost, and that produced the defect it was meant to cure:
    /// the term faded out across the band and then stopped, leaving a line across distant
    /// water exactly where the band ended. Vanilla's water is shiny everywhere it draws, so
    /// carrying this everywhere is the more faithful answer as well as the one with no edge.
    /// </summary>
    public float WaterGlint
    {
        get => waterGlint;
        set => waterGlint = Math.Clamp(value, 0f, 1f);
    }

    /// <summary>
    /// How much of vanilla's open-water surface noise cached water carries, 0 to 2. One is
    /// the literal vanilla term; values above one are a bounded diagnostic gain for matching
    /// the small residual difference at the renderer handoff. Carried
    /// everywhere for the same reason as <see cref="WaterGlint"/>, and resting on the same
    /// argument <see cref="BandSpeckle"/> makes for ground: vanilla's water is never one flat
    /// colour, and no adjustment to an average can close a difference not in the average.
    ///
    /// This is vanilla's OPEN-water term only. Its shoreline rim is derived from the depth
    /// texture during OIT; this shader runs in the opaque stage where that buffer is the live
    /// render target, so the rim has to come from mesh data and does not exist yet.
    /// </summary>
    public float WaterFoam
    {
        get => waterFoam;
        set => waterFoam = Math.Clamp(value, 0f, 2f);
    }

    /// <summary>
    /// Paint fragments the mask would hide bright red instead of hiding them, and stop
    /// dropping whole owned sections so nothing escapes the paint. A gap that turns red is
    /// the mask's doing; a gap that stays empty never was, and no further argument about
    /// ownership is needed to tell them apart.
    /// </summary>
    public bool MaskDebugPaint { get; set; }

    /// <summary>
    /// Saved master switch for the optional seam-colour and L0 block-texture pipeline.
    /// Ownership masking and the constant overlap sink are deliberately outside this switch:
    /// they prevent holes and depth fighting rather than blending the two renderers' look.
    /// </summary>
    public bool BlendPipelineEnabled
    {
        get => blendPipelineEnabled;
        set => SetBlendPipelineEnabled(value);
    }

    /// <summary>
    /// Whether every resident L0 palette that predates an off-to-on transition has had its
    /// derived texture slot restored. False keeps mesh emission plain while the owning-thread
    /// pipeline performs its bounded backfill.
    /// </summary>
    public bool TextureSlotSourceReady
    {
        get => textureSlotSourceReady;
        set
        {
            if (textureSlotSourceReady == value) return;
            textureSlotSourceReady = value;
            if (!value) SetTextureSlotEmission(false);
        }
    }

    /// <summary>
    /// Wear vanilla's colouring at the seam and relax outward, so the eye has no single line
    /// to find. Measured offline at the seam, cached ground is about 8% brighter than vanilla
    /// and loses the green cast, with red and blue moving nearly twice as far as green - a
    /// surface missing vanilla's dark pixels rather than one with a wrong average. The band
    /// spreads that step over many small steps; it does not close it.
    /// </summary>
    public bool BandEnabled { get; set; } = true;

    /// <summary>
    /// How far past the seam the correction reaches, in blocks. Absolute rather than scaled
    /// by view distance: the step being hidden is a fixed number of colour levels wherever
    /// the seam happens to sit, so the number of steps needed does not change when it moves.
    /// About twenty 8-block steps for a measured step of ~21 levels, which puts each step
    /// near the eye's threshold on a flat field.
    /// </summary>
    public float BandWidthBlocks { get; set; } = BandDefaultWidthBlocks;

    /// <summary>
    /// Blocks per visible step in the ramp. Never finer than the stored 8-block cell, and
    /// deliberately never continuous: a ramp varying below one world block reads as a GPU
    /// effect painted over the terrain, which is what sank an earlier smooth radial version.
    /// </summary>
    public float BandStepBlocks { get; set; } = VanillaSeamDistanceField.CellBlocks;

    /// <summary>How much of the correction lands at the seam. 1 is the full measured ratio.</summary>
    public float BandStrength { get; set; } = 1f;

    /// <summary>
    /// How far the measured seam sits inside the one a player sees, in blocks.
    ///
    /// <para>
    /// The band measures from the ownership edge and has to correct terrain at the edge a
    /// PLAYER sees, so it carries whatever gap ownership leaves between the two. That gap is
    /// the refused ring, so this is derived from it rather than fixed: it was 46 blocks while
    /// ownership was bounded, and it is zero once the exact rule puts the ownership edge on
    /// vanilla's real one. A fixed 46 under the exact rule would start the blend 46 blocks
    /// PAST the seam it exists to hide.
    /// </para>
    /// <para>
    /// Negative means derive. `.dlband inner &lt;blocks&gt;` pins it and `.dlband inner auto`
    /// gives it back, because only looking at the seam can settle whether the derivation and
    /// the eye agree.
    /// </para>
    /// </summary>
    public float BandInnerBlocks
    {
        get => bandInnerBlocks >= 0f ? bandInnerBlocks : RefusedOwnershipRingBlocks();
        set => bandInnerBlocks = value < 0f ? -1f : value;
    }

    /// <summary>Negative derives from the refused ring; zero or more pins it.</summary>
    float bandInnerBlocks = -1f;

    /// <summary>Whether the band's inner shift is following the ownership rule or pinned.</summary>
    public bool BandInnerDerived => bandInnerBlocks < 0f;

    /// <summary>
    /// Brightness multiply at full weight. The measured seam is 144.5 luminance on vanilla
    /// against 159.1 on cached ground, so vanilla is about 9% darker.
    ///
    /// This replaced a per-channel red/green/blue gain, which failed in play for a reason
    /// worth keeping written down: three channel multipliers tuned on grass encode
    /// grassland's own colour, so carrying them to sand or rock applies grassland's
    /// correction to a colour that was never grassland. The owner saw exactly that. The two
    /// operators here are functions of the pixel instead of assertions about it, so sand
    /// stays sand.
    /// </summary>
    /// <summary>
    /// Per-block brightness variation restored at the seam, fading out across the band.
    ///
    /// This is the term that addresses the actual defect. Vanilla terrain is never one flat
    /// colour - every block shows a texture tile, which at seam distances reads as fine
    /// per-block speckle - and cached terrain is a single averaged colour per block. Two
    /// attempts to close that with an average-matching correction each looked right on the
    /// terrain they were tuned on and worse than nothing on the next one, because the
    /// difference was never in the average.
    ///
    /// Not a texture lookup: cached meshes carry a colour per corner and no block identity to
    /// look one up with. This reproduces the statistics instead, one random value per world
    /// block, which is also all the eye can resolve at this range.
    ///
    /// 0.15 down to 0.08 on the owner's call, 2026-08-31, after the three colour corrections
    /// of 0.3.169 and 0.3.170 landed. It was carrying more than its own share: with cached
    /// ground 24% too bright and its stored colours drawn from a thirty-pixel sample, extra
    /// per-block variation was helping to disguise a mismatch that is now fixed at source.
    /// Once the colour itself is right, the same amount of speckle reads as noise rather
    /// than as texture.
    ///
    /// And back to 0.15 on the owner's call, 2026-09-03, for a reason that reverses that one
    /// rather than repeating it. What made 0.15 read as noise was that it was the ONLY
    /// variation cached ground had: one flat colour per block type across a whole chunk, with
    /// speckle scattered over it. Distant terrain now carries the game's own per-position
    /// seasonal row (0.3.187, confined to level 0 in 0.3.193), so a single chunk holds real
    /// colour variety before this term is added at all - and against that, the same amount
    /// reads as texture again. The number is unchanged; what it sits on is not.
    /// </summary>
    public float BandSpeckle { get; set; } = 0.15f;

    /// <summary>
    /// How close to the camera cached terrain stops being drawn, in blocks, while the mask is
    /// live. Zero draws it right up to the camera. See <see cref="MaskNearFloorBlocks"/> for
    /// what the old 48-block floor was guarding against.
    /// </summary>
    public float NearSuppressionBlocks { get; set; } = MaskNearFloorDefaultBlocks;

    /// <summary>
    /// Draw near ground with the game's own block textures instead of one averaged colour,
    /// with the seam band deciding how much of it shows. The owner's judgement after three
    /// average-matching attempts: vanilla terrain is never a flat colour, so nothing that
    /// only adjusts the average can meet it.
    /// </summary>
    public bool BlockTexturesEnabled { get; set; } = true;

    /// <summary>How much real texture shows at full band weight. 1 is all of it.</summary>
    public float BlockTextureStrength { get; set; } = 1f;

    /// <summary>
    /// How far into the atlas mip chain a fragment may ask before the texture hands back to
    /// the averaged colour, in mip levels.
    ///
    /// Deep mips blend a block's tile into its neighbours' in the shared atlas, which would
    /// paint another block's colour onto this ground. But the seam sits hundreds of blocks
    /// out, where one block covers only a few pixels and a 32-pixel tile is already asking
    /// for mip three or four - so a cautious limit would fade the texture out at exactly the
    /// distance it exists for.
    ///
    /// Raised to six once the texture became variation around the block's colour rather than
    /// the colour itself, because that made overreaching self-limiting: a heavily minified
    /// texel approaches its own texture's mean, the variation approaches one, and the texture
    /// fades to doing nothing on its own. Bleeding between neighbouring atlas tiles is the
    /// only remaining reason to stop, and it is bounded now rather than catastrophic.
    /// </summary>
    public float BlockTextureMipLimit { get; set; } = 6f;

    /// <summary>
    /// How many mip levels sharper than the distance warrants to sample the atlas.
    ///
    /// Correct minification is exactly why a faithful texture is invisible at seam distance:
    /// the hardware averages the tile down to its own mean, and a texture applied as variation
    /// around the block's colour then multiplies by one. Zero is physically right and does
    /// nothing; this reaches back up the mip chain for detail the eye can resolve, at the cost
    /// of aliasing. Two is a starting guess and this is the knob most likely to decide whether
    /// real textures are worth having here at all.
    /// </summary>
    public float BlockTextureDetail { get; set; } = 2f;

    public float BandDarken { get; set; } = 1f;

    /// <summary>
    /// Saturation multiply at full weight. Normalising both sides of the measured seam by
    /// their own luminance leaves vanilla about 25% further from grey than cached ground -
    /// which is what losing texture detail, crevice occlusion and drop shadows does to a
    /// surface. Pushing away from grey needs no opinion about which hue is there.
    /// </summary>
    public float BandSaturate { get; set; } = 1f;

    /// <summary>
    /// Extra overlap between cached and live vanilla terrain, in blocks, beyond the
    /// geometric slack the ownership denial already applies. Handing columns back early
    /// costs a wider band where both draw, which is what the sink is for.
    ///
    /// ZERO since 0.3.163, and the history is the point. This was 16 blocks because 16
    /// blocks closed a hole in play, and the reasoning attached to it - that vanilla unloads
    /// chunks on its own schedule and ownership is withdrawn only once loss is detected - was
    /// never established; `dev/TODO.md` carried it as "a mitigation" for six releases. It was
    /// a mitigation for two defects that had nothing to do with unload timing, and half a
    /// chunk of overlap happened to be wide enough to bury both:
    ///
    /// - the fragment shader charged a wall standing IN a chunk plane to the cell on the far
    ///   side of that plane, so a wall on the cache side of the seam was hidden as vanilla's
    ///   work while vanilla drew nothing there (0.3.162, G136);
    /// - the mesher let a crop cube hide the soil's top face and its neighbour's wall, an
    ///   occlusion the game does not provide for a block it draws see-through (0.3.163, G137).
    ///
    /// With both fixed the margin buys nothing and costs a permanently doubled band, so the
    /// default is what the owner had already been running by hand. It stays a knob because
    /// the distance at which vanilla lets a chunk go is the engine's business, not ours: if a
    /// hole appears at the seam, raising this is the first thing to try and the fact that it
    /// helps is evidence about where to look, not a fix.
    /// </summary>
    public float OverlapMarginBlocks { get; set; }

    /// <summary>
    /// Paint the band's weight instead of correcting with it, so its shape can be judged from
    /// above before any colour decision is made.
    /// </summary>
    public bool BandDebugPaint { get; set; }

    /// <summary>
    /// How far solid cached terrain is lowered, in sixteenths of a block. A constant global
    /// offset with no ramp and no edge: every cached surface moves together, so the only
    /// place it can be seen is where cached terrain overlaps live vanilla terrain. It exists
    /// for depth fighting, and it is worth most underwater, where a lake floor drawn twice
    /// fights between a pale bed and a dark water-tinted one and every flipped pixel is a
    /// maximum-contrast flip. Whether one voxel is enough separation at the distance the
    /// overlap sits is a depth-precision question only a run can answer.
    /// </summary>
    public int BandSinkVoxels { get; set; } = 1;

    /// <summary>
    /// Apply vanilla's own rule that an up-facing surface never darkens as the sun drops:
    /// `getBrightnessFromNormal` floors its shade at `normal.y * 0.95`, and the liquid shader
    /// does not shade by normal at all. Without it cached ground fell to 0.55 at dawn and
    /// dusk beside vanilla ground still at 0.95, which is why the colour matched at midday
    /// and not at other times once the albedo itself was exact. On by default; `.dltoplight
    /// off` restores the old shading for comparison.
    /// </summary>
    public bool FlatTopLight { get; set; } = true;

    /// <summary>
    /// Shade by the engine's own light vector rather than by the sun. `LightPosition3D` is
    /// the sun position lerped toward the moon as moonlight overtakes sunlight, and vanilla
    /// terrain shades by it; without this, cached slopes are lit all night by a sun that is
    /// below the horizon while the vanilla slopes beside them are lit by the moon. Identical
    /// to the old behaviour while the sun is up. `.dllight moondir off` to compare.
    /// </summary>
    public bool LightMoonDirection { get; set; } = true;

    /// <summary>
    /// Use vanilla's shade ramp, `max(0.45, 0.5 + 0.5 * dot)`, instead of
    /// `0.55 + 0.45 * max(0, dot)`. The old floor left every away-facing slope 22% brighter
    /// than the vanilla slope beside it, at every time of day. `.dllight ramp off` compares.
    /// </summary>
    public bool LightVanillaRamp { get; set; } = true;

    /// <summary>
    /// Light by `Ambient.BlendedAmbientColor`, which is exactly what vanilla's terrain
    /// shader multiplies by, instead of `SunColor * DayLightStrength`. Vanilla derives its
    /// ambient from `ReflectColor`, floored at a blue night colour once the sun is down;
    /// `SunColor` has no such floor and stays orange. Decoding the engine's own sunlight
    /// ramp offline showed the two hues invert after sundown - vanilla turns blue-grey while
    /// cached terrain turns orange - which is the reported "sunset and night look
    /// substantially different". `.dllight ambient off` to compare.
    /// </summary>
    public bool LightAmbientColor { get; set; } = true;

    /// <summary>
    /// Apply vanilla's daylight brightening, `1 + max(0, shadowIntensity * 2 - 1.66) / 1.5`.
    /// The engine uploads `shadowIntensity` from `DropShadowIntensity`, which holds at 1
    /// while the sun is high and falls to 0 as it sets, so vanilla terrain is 22.7% brighter
    /// than cached terrain at midday and equal to it at dusk. This is the only one of the
    /// four that changes broad daylight, which is the case the owner reports as already
    /// looking right, so it has its own switch. `.dllight boost off` to compare.
    /// </summary>
    public bool LightDayBoost { get; set; } = true;

    /// <summary>
    /// Multiply by the sun light level vanilla's terrain carries, instead of treating sky
    /// light as 1. This is the largest of the lighting corrections and the only one that is
    /// a flat factor at every hour.
    ///
    /// `applyLight` multiplies by `lightColor.a`, the sun level baked into the terrain
    /// vertex. The 0.3.51 derivation read the LOD case as "full sky light" and cancelled
    /// that factor: `bMax` renormalisation cancels the ambient colour's own mean, not the
    /// sun level, so the level survives into the pixel. Sky-exposed ground is not level 31
    /// either - it is `IWorldAccessor.SunBrightness`, 22 in the base game, which the
    /// engine's own ramp puts at 0.806.
    ///
    /// Every cached surface was therefore 1.24x too bright. Measured against a flat
    /// two-block test patch at the seam on 2026-08-31: cached granite gravel read 1.39x
    /// vanilla's, 1.07x of which was the stored colour, leaving 1.29x for this.
    /// `.dllight sun off` restores the old brightness for comparison.
    /// </summary>
    public bool LightSunLevel { get; set; } = true;

    /// <summary>
    /// The sun light level a sky-exposed vanilla terrain vertex carries, as the shader sees
    /// it. Read from the world rather than assumed: a server may replace either the ramp or
    /// the brightness, and both are public.
    ///
    /// Quantised the way the engine quantises it. The float in `SunLightLevels` is not what
    /// reaches the shader - `ColorUtil.LightUtil` bakes it into the vertex's alpha byte
    /// first, so level 22 arrives as 206/255 rather than 0.806. Matching the rounding costs
    /// nothing and keeps this exact instead of nearly right.
    /// </summary>
    float SunLightLevel()
    {
        float[]? levels = capi.World.SunLightLevels;
        if (levels == null || levels.Length == 0) return 1f;
        float level = levels[GameMath.Clamp(capi.World.SunBrightness, 0, levels.Length - 1)];
        return GameMath.Clamp(MathF.Round(level * 255f) / 255f, 0f, 1f);
    }

    /// <summary>
    /// Fade the far edge of the cache toward the sky the engine actually draws. Every engine
    /// shader that calls `getSkyColorAt` passes `SkyDaylight`, which is
    /// `1.25 * max(DayLightStrength - MoonLightStrength / 2, 0.05)` attenuated above 1,000
    /// blocks over sea level - not `DayLightStrength`, which is what this shader passed. The
    /// band was therefore dissolving terrain into a sky a quarter too dim by day and too dark
    /// at night, with the error moving through the day like the terrain terms above.
    /// `.dllight sky off` to compare.
    /// </summary>
    public bool LightSkyDayLight { get; set; } = true;

    /// <summary>
    /// The engine's `SkyDaylight`, reconstructed. `DefaultShaderUniforms.SkyDaylight` is
    /// `internal`, so a mod cannot read the value the engine computed; every input to it is
    /// public, and the formula is transcribed from `SystemRenderSky.OnRenderFrame3D`.
    /// </summary>
    float SkyDayLight()
    {
        IClientGameCalendar calendar = capi.World.Calendar;
        float light = 1.25f * Math.Max(calendar.DayLightStrength - calendar.MoonLightStrength / 2f, 0.05f);
        // Thin air high above the world dims the sky rather than brightening it. Sea level
        // comes from the world, not from a constant: it moves with world configuration.
        double above = (capi.World.Player.Entity.Pos.Y - capi.World.SeaLevel - 1000.0) / 30000.0;
        float attenuation = 1f - (float)Math.Clamp(above, 0.0, 1.0);
        return Math.Max(0f, light * attenuation);
    }

    /// <summary>
    /// Opaque terrain has outward counter-clockwise winding and uses hardware back-face
    /// rejection; blended water and thin cover stay two-sided. Human testing across cliffs,
    /// caves, overhangs, and high/low views found no visual difference, while a same-view
    /// A/B improved 218 to 260 FPS. `.dlbackface off` remains the immediate comparison and
    /// compatibility fallback.
    /// </summary>
    public bool OpaqueBackfaceCulling { get; set; } = true;

    /// <summary>
    /// Cull each section with the vertical extent of its own mesh rather than a
    /// bedrock-to-sky box. Sections never recorded how tall the terrain inside them was,
    /// so looking up or down kept every section the side planes did not reject. The gate
    /// is one-sided - a bound that is too tight deletes terrain a player can see - so
    /// `.vhheight off` restores the full-height box for an immediate same-view comparison.
    /// </summary>
    public bool SectionHeightCulling { get; set; } = true;

    /// <summary>
    /// Cull whole quadtree SUBTREES with the aggregate extent of the meshes resident under
    /// them rather than a bedrock-to-sky box. The per-section bound above only tightens the
    /// leaf a section draws with; a coarse node still had to be traversed with a box
    /// spanning the world, and rejecting a coarse node rejects everything beneath it.
    /// Same one-sided gate, same fallback rule: a subtree that cannot supply an aggregate
    /// keeps the full-height box. `VINTAGEHORIZONS_SUBTREE_HEIGHT_CULLING=0` restores it
    /// everywhere for a controlled comparison.
    /// </summary>
    public bool SubtreeHeightCulling { get; set; } = true;

    /// <summary>
    /// Do not build underground geometry the surface-light envelope cannot reach. Measured
    /// offline at about 61% of estimated subterranean geometry, and unlike the culling
    /// switches above it removes the
    /// geometry rather than deciding whether to draw it - so it also takes memory, mesh
    /// time and upload bandwidth with it.
    ///
    /// Default ON after the surface-only rule and its tunnel guards were accepted in game.
    /// `.dlcavecull off` remains the immediate fallback; changing it re-meshes the world so the
    /// switch can be judged in one place without relogging. The environment can pin it off.
    /// </summary>
    public bool CaveCulling { get; set; } =
        Environment.GetEnvironmentVariable("VINTAGEHORIZONS_CAVE_CULLING") != "0";

    int halfBlockLevels = NearHalfBlockLevels;

    /// <summary>
    /// What `.dlslabs surface` and `.dlslabs near` select: all of level 0, out to the
    /// configured L1 transition (512 blocks by default), independently of the blend band.
    /// </summary>
    internal const int NearHalfBlockLevels = 1;

    /// <summary>
    /// How many detail levels draw terrain slabs at half height, counting from the finest.
    /// Mods that smooth the landscape with half blocks are otherwise rebuilt as full cubes,
    /// which puts back exactly the stair-steps they removed and stands the ground half a
    /// block high.
    ///
    /// Surface-only level 0 is the default after the owner's same-scene test found very
    /// little performance cost. It applies throughout level 0, not merely in the blend
    /// band, and retains the landscape result while the exact all-run version remains
    /// available through `.dlslabs near` and `.dlslabs all`.
    /// </summary>
    public int HalfBlockLevels
    {
        get => halfBlockLevels;
        set => halfBlockLevels = Math.Clamp(value, 0, 8);
    }

    /// <summary>
    /// Keep half-block detail to the highest solid run in each column. This is the
    /// performance-oriented terrain mode: outdoor slabs stay exact, while lower distant
    /// cave and overhang slabs use the established full-block LOD approximation.
    /// </summary>
    public bool SurfaceHalfBlocksOnly { get; set; } = true;

    /// <summary>
    /// Whether a block the GAME draws see-through - a crop, tall grass, a berry bush, a vine -
    /// may hide the faces around it in the cached mesh.
    ///
    /// Off, which is correct. The cached mesh draws these as solid cubes so a meadow still
    /// reads as a meadow, but the game draws them as transparent tufts, so letting one act as
    /// cover claims an occlusion the game does not provide. At the vanilla seam the game draws
    /// where we do not, and the two faces we dropped - the soil's top under a crop and the wall
    /// between two of them - become a window through the hollow cached terrain to the sky.
    /// Reported in play at a seam through a flax field.
    ///
    /// On restores the behaviour before 0.3.163. The cost of the fix is geometry that is
    /// invisible everywhere except that seam, so this exists to A/B the frame rate rather than
    /// the picture. Changing it re-meshes every resident section. See `.dlcutout`.
    /// </summary>
    public bool CutoutBlocksCover { get; set; }

    /// <summary>
    /// Darken submerged ground and deep water by the water standing above it. Changing it
    /// re-meshes every resident section. See `.dlwater depth`.
    ///
    /// Water absorbs one light level per block, so vanilla draws a lakebed under 8 blocks at
    /// 70% of the shore beside it and one under 20 at 26%. Cached terrain carries no
    /// per-position light at all - one sky-exposed level multiplies every surface in the world
    /// - so without this a lakebed is as bright as an open field and distant water reads as
    /// flat paint. Measured at +0.24% vertices over 481,287 columns of a real cache.
    /// </summary>
    public bool WaterDepthShading
    {
        get => waterDepthShading;
        set
        {
            if (waterDepthShading == value) return;
            waterDepthShading = value;
            RemeshAll();
        }
    }

    bool waterDepthShading = true;
    float[]? waterAbsorption;
    bool waterAbsorptionDeferred;

    /// <summary>
    /// Apply the engine's screen-space murk and fog removal to cached ground beneath
    /// water. A session-only comparison switch, independent of baked depth shading.
    /// </summary>
    public bool WaterGroundMurk { get; set; } = true;

    public bool WaterDepthCoverage { get; set; } = true;

    /// <summary>Session-only isolation of cached water, preserving ground and thin cover.</summary>
    public bool WaterSurfaceVisible { get; set; } = true;

    /// <summary>
    /// Sample the standard water tile from the game's live terrain atlas in the dedicated
    /// surface shader. This is session-only while the boundary comparison is open.
    /// </summary>
    public bool WaterSurfaceTexture { get; set; } = true;

    /// <summary>The live block-table slot for game:water-still-7, resolved per world.</summary>
    public int WaterSurfaceTextureSlot { get; set; }

    /// <summary>
    /// The tint slot carried by standard water. The water buffer also contains lava and ice;
    /// comparing this value keeps the standard water texture off those materials without
    /// adding another vertex attribute.
    /// </summary>
    public int WaterSurfaceTintSlot { get; set; } = -1;

    bool WaterSurfaceTextureReady => WaterSurfaceTexture
        && WaterSurfaceTextureSlot > 0 && WaterSurfaceTintSlot > 0
        && BlockTextureTableCurrent && blockTextures != null
        && blockTextures.AtlasTextureId > 0;

    public string WaterSurfaceTextureStatus => !WaterSurfaceTexture ? "off"
        : WaterSurfaceTextureSlot <= 0 || WaterSurfaceTintSlot <= 0
            ? "on (standard water tile unavailable)"
            : WaterSurfaceTextureReady
                ? $"on (live atlas slot {WaterSurfaceTextureSlot})"
                : $"on (atlas slot {WaterSurfaceTextureSlot} pending upload)";

    /// <summary>
    /// How shallow the water shader assumes cached water is, 0 to 1. Zero is vanilla's answer
    /// for open water and the default; 1 is the "infinitely shallow" substitution every build
    /// before this one made, which pinned vanilla's shoreline foam on over every ocean.
    /// See `waterShallowness` in lodterrainbody.fsh.
    /// </summary>
    public float WaterShallowness { get; set; }

    /// <summary>
    /// The highest pre-texture Fresnel alpha cached water may reach, 0.05 to 2. Vanilla clamps
    /// this exact intermediate value to 2 before climate-mapped texture alpha and foam reduce or
    /// raise it into the final 0..1 material alpha. Lower values deliberately expose more of the
    /// depth-shaded bed beneath distant water.
    ///
    /// Version 0.4.8 briefly used 1.0 before the dedicated OIT surface and climate-map alpha
    /// existed, and looked worse. Under the corrected material path the owner compared the same
    /// scene across viewing angles and selected 1.3: 1.0 remained too transparent at a shallow
    /// angle, while higher values overshot before reaching vanilla's theoretical 2.0 ceiling.
    ///
    /// This is deliberately a material control rather than a final-alpha clamp: applying it after
    /// texture mapping would no longer be the same quantity the vanilla vertex shader limits.
    /// </summary>
    public float WaterMaxAlpha { get; set; } = 1.3f;

    /// <summary>
    /// How many blocks of water the view ray travels before it has lost about two thirds of
    /// what it carried, in blocks. Smaller is murkier and darkens with depth faster.
    ///
    /// Changing it rebuilds every resident mesh, because the factor is baked in when the mesh
    /// is built. See `.dlwater clarity`.
    /// </summary>
    public float WaterClarity
    {
        get => waterClarity;
        set
        {
            float wanted = value <= 0f ? 0f : Math.Clamp(value, 1f, 64f);
            if (Math.Abs(waterClarity - wanted) < 0.01f) return;
            waterClarity = wanted;
            waterAbsorption = null;
            RemeshAll();
        }
    }

    // The owner selected 25 after comparing the rebuilt depth-shaded seabed through the
    // required dedicated OIT surface. Earlier values were judged while surface alpha,
    // atmospheric opacity and, in one build, the active shader path were different.
    float waterClarity = 40f;

    /// <summary>
    /// How many blocks either side of a 32-block vanilla chunk plane keep one water quad per
    /// block, so the surface can hold vanilla's wave where a seam can appear. 0 turns the wave
    /// off entirely. Changing it rebuilds every resident mesh. See `.dlwater wave`.
    ///
    /// One block is the measured sweet spot: it is full per-block fidelity in the only place a
    /// seam can land, for +1.02% of the mesh, against +16.32% for doing it everywhere and
    /// +4.24% for 2x2 patches that would be worse at the seam than this is.
    /// </summary>
    public int WaterSeamStripBlocks
    {
        get => waterSeamStripBlocks;
        set
        {
            int wanted = Math.Clamp(value, 0, 4);
            if (waterSeamStripBlocks == wanted) return;
            waterSeamStripBlocks = wanted;
            RemeshAll();
        }
    }

    int waterSeamStripBlocks = 1;

    /// <summary>
    /// Drop water walls against any neighbour we have no capture for, not just an absent
    /// section. A DIAGNOSTIC first: it exists so the seam band can be attributed by flipping
    /// it. Default false, which is the established behaviour. See `.dlwater walls`.
    /// </summary>
    public bool SuppressUnknownWaterWalls
    {
        get => suppressUnknownWaterWalls;
        set
        {
            if (suppressUnknownWaterWalls == value) return;
            suppressUnknownWaterWalls = value;
            RemeshAll();
        }
    }

    bool suppressUnknownWaterWalls;

    /// <summary>Blocks past the handoff over which the wave fades to nothing.</summary>
    public float WaterWaveReachBlocks { get; set; } = 64f;


    /// <summary>
    /// Brightness under n blocks of water as a share of open air, read from the world's own
    /// ramp. Rebuilt only when the ramp or the maximum sun level changes, which is a world
    /// load rather than a frame.
    /// </summary>
    float[]? WaterAbsorptionTable()
    {
        if (!waterDepthShading) return null;
        if (waterAbsorption != null) return waterAbsorption;

        float[]? levels = capi.World.SunLightLevels;
        if (levels == null || levels.Length == 0) return null;
        int sky = GameMath.Clamp(capi.World.SunBrightness, 0, levels.Length - 1);
        float open = levels[sky];
        if (open <= 0f) return null;

        // TWO paths through the water, not one. The engine's light absorption is the sunlight
        // going DOWN - one level per block - and it is the whole of what the illuminator knows.
        // It is not the whole of what the eye sees: the light then travels back UP through the
        // same water to reach the camera, and that second pass is what vanilla's semi-
        // transparent surface does per pixel and a shaded bed does not.
        //
        // Modelling only the downward half made the ramp far too gradual, which is what the
        // owner saw: at the same viewpoint, vanilla's mid-depth water measured luma 40 against
        // our 63, while the shallows already matched at 76. Beer-Lambert on the return path
        // costs nothing here - same merge key, same baked colour - and is the physically
        // sensible shape rather than a curve fitted to a screenshot.
        //
        // The clarity is a LOOK setting and deliberately a dial: real water's extinction depends
        // on what is dissolved in it, the engine models none of that, and the only judge of
        // whether a lake reads right is a person looking at one.
        // Zero clarity means the engine's own absorption ALONE, which is the honest default:
        // vanilla has no return-path term of any kind. Its bed carries the sunlight that
        // reached it and nothing else, and everything that looks like extinction comes from a
        // transparent surface blended over that bed, per pixel. That is why this term could
        // never be tuned to hold: it is a function of depth, and the real thing is a function
        // of depth AND the angle you look through the water at, which is why a clarity that
        // matched at one render distance stopped matching at another.
        float clarity = Math.Max(0f, WaterClarity);
        var table = new float[LodMesher.MaxSubmergedDepth + 1];
        for (int depth = 0; depth < table.Length; depth++)
        {
            int level = GameMath.Clamp(sky - depth, 0, levels.Length - 1);
            float sunlight = Math.Clamp(levels[level] / open, 0f, 1f);
            float viewPath = clarity > 0f ? MathF.Exp(-depth / clarity) : 1f;
            table[depth] = Math.Clamp(sunlight * viewPath, 0f, 1f);
        }

        // A table of ones darkens nothing while still splitting the merge, so the feature
        // would cost its whole geometry bill and draw exactly what it drew before - which is
        // indistinguishable in play from the switch not working at all. It happens when the
        // world is not ready to answer yet: a SunBrightness of 0 makes every lookup clamp to
        // the same floor and every ratio come out 1. Not cached, so the next mesh asks again
        // rather than freezing the world's start-up state for the session.
        if (table[table.Length - 1] >= 0.999f)
        {
            if (!waterAbsorptionDeferred)
            {
                waterAbsorptionDeferred = true;
                capi.Logger.Notification(
                    "[Landscapes] water depth shading has nothing to apply yet: "
                    + $"sun brightness {capi.World.SunBrightness}, {levels.Length} levels, "
                    + $"open water {open:0.000}. Retrying on the next mesh.");
            }
            return null;
        }

        waterAbsorption = table;
        capi.Logger.Notification(
            "[Landscapes] water depth shading resolved: " + DescribeWaterAbsorption());

        // Anything meshed while the answer was unavailable is built unshaded and would stay
        // that way for the session - the join is exactly when the world is least ready to
        // answer, and it is also when most of the view is built. Rebuild those once.
        if (waterAbsorptionDeferred)
        {
            waterAbsorptionDeferred = false;
            int queued = RemeshAll();
            if (queued > 0)
            {
                capi.Logger.Notification(
                    $"[Landscapes] rebuilding {queued} sections meshed before it resolved.");
            }
        }
        return table;
    }

    /// <summary>
    /// What the depth table actually resolved to, for the log and for `.dlwater`. A person
    /// cannot copy chat out of the game, and "it had no effect" and "it is applying 1.00 at
    /// every depth" look identical on screen; this is the line that tells them apart.
    /// </summary>
    public string DescribeWaterAbsorption()
    {
        if (!waterDepthShading) return "off";

        // RESOLVE it, do not report the cache. Every setter that changes the shading clears
        // waterAbsorption and queues a rebuild, and the command then asked for a description
        // in the same breath - so the answer was always "no table resolved yet", on every
        // single call, which is exactly the line that was supposed to tell a real failure
        // apart from a working one. It said nothing at all for a whole testing session.
        float[]? table = WaterAbsorptionTable();
        if (table == null) return "on, but the world has not answered with a light ramp yet";
        return $"clarity {waterClarity:0} blocks, sky level {capi.World.SunBrightness}, "
            + $"1 block {table[Math.Min(1, table.Length - 1)]:0.00}, "
            + $"4 {table[Math.Min(4, table.Length - 1)]:0.00}, "
            + $"8 {table[Math.Min(8, table.Length - 1)]:0.00}, "
            + $"16 {table[Math.Min(16, table.Length - 1)]:0.00}, "
            + $"{table.Length - 1}+ {table[table.Length - 1]:0.00} of shore brightness";
    }

    int caveCullReach = LodCaveCull.DefaultReach;

    /// <summary>How far daylight spreads before a cavity counts as unreachable, in blocks.</summary>
    public int CaveCullReach
    {
        get => caveCullReach;
        set => caveCullReach = Math.Clamp(value, 1, 200);
    }

    /// <summary>
    /// Rebuild every resident mesh. Cave culling decides what geometry EXISTS rather than
    /// what is drawn, so flipping it changes nothing until the meshes are made again - and
    /// a switch whose effect only arrives on the next relog cannot be A/B'd by a person.
    /// </summary>
    public int RemeshAll()
    {
        int queued = 0;
        foreach (long key in sectionMeshes.Keys.Concat(waterMeshes.Keys).Distinct().ToArray())
        {
            world.RenderDirty.Add(key);
            queued++;
        }
        return queued;
    }

    /// <summary>
    /// Submit opaque cached sections nearest-first so mountain depth can reject farther
    /// hidden fragments before their shader runs. Water keeps the traversal order because
    /// alpha blending has different ordering rules. A same-view human A/B improved 149 to
    /// 173 FPS with no visible change; `.dlfront off` remains the immediate fallback.
    /// </summary>
    public bool OpaqueFrontToBack { get; set; } = true;

    /// <summary>
    /// Delayed exact-geometry occlusion. Ordinary opaque draws are sampled
    /// asynchronously after vanilla has populated depth. A zero-sample result skips later
    /// submissions. Profile/projection and selected camera-threshold changes invalidate
    /// globally; streamed scene changes invalidate only affected pieces and rely on the
    /// periodic exact probe for convergence. Camera turns shorten that probe cadence.
    /// Unlike the rejected prototype this has no proxy boxes, conditional rendering, or
    /// same-frame dependency. The aggressive profile is default-on after in-game testing
    /// found a substantial FPS gain; mixed ownership and a narrow turning-edge band are
    /// deliberately protected from delayed hiding.
    /// </summary>
    public bool TemporalOcclusionEnabled { get; set; } =
        Environment.GetEnvironmentVariable("VINTAGEHORIZONS_TEMPORAL_OCCLUSION") != "0";

    double temporalOcclusionTranslationLimitBlocks = 2.0;
    float temporalOcclusionRotationMatrixLimit = float.PositiveInfinity;
    int temporalOcclusionVisibleQueryIntervalFrames = 8;
    int temporalOcclusionHiddenProbeIntervalFrames = 16;
    int temporalOcclusionTurningProbeIntervalFrames = 4;
    double temporalOcclusionEdgeGuard = 0.06;
    public string TemporalOcclusionProfileName { get; private set; } = "aggressive";

    public bool SetTemporalOcclusionProfile(string profile)
    {
        switch (profile.ToLowerInvariant())
        {
            case "safe":
                temporalOcclusionTranslationLimitBlocks = SafeTemporalTranslationBlocks;
                temporalOcclusionRotationMatrixLimit = SafeTemporalRotationMatrix;
                temporalOcclusionVisibleQueryIntervalFrames = 4;
                temporalOcclusionHiddenProbeIntervalFrames = 8;
                temporalOcclusionTurningProbeIntervalFrames = 8;
                temporalOcclusionEdgeGuard = 0;
                break;
            case "aggressive":
                temporalOcclusionTranslationLimitBlocks = 2.0;
                temporalOcclusionRotationMatrixLimit = float.PositiveInfinity;
                temporalOcclusionVisibleQueryIntervalFrames = 8;
                temporalOcclusionHiddenProbeIntervalFrames = 16;
                temporalOcclusionTurningProbeIntervalFrames = 4;
                temporalOcclusionEdgeGuard = 0.06;
                break;
            case "extreme":
                temporalOcclusionTranslationLimitBlocks = double.PositiveInfinity;
                temporalOcclusionRotationMatrixLimit = float.PositiveInfinity;
                temporalOcclusionVisibleQueryIntervalFrames = 16;
                temporalOcclusionHiddenProbeIntervalFrames = 32;
                temporalOcclusionTurningProbeIntervalFrames = 8;
                temporalOcclusionEdgeGuard = 0;
                break;
            default:
                return false;
        }

        TemporalOcclusionProfileName = profile.ToLowerInvariant();
        InvalidateTemporalOcclusionScene();
        return true;
    }

    /// <summary>
    /// Render cached terrain just after vanilla terrain so the ordinary depth test can
    /// reject fragments hidden behind current chunks. The established order stays the
    /// default after an owner A/B measured a substantial gain and found only minute,
    /// acceptable distant changes. Changing this property re-registers the renderer because
    /// the engine sorts render order only during registration. The pre-vanilla order remains
    /// the immediate compatibility fallback.
    /// </summary>
    public bool OcclusionCullingEnabled
    {
        get => postVanillaDepthCulling;
        set
        {
            if (postVanillaDepthCulling == value) return;
            if (!rendererRegistered)
            {
                postVanillaDepthCulling = value;
                return;
            }

            bool previous = postVanillaDepthCulling;
            capi.Event.UnregisterRenderer(this, EnumRenderStage.Opaque);
            rendererRegistered = false;
            postVanillaDepthCulling = value;
            InvalidateTemporalOcclusionScene();
            try
            {
                RegisterRenderer();
            }
            catch (Exception orderError)
            {
                postVanillaDepthCulling = previous;
                try
                {
                    RegisterRenderer();
                }
                catch (Exception restoreError)
                {
                    throw new AggregateException(
                        "Changing cached-terrain render order failed, and restoring its previous order also failed.",
                        orderError, restoreError);
                }
                capi.Logger.Warning(
                    "[Landscapes] Post-vanilla depth culling unavailable; restored ordinary render order: {0}",
                    orderError.Message);
            }
        }
    }
    bool postVanillaDepthCulling = true;
    bool rendererRegistered;
    public double ReadinessOwnedWithoutGeometryNearest { get; private set; }
    public double ReadinessOwnedWithoutGeometryFarthest { get; private set; }
    public long VanillaOwnedDrawsSkipped { get; private set; }
    public long CoarseWaitingLoad { get; private set; }
    public long CoarseWaitingMesh { get; private set; }
    public long CoarseWaitingSchedule { get; private set; }
    public long CoarseWaitingOther { get; private set; }

    public void ResetPhaseCosts()
    {
        PruneCost.Reset();
        ScheduleCost.Reset();
        UploadCost.Reset();
        GlUploadCost.Reset();
        MeshDisposeCost.Reset();
        EvictCost.Reset();
        SeasonalCost.Reset();
        FarDistanceCost.Reset();
        ReadinessCost.Reset();
        WalkCost.Reset();
        frameTimeline.Reset();
        DrawCost.Reset();
        ProjectionResetCount = 0;
        MeshUploadBytes = 0;
        MeshSnapshotBytes = 0;
        MeshSnapshotItems = 0;
        MeshUploadItems = 0;
        StaleMeshResultsRejected = 0;
        OpaqueDrawCalls = 0;
        WaterDrawCalls = 0;
        OpaqueDrawVertices = 0;
        OpaqueDrawIndices = 0;
        WaterDrawVertices = 0;
        WaterDrawIndices = 0;
        gpuTelemetry.ResetInterval();
        TemporalOcclusionQueries = 0;
        TemporalOcclusionResults = 0;
        TemporalOcclusionHiddenResults = 0;
        TemporalOcclusionStaleResults = 0;
        TemporalOcclusionGlobalInvalidations = 0;
        TemporalOcclusionDrawsSkipped = 0;
        ReadinessProbes = 0;
        ReadinessTrueResults = 0;
        ReadinessFalseResults = 0;
        ReadinessReadyTransitions = 0;
        ReadinessLostTransitions = 0;
        ReadinessProbeErrors = 0;
        ReadinessWindowChanges = 0;
        ReadinessResizes = 0;
        ReadinessFullRevalidations = 0;
        ReadinessAggregateRepairs = 0;
        ReadinessStaleCommittedFound = 0;
        ReadinessEmptyChunksSeen = 0;
        ReadinessDrawnWithoutGeometry = 0;
        ReadinessOwnedWithoutGeometry = 0;
        ReadinessOwnershipDeniedNoGeometry = 0;
        ReadinessOwnershipDeniedBeyondViewDistance = 0;
        // Per interval like everything else here, so the exact/bounded split describes what
        // the renderer is doing NOW. A lifetime total would stay dominated by whatever the
        // first seconds after a join did, which is exactly when the answer is least settled.
        ReadinessExactRangeDecisions = 0;
        ReadinessBoundedRangeDecisions = 0;
        ReadinessRangeSweeps = 0;
        ReadinessRangeSweepLosses = 0;
        ReadinessRangeSweepPeakMs = 0;
        ReadinessMaskResyncs = 0;
        ReadinessOwnedWithoutGeometryNearest = -1;
        ReadinessOwnedWithoutGeometryFarthest = -1;
        Array.Clear(readinessNoGeometryByY);
        VanillaOwnedDrawsSkipped = 0;
        CoarseWaitingLoad = 0;
        CoarseWaitingMesh = 0;
        CoarseWaitingSchedule = 0;
        CoarseWaitingOther = 0;
        SeamRepairsQueued = 0;
        VerticalCulledSections = 0;
        VerticalCulledMax = 0;
        SubtreeCulledNodes = 0;
        SubtreeCulledMeshes = 0;
        SubtreeCulledMax = 0;
        // Through the shared reset rather than by repeating it. The depth report prints CPU
        // time, GPU time, hidden share and the picture's own frame counts on adjacent lines,
        // and if the periodic reset cleared some of those and not others, the reader would be
        // comparing figures from different intervals with nothing on screen saying so.
        ResetDepthPyramidCounters();
        HzbHiddenBeyondQueries = 0;
        HzbAgreedHidden = 0;
        HzbVerdictsAcrossViewChange = 0;
        sectionHeightStats.Reset();
        readiness?.ResetTelemetry();
        readinessMask?.ResetTelemetry();
    }

    readonly Dictionary<long, MeshRef> sectionMeshes = new();
    readonly Dictionary<long, MeshRef> waterMeshes = new();
    readonly Dictionary<long, LodLiveMeshStats> liveMeshStats = new();

    readonly record struct LodLiveMeshStats(
        int OpaqueVertices, int OpaqueIndices, int WaterVertices, int WaterIndices,
        LodSectionHeights Heights, bool HasTextureSlots,
        LodMeshGroundCoverage OpaqueGroundCoverage)
    {
        public long Bytes => OpaqueVertices * 16L + OpaqueIndices * sizeof(int)
            + WaterVertices * 16L + WaterIndices * sizeof(int);
    }
    readonly LodSubtreeHeights subtreeHeights = new();
    readonly LodMeshBounds meshBounds = new();
    readonly LodFarPlaneState farPlaneState = new();
    readonly HashSet<long> meshJobInFlight = new();
    readonly LodRenderDirtyScheduler dirtyScheduler = new();
    // The first LOD band is where the eye judges sharpness first. Give it three quarters
    // of the unchanged global admission/request allowance while reserving enough outward
    // capacity to keep all eight lanes moving. This changes priority, not total pressure.
    internal const int FoundationDemandOutstanding = 24;
    internal const int FoundationDemandRequestsPerFrame = 6;
    internal const int OutwardDemandOutstanding = 8;
    internal const int OutwardDemandRequestsPerFrame = 2;
    readonly LodRadialDemandPlanner foundationDemand = new(
        outstandingLimit: FoundationDemandOutstanding,
        requestsPerFrame: FoundationDemandRequestsPerFrame);
    readonly LodRadialDemandPlanner radialDemand = new(
        outstandingLimit: OutwardDemandOutstanding,
        requestsPerFrame: OutwardDemandRequestsPerFrame);
    readonly Predicate<long> keepRenderDirty;
    readonly Predicate<long> renderDirtyBlocked;
    readonly Predicate<long> radialDemandReady;
    readonly Predicate<long> radialDemandPending;
    readonly Predicate<long> radialDemandTerminal;
    readonly System.Func<long, bool> startRadialDemand;
    // Visibility is intentionally absent from this state. The camera may stop walking
    // an off-screen subtree, but distance-based residency still keeps appropriate meshes
    // warm for a turn-around (G8).
    readonly Dictionary<long, long> lastResidencyMs = new();
    readonly Queue<long> meshEvictionOrder = new();
    readonly HashSet<long> meshEvictionQueued = new();
    long frameCounter;

    sealed class TemporalOcclusionQuery
    {
        public int QueryId;
        public readonly LodTemporalOcclusionState State = new();
    }

    readonly Dictionary<long, TemporalOcclusionQuery> temporalOcclusionQueries = new();
    readonly List<long> temporalOcclusionPending = new();
    readonly float[] temporalOcclusionView = new float[16];
    readonly float[] temporalOcclusionProjection = new float[16];
    readonly float[] temporalOcclusionPreviousFrameView = new float[16];
    double temporalOcclusionCameraX;
    double temporalOcclusionCameraY;
    double temporalOcclusionCameraZ;
    long temporalOcclusionEpoch = 1;
    bool temporalOcclusionHasView;
    bool temporalOcclusionHasPreviousFrameView;
    bool temporalOcclusionTurningThisFrame;
    bool temporalOcclusionQueryTargetAvailable;
    bool temporalOcclusionFailed;
    bool temporalOcclusionFailureReported;
    int temporalOcclusionQueriesIssuedThisFrame;
    const int TemporalOcclusionQueryIssuesPerFrame = 8;
    const int TemporalOcclusionResultChecksPerFrame = 16;

    /// <summary>
    /// Sides (W, E, N, S as bits 0-3) whose live mesh was built against a neighbour that
    /// held stored data but was not in RAM. Only these owe a repair mesh when the
    /// neighbour lands - which is the whole point of recording them, because re-meshing
    /// all four neighbours of every arriving section would roughly double the mesh work
    /// of a warm join. Keys with nothing assumed are not stored.
    /// </summary>
    readonly Dictionary<long, byte> meshedWithoutNeighbor = new();

    /// <summary>Repair meshes queued because a guessed-at neighbour finally arrived.</summary>
    public int SeamRepairsQueued { get; private set; }

    // Phase 1 shadow tracker. It follows vanilla readiness and maintains aggregate
    // classifications, but no draw path reads those classifications yet. Pixel ownership
    // remains on the existing radial fallback until the GPU mask is implemented.
    VanillaRenderReadiness? readiness;
    readonly EntityPos readinessProbePos = new();
    int readinessSeedCursor;
    int readinessBoundaryCursor;
    int readinessInteriorCursor;
    bool readinessDisabled;
    bool readinessFailureReported;
    int[] readinessColumnReady = Array.Empty<int>();
    readonly StringBuilder readinessColumnText = new();
    readonly StringBuilder readinessNoGeometryText = new();
    readonly StringBuilder levelReport = new();
    readonly LodNearHandoffState nearHandoff = new();
    // Per-cell ownership, on by default since 0.3.17: the draw-range clause closed the band
    // that had kept it opt-in, and a player confirmed both the closure and the seam overlap
    // it accepts. The saved setting is applied over this at startup.
    //
    // The environment variable is now an override in both directions, because the benchmark
    // harness has to be able to measure the radial path after it stopped being the default:
    // `0` forces the mask off, `1` forces it on, anything else leaves the setting alone.
    bool chunkMaskRequested =
        Environment.GetEnvironmentVariable("VINTAGEHORIZONS_CHUNK_MASK") != "0";
    VanillaReadinessMask? readinessMask;
    LoadedTexture? readinessMaskTexture;
    bool readinessMaskFailed;
    bool readinessMaskActive;
    long waterFootprintRevision = -1;

    /// <summary>The growth the current water footprint was built with, so a change to it rebuilds.</summary>
    int waterFootprintDilation = -1;
    long readinessMaskUploadUs;
    VanillaSeamDistanceField? seamField;
    LoadedTexture? seamFieldTexture;
    bool seamFieldFailed;
    bool seamFieldActive;
    long seamFieldBuiltRevision = -1;
    long seamFieldLastBuildMs;
    long seamFieldBuildUs;
    long seamFieldUploadUs;
    int seamFieldWindowMinChunkX = int.MinValue;
    int seamFieldWindowMinChunkZ = int.MinValue;
    int seamFieldWindowWidth;
    int seamFieldWindowDepth;

    /// <summary>The margin the live field was built with; -1 until it has been built once.</summary>
    int seamFieldBuiltMarginCells = -1;
    bool seamFieldCapacityWarned;

    // ---- Real vanilla block textures -------------------------------------------------
    //
    // Cached ground is one averaged colour per block; vanilla ground never is. Rather than
    // approximate the difference a fourth time, near ground samples the game's OWN block
    // atlas, and the seam band decides how much of it shows. blockTable holds where each
    // block's texture sits inside that atlas plus the averaged colour to fall back to, so a
    // quad can carry a block slot instead of a colour and lose nothing.
    readonly LodBlockTextureRegistry? blockTextures;
    System.Func<int, (int X, int Z)>? wallColorsOf;
    int blockTableTexture;
    int blockTableVersion = -1;
    bool blockTableFailed;
    bool blockTexturesLive;
    bool blendPipelineEnabled = true;
    bool textureSlotSourceReady = true;
    bool textureSlotsReady;
    long textureModeRevision = 1;
    int liveTexturedMeshes;
    float readinessHandoffDistance;
    bool readinessOwnsHandoff;
    float readinessOwnedRadius;
    double readinessHandoffCameraX;
    double readinessHandoffCameraZ;
    int[] readinessNoGeometryByY = Array.Empty<int>();
    bool readinessHandoffStale = true;
    bool readinessSweepShellNow;
    long readinessLastFullRevalidateMs;
    bool readinessAggregateReported;

    /// <summary>
    /// This frame's copy of the engine's range-cull inputs, and the previous one.
    ///
    /// <para>
    /// Captured before the probe loop so every decision in a frame uses one consistent set,
    /// and kept so the sweep can tell whether anything it depends on actually moved. Both are
    /// value types, so this holds no reference into the client.
    /// </para>
    /// </summary>
    VanillaDrawRangeState drawRange;
    VanillaDrawRangeState lastSweptDrawRange;
    bool rangeSweepNow = true;
    readonly List<VanillaChunkCell> rangeLosses = new();

    /// <summary>Meshes outside the residency band for one minute get evicted.</summary>
    const long EvictAfterMs = 60_000;
    const int MeshEvictionChecksPerFrame = 4;

    public int EvictedTotal { get; private set; }
    readonly Matrixf modelMat = new();
    readonly List<long> drawList = new();
    readonly List<LodOpaqueDrawEntry> opaqueFrontToBack = new();
    IShaderProgram? prog;

    /// <summary>
    /// The same shader source compiled with VH_INDIRECT, so its per-section values come
    /// from the record buffer instead of uniforms. Not a second shader: one body file,
    /// two programs, which is what keeps "the fast path draws identical pixels" a claim
    /// the two paths cannot quietly fall out of.
    /// </summary>
    IShaderProgram? indirectProg;
    IShaderProgram? packedProg;
    bool shaderOk;
    bool indirectShaderOk;
    bool packedShaderOk;
    /// <summary>Whether the one-line "shaders ready" report has been made since the last failure.</summary>
    bool reportedShadersOk;
    bool reportedPackedShaderOk;
    IShaderProgram? hzbProg;
    bool hzbShaderOk;

    /// <summary>
    /// Build the private depth pyramid each frame. Phase 4 shadow work: it copies, reduces
    /// and times itself, and decides nothing. Off by default because it costs GPU time for
    /// no picture change, and because its cost against the drawing it could remove is
    /// exactly what has to be measured before anything is allowed to act on it.
    /// `.dlhzb on` turns it on for a session.
    /// </summary>
    public bool DepthPyramidEnabled
    {
        get => depthPyramidEnabled;
        set
        {
            // Switching it on restarts the counts. Otherwise the first reading a person
            // takes is diluted by however long the interval had already been running with
            // the feature off, and reads as a much smaller hidden share than it is.
            if (value && !depthPyramidEnabled) ResetDepthPyramidCounters();

            // A pyramid that is on is a pyramid being measured. Its cost is GPU time and
            // nothing else, and until now the timers armed only for a session started under
            // the benchmark harness - so an ordinary run reported 0.0us, which reads as
            // "free" rather than "nobody measured". Here rather than in the chat command,
            // because .vhcull on turns the pyramid on by assignment and would otherwise
            // leave the clock stopped (G72).
            if (value) gpuTelemetry.RequestTiming();

            depthPyramidEnabled = value;
        }
    }

    bool depthPyramidEnabled;

    /// <summary>
    /// Split cached opaque terrain at the measured 512-block boundary. The near bucket draws
    /// first, a fresh depth picture is taken, and the far bucket is culled and drawn against
    /// that picture in the SAME frame. Phase 6.
    ///
    /// The config and command retain their old "late" names so an existing saved opt-in now
    /// selects the safe replacement automatically. No verdict crosses a frame boundary.
    /// </summary>
    public bool LateDepthPyramid
    {
        get => lateDepthPyramid;
        set
        {
            if (lateDepthPyramid == value) return;
            lateDepthPyramid = value;

            ResetDepthPyramidCounters();
        }
    }

    bool lateDepthPyramid =
        Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_LATE") != "0";
    readonly float[] explainViewProjection = new float[16];
    public long SplitDepthFrames { get; private set; }
    public long SplitDepthNearCommands { get; private set; }
    public long SplitDepthFarCommands { get; private set; }
    public long SplitDepthMidBuilds { get; private set; }

    /// <summary>
    /// Starts a fresh measurement interval for everything the depth work is judged on.
    ///
    /// The GPU timers are in here, and that is the point. The phase's gate is GPU time and
    /// nothing else, and without this a figure read after flipping a switch is an average over
    /// BOTH arrangements - which looks entirely plausible and answers nothing. Resetting the
    /// ring also advances its epoch, so results still in flight from the old arrangement are
    /// discarded rather than landing in the new interval's total.
    ///
    /// Opaque and water GPU time reset too, deliberately: opaque time is the number that should
    /// FALL when culling works, so it belongs to the arrangement being measured just as much as
    /// the pyramid's own cost does.
    /// </summary>
    public void ResetDepthPyramidInterval() => ResetDepthPyramidCounters();

    void ResetDepthPyramidCounters()
    {
        HzbCost.Reset();
        gpuTelemetry.ResetInterval();
        depthPyramid?.ResetInterval();
        hzbClassifier?.ResetInterval();
        cullPass?.ResetInterval();
        SplitDepthFrames = 0;
        SplitDepthNearCommands = 0;
        SplitDepthFarCommands = 0;
        SplitDepthMidBuilds = 0;
        HzbHiddenButQuerySawIt = 0;
        HzbMissedWhatQueryHid = 0;
        HzbVerdictsChecked = 0;
        hzbLastVerdict.Clear();
    }

    LodGpuDepthPyramid? depthPyramid;

    /// <summary>CPU time spent asking for the pyramid, separate from the GPU time it takes.</summary>
    public LodPhaseCost HzbCost;
    internal LodHzbBuildResult LastHzbBuild { get; private set; }

    /// <summary>Owns the vertex array and the per-frame command and record buffers.</summary>
    LodGpuIndirectDrawer? indirectDrawer;
    LodGpuIndirectDrawer? packedDrawer;
    LodGpuCullPass? cullPass;
    readonly float[] cullViewProjection = new float[16];

    /// <summary>
    /// Whether depth verdicts may stop a cached section being drawn.
    ///
    /// OFF by default and not saved, like every switch that can change what reaches the
    /// screen. This is the first one whose failure mode is missing terrain rather than a
    /// slower frame, so it turns on only when someone asks for it, and only for that session.
    /// It needs the indirect path: culling works by zeroing an indirect command, and the
    /// established path has no commands to zero.
    /// </summary>
    public bool GpuCullEnabled { get; set; } =
        Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_CULL") != "0";

    /// <summary>
    /// Sections the indirect pass could not batch: the arenas do not hold them, or a
    /// multi-draw failed. They are drawn exactly as the established path draws them, in a
    /// second sub-pass, so partial arena coverage costs submissions rather than terrain.
    /// </summary>
    readonly List<long> indirectLeftovers = new();

    /// <summary>Far-bucket leftovers draw only after the mid-frame picture is taken.</summary>
    readonly List<long> indirectFarLeftovers = new();

    /// <summary>Keys written into this frame's command list, kept for the same reason.</summary>
    readonly List<long> indirectRecorded = new();

    /// <summary>Keys recorded into the far command list, retained for failure accounting.</summary>
    readonly List<long> indirectFarRecorded = new();

    /// <summary>True only inside the walk of an indirect opaque pass.</summary>
    bool indirectPassActive;

    /// <summary>
    /// Decided once, before anything this frame reads it. A path chosen halfway through a
    /// frame would draw some sections twice and others not at all, and occlusion queries
    /// resolved under one path would be applied under the other.
    /// </summary>
    bool indirectDrawingThisFrame;
    string? reportedRequiredTerrainFailure;

    /// <summary>
    /// Fixed once per frame. The split is meaningful only when both indirect drawing and
    /// depth culling are active; otherwise one ordinary command list is the complete path.
    /// </summary>
    bool depthSplitThisFrame;
    float appliedZFar;
    Vec3d camPos = new();

    /// <summary>Dev/testing: keep the game unpaused even without window focus.</summary>
    public bool AutoUnpause;

    /// <summary>Saved player choice: keep the bounded LOD work source moving behind pause UI.</summary>
    public bool ContinueLoadingWhilePaused { get; set; } = true;

    /// <summary>
    /// Supplied by the owning mod system. Called only on the render/main thread while paused;
    /// that owner rate-limits it to the normal game-tick cadence.
    /// </summary>
    public Action? PausedLoadingPump { get; set; }

    // Live seasonal state, refreshed incrementally and fed to the shader as uniforms.
    const long SeasonalRefreshIntervalMs = 30_000;
    const long SnowObservationRefreshDebounceMs = 1_000;
    float snowLineY = 99999;
    float pendingSnowLineY = 99999;
    long lastSeasonRefreshMs;
    bool seasonalStateInitialized;
    bool seasonalRefreshActive;
    int seasonalRefreshSlot;
    int seasonalRefreshX;
    int seasonalRefreshZ;
    double seasonalRefreshTotalDays;
    long seasonalRefreshSnowObservationRevision;
    long publishedSnowObservationRevision;
    readonly BlockPos climatePos = new(0, 0, 0);
    readonly ClimateCondition snowClimateLow = new();
    readonly ClimateCondition snowClimateHigh = new();

    /// <summary>Optional hard cap in blocks; 0 = unlimited (render every cached section).</summary>
    public int FarViewDistanceCap = 0;

    /// <summary>Current far edge in blocks: the farthest loaded LOD data, independent of the vanilla view distance.</summary>
    public float EffectiveFarDistance { get; private set; } = 3000;

    public int MeshCount => sectionMeshes.Count;
    public int LastDrawCount { get; private set; }

    /// <summary>Sections selected by the walk but skipped this frame as off-screen.</summary>
    public int LastCulledCount { get; private set; }

    /// <summary>
    /// Frames that returned before the selection walk because no mesh existed yet. On an
    /// ordinary join this is a handful; a large number means the bootstrap never started.
    /// </summary>
    public long FramesWithoutMeshes { get; private set; }

    /// <summary>Whole quadtree subtrees rejected before descent this frame.</summary>
    public int LastTraversalCulledCount { get; private set; }

    readonly LodFrustum frustum = new();
    int worldHeight = 1024;

    /// <summary>Blocks from bedrock to build limit, as the block accessor reports it.</summary>
    public int WorldHeight => worldHeight;


    /// <summary>
    /// Why each coarse node in the current draw list is not descending. Written for one
    /// specific failure: a node drawn far below its wanted level, with the pipeline idle, so
    /// no amount of waiting changes it. Reports each child's actual state rather than an
    /// inference - which is what three wrong diagnoses in a row cost.
    /// </summary>
    public string ExplainCoarseDraws(double px, double pz, int maxNodes = 6)
    {
        var sb = new System.Text.StringBuilder();
        int shown = 0;

        foreach (long key in drawList)
        {
            int level = LodWorld.KeyLevel(key);
            int wanted = WantedLevel(NearestDistanceTo(key));
            if (level <= wanted || shown >= maxNodes) continue;

            shown++;
            double dist = Math.Sqrt(LodWorld.NearestDistanceSqTo(key, px, pz));
            sb.Append($"\n  L{level} at {LodWorld.KeySx(key) * LodWorld.KeyFootprintBlocks(key)},")
              .Append($"{LodWorld.KeySz(key) * LodWorld.KeyFootprintBlocks(key)} dist {(int)dist} wants L{wanted}:");

            for (int qz = 0; qz < 2; qz++)
            {
                for (int qx = 0; qx < 2; qx++)
                {
                    long ck = LodWorld.ChildKey(key, qx, qz);
                    string state;
                    if (!world.HasDataSet.Contains(ck)) state = "no-data";
                    else if (!world.Sections.TryGetValue(ck, out LodSection? cs))
                    {
                        state = world.LoadsInFlight.Contains(ck) ? "loading"
                            : world.LoadFailed.Contains(ck) ? "load-failed"
                            : "not-resident";
                    }
                    else if (cs.CapturedColumns == 0) state = "empty";
                    else if (!HasAnyMesh(ck)) state = world.RenderDirty.Contains(ck) ? "meshing" : "no-mesh!";
                    else state = "ok";
                    sb.Append(' ').Append(state);
                }
            }
            sb.Append("; covering ").Append(LodStandInPolicy.Describe(StandInQuartersFor(key)));
        }

        return shown == 0 ? "no coarse draws: every drawn node is at or below its wanted level" : sb.ToString();
    }

    public string DescribeDrawnLevels()
    {
        var counts = new int[LodWorld.MaxLevel + 1];
        foreach (long key in drawList) counts[LodWorld.KeyLevel(key)]++;
        return string.Join(" ", counts.Select((c, i) => $"L{i}:{c}"));
    }

    readonly LodTintRegistry tints;

    /// <summary>
    /// The slot table version currently on the GPU. ONE field, not one per program: it lives in
    /// a texture, so it is not part of any program's state and a single upload serves all three
    /// programs.
    ///
    /// It was one per program while the table was uniform arrays, which are program state, and
    /// carrying that shape forward to a texture made the old height ramp re-upload on every
    /// frame - the upload method assigned the field while the two indirect programs passed their
    /// own counter by ref, so their guard never closed. See G143.
    /// </summary>
    int uploadedSlotTableVersion = -1;
    int uploadedClimateMapVersion = -1;

    /// <summary>
    /// Climate over the world around the player, so a vertex can be tinted by the climate it
    /// stands in rather than the climate under the player's feet. 128 cells of 64 blocks reach
    /// 4,096 blocks each way; terrain past that clamps to the window edge, which continues the
    /// spread rather than drawing an edge.
    /// </summary>
    readonly LodClimateField climateField = new(128);
    readonly LodClimateMapPages climatePages = new();
    readonly LodSeasonMapPages seasonPages = new();

    /// <summary>
    /// Climate the client has answered before, so exploration is cumulative rather than
    /// relearned every session. See <see cref="LodClimateMemory"/>.
    /// </summary>
    public LodClimateMemory ClimateMemory { get; } = new();

    /// <summary>
    /// Where remembered climate is read and written. Set by the mod system once the pipeline
    /// exists; null on a client with no cache, where the field simply learns nothing durable.
    /// </summary>
    public LodStorageThread? ClimateStorage { get; set; }
    LoadedTexture? climateFieldTexture;
    LoadedTexture? climateMapTexture;
    LoadedTexture? seasonColumnTexture;
    int uploadedSeasonMapVersion = -1;
    int uploadedSeasonColumn = -1;

    /// <summary>
    /// How much of vanilla's per-position seasonal variation distant leaves carry, 0 to 1.
    ///
    /// The game picks a row of the seasonal map per position, and for leaves that noise runs at
    /// a 2-to-6 block scale - tree sized. Averaging the rows is right once one voxel is a whole
    /// neighbourhood (G47) and wrong while a tree is still a tree, so the two are blended by the
    /// voxel's own size and this scales the near end of that. 1 is vanilla's full spread.
    /// </summary>
    public float LeafSeasonVariation { get; set; } = 1f;

    /// <summary>
    /// The voxel size, in blocks, at which that variation has faded to nothing. 2 confines it to
    /// level 0; 8 fades it out across levels 0 to 2.
    ///
    /// This is the feature's price, not a look setting with a cost attached. A section pays for
    /// the variation per vertex, and a section is the same 64x64 grid whatever level it holds,
    /// so the bill scales with how many LEVELS are included - and the near levels are not the
    /// small ones. A measured view drew 31 sections at level 0 and 57 at level 1. Fading out at
    /// 8 blocks therefore charged three levels and 97 of 110 drawn sections for a variation that
    /// is already at 0.57 strength by the third of them.
    ///
    /// Level 0 is where it is worth paying for: one voxel is one block, so the 2-to-6 block leaf
    /// noise is reproduced at the scale vanilla draws it, beside the vanilla trees it has to
    /// match. Defaults to 2 for that reason; `.dltint leafwide on` restores the wider fade.
    /// </summary>
    public float LeafSeasonFadeBlocks { get; set; } = 2f;

    /// <summary>
    /// Swap the engine's value noise for a cheap position hash while leaving the seasonal map
    /// fetch scattered. A MEASUREMENT INSTRUMENT: it draws the wrong rows on purpose.
    ///
    /// The per-tree variation costs about 10 fps at level 0, and it spends that on two things -
    /// two value noises per vertex and one scattered fetch into the paged seasonal map. Baking
    /// the row into the mesh at build time can only remove the first, so the split decides
    /// whether that work is worth starting. Turning this on prices the noise alone.
    ///
    /// Session-only and deliberately unsaved: a wrong-coloured world must not survive a
    /// restart. See `.dltint leafcheap`.
    /// </summary>
    public bool LeafSeasonCheapRow { get; set; }
    LoadedTexture? slotTableTexture;

    /// <summary>
    /// Tint distant terrain from the climate field instead of from the height ramp. The ramp
    /// stays until this is judged in place: it is one colour per block type sampled where the
    /// player stands, which is exactly the defect this replaces, but it is also the behaviour
    /// every accepted build has drawn. See .dltint field.
    /// </summary>
    public bool ClimateFieldTint { get; set; } = true;

    /// <summary>
    /// The voxel size, in blocks, past which distant ground stops working out its own climate
    /// and wears the flat per-slot fallback instead. Zero spends it at every level, which is
    /// what 0.3.179 through 0.3.194 did.
    ///
    /// <para>
    /// This is the climate lookup's own version of <see cref="LeafSeasonFadeBlocks"/>, and it
    /// exists for the same reason: a section costs the same per vertex whatever level it
    /// holds, so the bill is set by how many LEVELS pay, not by how much area they cover. The
    /// lookup is four climate-field samples, four climate-map samples, a season row and a
    /// weight, per vertex - measured as the largest single piece of the 0.3.127 -> 0.3.194
    /// frame-time regression.
    /// </para>
    /// <para>
    /// The default spends it out to 4 blocks per voxel - levels 0 to 2 - because that is
    /// where a biome edge is still a shape a player can see. Past it the two climates being
    /// chosen between have usually already converged, and where they have not the cut falls
    /// on a section boundary far enough out that the fallback and the finished tint differ by
    /// less than the fog over them. It is a switch because only looking at a horizon with a
    /// biome edge on it can say whether that is true in a given world; see `.dltint
    /// climatewide`.
    /// </para>
    /// </summary>
    public float ClimateFadeBlocks { get; set; } = LodVisualSettings.CloseClimateTintRange;

    /// <summary>
    /// Hand every patch learned since the last call to the storage worker. Nothing here touches
    /// SQLite; the worker owns the connection, and its queue drains on shutdown so a flush
    /// before that disposal still lands.
    ///
    /// Public because the ordering matters and only the caller knows it: this must run BEFORE
    /// the pipeline closes, since closing disposes the worker this hands to. A patch whose
    /// values did not change is not dirty and costs nothing, which is the common case once an
    /// area has been visited.
    /// </summary>
    public void SaveClimateMemory()
    {
        if (ClimateStorage == null) return;
        foreach (long dirty in ClimateMemory.TakeDirtyKeys())
        {
            byte[]? blob = ClimateMemory.Encode(dirty);
            if (blob != null) ClimateStorage.SaveClimate(dirty, blob);
        }
    }

    /// <summary>
    /// Raw temperature, rainfall and current snow threshold for one cell centre, or false where
    /// neither the client nor remembered climate can answer. Cached rather than built per row:
    /// it is called once per cell.
    /// </summary>
    LodClimateField.SeasonalSampler? climateSampler;
    LodClimateField.SeasonalSampler ClimateSampler => climateSampler ??=
        (int bx, int bz, out byte temperature, out byte rainfall, out float snowLine,
            out bool snowStateIsObserved) =>
        {
            temperature = 0;
            rainfall = 0;
            snowLine = NoSnowLineY();
            snowStateIsObserved = false;
            int cellX = LodClimateField.CellOf(bx);
            int cellZ = LodClimateField.CellOf(bz);

            // The client first: it is authoritative where it can answer, and answering also
            // teaches the memory, which is what makes coverage grow with exploration.
            if (LodClimateProbe.TryRawClimate(capi.World, bx, bz, out float t, out float r))
            {
                temperature = (byte)Math.Clamp((int)MathF.Round(t), 0, 255);
                rainfall = (byte)Math.Clamp((int)MathF.Round(r), 0, 255);
                ClimateMemory.Learn(cellX, cellZ, temperature, rainfall);
                snowLine = CalculateSnowLine(bx, bz, temperature, rainfall, seasonalRefreshTotalDays);
                snowStateIsObserved = world.HasObservedSnowStateAt(bx, bz);
                return true;
            }

            // Then what we were told the last time we were here. A real answer from a previous
            // session beats spreading a neighbour's climate across it.
            if (ClimateMemory.TryGet(cellX, cellZ, out temperature, out rainfall))
            {
                snowLine = CalculateSnowLine(bx, bz, temperature, rainfall, seasonalRefreshTotalDays);
                snowStateIsObserved = world.HasObservedSnowStateAt(bx, bz);
                return true;
            }

            // Otherwise ask the store, once per patch, and let the spread cover this refresh.
            if (ClimateStorage != null && ClimateMemory.ShouldRequest(cellX, cellZ, out long key))
                ClimateStorage.RequestClimate(key);
            return false;
        };

    public LodTerrainRenderer(
        ICoreClientAPI capi,
        LodWorld world,
        LodWorker worker,
        LodTintRegistry tints,
        Func<long>? currentWorldEpoch = null,
        LodBlockTextureRegistry? blockTextures = null)
    {
        this.capi = capi;
        this.world = world;
        this.worker = worker;
        this.tints = tints;
        this.blockTextures = blockTextures;
        this.currentWorldEpoch = currentWorldEpoch ?? (() => 0);
        bool gpuTimingRequested =
            Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_STATS") == "1";
        gpuRendererPreference =
            Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_RENDERER");
        gpuFailureInjection = LodGpuFailureInjection.Parse(
            Environment.GetEnvironmentVariable(LodGpuFailureInjection.EnvironmentVariable));
        if (gpuFailureInjection.Stage == LodGpuFailureStage.Invalid)
        {
            capi.Logger.Warning(
                "[Landscapes] Ignoring unknown {0} value '{1}'. Expected arena, shader, "
                + "depth-copy, draw, or off.",
                LodGpuFailureInjection.EnvironmentVariable,
                gpuFailureInjection.Requested);
        }
        else if (gpuFailureInjection.Armed)
        {
            capi.Logger.Warning(
                "[Landscapes] Phase 9 GPU failure injection armed for {0}; this is a "
                + "test-only session and cached opaque terrain must fail visibly.",
                gpuFailureInjection.Stage);
        }
        gpuTelemetry = new LodGpuTelemetry(
            gpuTimingRequested,
            LodRenderPathPolicy.RequestsRuntimeValidation(gpuRendererPreference),
            message => capi.Logger.Notification("{0}", message),
            message => capi.Logger.Warning("{0}", message),
            GpuFrameSize);
        gpuShadow = new LodGpuShadowRenderPath();
        renderPaths = new LodRenderPathCoordinator(
            new LodLegacyRenderPath(
                PublishLegacy,
                RemoveMeshesLegacy,
                PrepareLegacyFrame,
                DrawPackedOpaque,
                DrawWaterSurface,
                ClearMeshesLegacy),
            gpuShadow,
            message => capi.Logger.Warning("{0}", message));
        keepRenderDirty = KeepRenderDirty;
        renderDirtyBlocked = RenderDirtyBlocked;
        radialDemandReady = HasAnyMesh;
        radialDemandPending = RadialDemandPending;
        radialDemandTerminal = RadialDemandTerminal;
        startRadialDemand = RequestMesh;
        maxWorkerMeshBacklog = worker.MeshThreads * MeshBacklogPerThread;
        world.SectionBecameResident += OnSectionBecameResident;

        // Shaders are first built by the engine's reload, never here. This constructor runs
        // inside the engine's StartModsFully, which is followed at once by its ReloadShaders
        // ("Reloaded shaders now with mod assets"). Registering a program makes the engine read
        // its files at that moment against an include table it rebuilds only on a full load, so
        // a load from here could not find lodterrainbody/lodwaterbody: six programs failed to
        // link every session, and the engine's "Captured N issues during startup" summary
        // printed the failures a second time at world join (2026-10-13; see LoadShader).
        capi.Event.ReloadShader += LoadShader;
        capi.Event.ChunkDirty += OnReadinessChunkDirty;
        RegisterRenderer();
        RegisterWaterDepthStages();
        RegisterSilhouetteFogStage();
    }

    void RegisterRenderer()
    {
        capi.Event.RegisterRenderer(this, EnumRenderStage.Opaque, "dodenslandscapes-lod");
        rendererRegistered = true;
    }

    void OnReadinessChunkDirty(Vec3i chunkCoord, IWorldChunk chunk, EnumChunkDirtyReason reason)
    {
        // ClientWorldMap raises this from the main thread after installing the chunk but
        // before tessellation. Store only value coordinates; never retain the engine chunk.
        var cell = new VanillaChunkCell(chunkCoord.X, chunkCoord.Y, chunkCoord.Z);
        readiness?.EnqueueCandidate(cell, frameCounter, engineEvent: true);
        // Before tessellation means the probe above usually finds nothing to hand over yet.
        // Watched, the cell is rechecked every frame until its mesh is up instead of waiting
        // for the interior sweep while vanilla and the cache both draw it. Recorded whether or
        // not fast confirmation is on, so `.dlstandin off` still measures the old delay.
        readiness?.Watch(cell, Environment.TickCount64);
        NoteGapChunkEvent(chunkCoord.X, chunkCoord.Y, chunkCoord.Z, reason);
    }

    /// <summary>
    /// How many times shaders have been built. Until 0.4.140 the first build ran from the
    /// constructor, before the engine had refilled its shader-include table from mod assets,
    /// so it always failed and was treated as provisional (quiet), while the engine's own
    /// link errors and its end-of-startup summary still put twelve red lines in every log.
    /// Now the first build is the engine's ReloadShader event, the include table is current,
    /// and every attempt reports at full volume. <see cref="EnsureShadersLoaded"/> covers a
    /// session where that event never came.
    /// </summary>
    int shaderLoadAttempts;

    /// <summary>
    /// Safety net for the first frame: if the engine's reload never reached this renderer,
    /// build the shaders now rather than draw with none.
    /// </summary>
    void EnsureShadersLoaded()
    {
        if (shaderLoadAttempts > 0) return;
        capi.Logger.Warning("[Landscapes] the engine's shader reload had not reached this mod by the "
            + "first frame; building its shaders now.");
        LoadShader();
    }

    public bool LoadShader()
    {
        shaderLoadAttempts++;
        prog = capi.Shader.NewShaderProgram();
        prog.AssetDomain = "dodenslandscapes";

        prog.VertexShader = capi.Shader.NewShader(EnumShaderType.VertexShader);
        prog.FragmentShader = capi.Shader.NewShader(EnumShaderType.FragmentShader);

        capi.Shader.RegisterFileShaderProgram("lodterrain", prog);

        // The shaders carry their own `const int TINT_SLOTS`, because this game version
        // exposes no way to inject a #define, and a mismatch decodes water as opaque and
        // thin plants as water with no compile error. That used to be guarded here by
        // comparing MaxSlots against a hand-maintained C# mirror of the shader's value -
        // which is two constants in one file, and so could never notice a shader being
        // edited. The compiler agreed: the branch raised CS0162, unreachable code.
        // The check that works reads the shader files, in the fast tier of check.sh.

        // A fresh program object does not invalidate a texture, but the re-upload is one
        // frame's worth of nothing and this stays honest about the programs being new.
        uploadedSlotTableVersion = -1;
        uploadedClimateMapVersion = -1;
        uploadedSeasonMapVersion = -1;
        uploadedSeasonColumn = -1;
        WarnIfIncludeMissing(prog, "lodterrain");
        shaderOk = prog.Compile();
        if (shaderOk) LodTerrainSamplerLayout.Configure(prog);
        if (!shaderOk)
            capi.Logger.Error("[Landscapes] lodterrain shader failed to compile; LOD rendering disabled");

        // The expanded indirect variant remains a diagnostic while its old representation
        // is removed. It is compiled for comparison but is no longer a visible fallback.
        indirectProg = capi.Shader.NewShaderProgram();
        indirectProg.AssetDomain = "dodenslandscapes";
        indirectProg.VertexShader = capi.Shader.NewShader(EnumShaderType.VertexShader);
        indirectProg.FragmentShader = capi.Shader.NewShader(EnumShaderType.FragmentShader);
        capi.Shader.RegisterFileShaderProgram("lodterrainindirect", indirectProg);
        WarnIfIncludeMissing(indirectProg, "lodterrainindirect");

        // Same body again, with the packed input block enabled. Per-section attributes,
        // every fragment calculation and every uniform remain shared with the already
        // played indirect path; only vertex acquisition changes.
        packedProg = capi.Shader.NewShaderProgram();
        packedProg.AssetDomain = "dodenslandscapes";
        packedProg.VertexShader = capi.Shader.NewShader(EnumShaderType.VertexShader);
        packedProg.FragmentShader = capi.Shader.NewShader(EnumShaderType.FragmentShader);
        capi.Shader.RegisterFileShaderProgram("lodterrainpacked", packedProg);
        WarnIfIncludeMissing(packedProg, "lodterrainpacked");
        // The depth-pyramid reduction. It shares nothing with the terrain shaders - no
        // include, no tint constant, no vertex format - because it draws one attributeless
        // triangle and writes depth. A driver that refuses it costs the pyramid and
        // nothing else, and the established renderer never learns it existed.
        hzbProg = capi.Shader.NewShaderProgram();
        hzbProg.AssetDomain = "dodenslandscapes";
        hzbProg.VertexShader = capi.Shader.NewShader(EnumShaderType.VertexShader);
        hzbProg.FragmentShader = capi.Shader.NewShader(EnumShaderType.FragmentShader);
        capi.Shader.RegisterFileShaderProgram("hzbreduce", hzbProg);
        hzbShaderOk = hzbProg.Compile();
        if (!hzbShaderOk)
            capi.Logger.Warning(
                "[Landscapes] the hzbreduce shader failed to compile; depth-pyramid "
                + "occlusion is unavailable and nothing else changes");

        indirectShaderOk = indirectProg.Compile();
        if (indirectShaderOk) LodTerrainSamplerLayout.Configure(indirectProg);
        if (!indirectShaderOk)
            capi.Logger.Warning(
                "[Landscapes] the indirect lodterrain variant failed to compile; "
                + "the retired expanded diagnostic is unavailable");

        packedShaderOk = packedProg.Compile();
        if (packedShaderOk) LodTerrainSamplerLayout.Configure(packedProg);
        LoadWaterDepthShader();
        LoadWaterSurfaceShader();
        LoadWaterOitShader();
        LoadSkyWitnessShader();
        LoadTaaShaders();
        LoadSilhouetteShader();
        if (gpuFailureInjection.Take(LodGpuFailureStage.FastShaders))
        {
            indirectShaderOk = false;
            packedShaderOk = false;
            capi.Logger.Error(
                "[Landscapes] REQUIRED PACKED TERRAIN PATH FAILED: Phase 9 injected "
                + "a shader failure; cached opaque terrain will be suppressed until reload.");
        }
        if (!packedShaderOk)
            capi.Logger.Error(
                "[Landscapes] REQUIRED PACKED TERRAIN PATH FAILED: the packed shader "
                + "did not compile; cached opaque terrain will be suppressed.");

        // Said once, when it actually worked, so the log carries positive evidence rather
        // than only the absence of a complaint.
        //
        // G68 asked for this and only half of it was ever built: the indirect variant
        // reported its success while the MAIN program - the one whose failure means no
        // distant terrain at all - reported nothing but its failure. A log that can only
        // complain cannot answer "did this work". That cost reading time in session 42 and
        // again in session 112, back when a doomed first attempt filled the log with errors.
        //
        // Re-armed on a real failure, so a driver that breaks on one reload and recovers on
        // the next says so instead of going quiet after its first success.
        if (!shaderOk) reportedShadersOk = false;

        if (shaderOk && !reportedShadersOk)
        {
            reportedShadersOk = true;
            capi.Logger.Notification(
                "[Landscapes] shaders ready after {0} attempt(s): distant terrain {1}, "
                + "packed path {2}, indirect path {3}, depth pyramid {4}.",
                shaderLoadAttempts,
                shaderOk ? "ok" : "FAILED",
                packedShaderOk ? "ok" : "unavailable",
                indirectShaderOk ? "ok" : "unavailable",
                hzbShaderOk ? "ok" : "unavailable");
        }

        return shaderOk;
    }

    /// <summary>
    /// Both programs are three-line files whose real content arrives through one #include,
    /// and the engine resolves those against a table it refills only on a full shader load
    /// (ShaderRegistry, decompiled 1.22.7). If our body were ever missing from that table the
    /// engine would log a bare "Include file not found. Ignoring." and hand the compiler an
    /// empty shader - which then fails for a reason that names nothing.
    ///
    /// Naming it here costs one substring search per shader load and turns that into a
    /// sentence someone can act on.
    /// </summary>
    void WarnIfIncludeMissing(IShaderProgram program, string name)
    {
        string? vertex = program.VertexShader?.Code;
        string? fragment = program.FragmentShader?.Code;
        bool spliced = vertex != null && vertex.Contains("TINT_SLOTS", StringComparison.Ordinal)
            && fragment != null && fragment.Contains("TINT_SLOTS", StringComparison.Ordinal);
        if (spliced) return;

        capi.Logger.Error(
            "[Landscapes] the {0} shader body was not spliced in: the engine did not "
            + "find lodterrainbody.vsh/.fsh among its shader includes. Cached terrain will "
            + "not draw. This is ours, not your setup.", name);
    }

    public void ApplyZFar()
    {
        // Silhouettes widen the projection only while they draw something (plan 4.10).
        float needed = Math.Max(LodFarDistance.RequiredProjection(EffectiveFarDistance),
            SilhouetteProjectionDistance);
        LodFarPlaneUpdate update = farPlaneState.Update(needed, Environment.TickCount64);
        var clientMain = (ClientMain)capi.World;

        if (!update.Changed && clientMain.MainCamera.ZFar >= update.Distance
            && appliedZFar == update.Distance) return;

        clientMain.MainCamera.ZFar = update.Distance;
        capi.Render.Reset3DProjection();
        appliedZFar = update.Distance;
        ProjectionResetCount++;
    }

    void UpdateEffectiveFarDistance(float vanillaViewDistance)
    {
        if (meshBounds.Dirty) meshBounds.Rebuild(sectionMeshes.Keys, waterMeshes.Keys);
        double farthest = meshBounds.FarthestDistanceTo(camPos.X, camPos.Z);
        EffectiveFarDistance = LodFarDistance.Effective(farthest, vanillaViewDistance, FarViewDistanceCap);
    }

    // ---- Detail selection (quadtree walk) ----

    double NearestDistanceTo(long key)
    {
        int footprint = LodWorld.KeyFootprintBlocks(key);
        double minX = LodWorld.KeySx(key) * (double)footprint;
        double minZ = LodWorld.KeySz(key) * (double)footprint;
        double dx = Math.Max(0, Math.Max(minX - camPos.X, camPos.X - (minX + footprint)));
        double dz = Math.Max(0, Math.Max(minZ - camPos.Z, camPos.Z - (minZ + footprint)));
        return Math.Sqrt(dx * dx + dz * dz);
    }

    static int WantedLevel(double distance) => LodWorld.WantedLevelFor(distance);

    /// <summary>
    /// Squared distance to the section's nearest edge. The hot paths compare and pick a
    /// level, and both work on the square, so neither needs the root.
    /// </summary>
    double NearestDistanceSqTo(long key) => LodWorld.NearestDistanceSqTo(key, camPos.X, camPos.Z);

    bool HasAnyMesh(long key) => sectionMeshes.ContainsKey(key) || waterMeshes.ContainsKey(key);

    /// <summary>
    /// The quarters whose finer piece is in view but has no mesh yet, one bit per child as
    /// <see cref="LodStandInPolicy.QuarterBit"/> numbers them. Zero means every visible finer
    /// piece is ready and the parent may stop drawing. Every child is examined, uncovered or
    /// not, so the coarse-wait counters see all of them.
    /// </summary>
    int UncoveredVisibleChildren(long key)
    {
        int uncovered = 0;
        for (int qz = 0; qz < 2; qz++)
        {
            for (int qx = 0; qx < 2; qx++)
            {
                long ck = LodWorld.ChildKey(key, qx, qz);
                if (!world.HasDataSet.Contains(ck)) continue;
                // Full height here, deliberately, while the walk below uses the aggregate.
                // This gate does not decide what is drawn; it decides whether the parent
                // may STOP drawing its own coarse mesh over this quadrant. Those are not
                // the same question, and the aggregate cannot answer the second one: it
                // covers the meshes resident under the child, not the parent's coarser
                // rendition of the same ground, which has its own vertical extent. A child
                // holding nothing but a deep cave mesh would be rejected vertically, the
                // parent would descend believing the quadrant covered, and the surface the
                // parent was drawing there would become a hole.
                if (!LodTraversalPolicy.NodeInView(frustum, ck,
                    camPos.X, camPos.Y, camPos.Z, worldHeight)) continue;

                // A resident section with nothing captured will never get a mesh --
                // RequestMesh refuses it, by design. Counting it as uncovered pins the
                // parent at its own level permanently, which is not the transient
                // wait-for-mesh this gate exists for: it showed up as hard-edged coarse
                // plates over ground whose finer data was already loaded, with the whole
                // pipeline idle. Treat it like an absent child instead.
                if (world.Sections.TryGetValue(ck, out LodSection? child) && child.CapturedColumns == 0)
                {
                    continue;
                }

                // Gate meshes are load-bearing even when never drawn. Residency is
                // maintained independently by the distance-based eviction sweep.
                if (!HasAnyMesh(ck))
                {
                    // The radial planner owns demand eligibility. The parent keeps
                    // covering until that orientation-independent wave reaches this gate.
                    uncovered |= LodStandInPolicy.QuarterBit(qx, qz);
                    CountCoarseWait(ck);
                }
            }
        }
        return uncovered;
    }

    /// <summary>
    /// Records why a child had no mesh, and therefore why its parent is still drawing the
    /// area coarsely. Human testing reported terrain becoming coarser than expected during
    /// fast flight, and the four causes want four different fixes: waiting on storage,
    /// waiting on a mesh worker, waiting for a scheduling slot, or nothing pending at all,
    /// which would mean the obligation was dropped. Guessing between them from a frame rate
    /// is how the wrong one gets optimised.
    ///
    /// Only reached on the uncovered path, which is rare once terrain settles, and the
    /// lookups sit beside a RequestMesh call that already costs more.
    /// </summary>
    void CountCoarseWait(long childKey)
    {
        if (world.LoadsInFlight.Contains(childKey)) CoarseWaitingLoad++;
        else if (meshJobInFlight.Contains(childKey)) CoarseWaitingMesh++;
        else if (world.RenderDirty.Contains(childKey)) CoarseWaitingSchedule++;
        else CoarseWaitingOther++;
    }

    bool CollectDrawNodes(long key)
    {
        // A rejected parent contains every descendant, so the conservative p-vertex box
        // test makes it safe to skip all draw selection, mesh demand, and child traversal.
        LodHeightSpan subtree = SubtreeSpan(key);
        if (!LodTraversalPolicy.NodeInView(frustum, key,
            camPos.X, camPos.Y, camPos.Z, worldHeight, subtree))
        {
            traversalCulledThisFrame++;

            // Only on the reject path, and only for nodes the aggregate is what rejected:
            // how many of them the old full-height box would have traversed, and how many
            // resident meshes went with them. The node count alone cannot separate
            // rejecting one leaf from rejecting a whole L6 quadrant, and that difference
            // is the entire question this work is funded on.
            if (subtree.HasGeometry
                && LodTraversalPolicy.NodeInView(frustum, key,
                    camPos.X, camPos.Y, camPos.Z, worldHeight))
            {
                subtreeCulledThisFrame++;
                subtreeCulledMeshesThisFrame += subtreeHeights.Of(key).Meshes;
            }
            return false;
        }

        bool hasMesh = HasAnyMesh(key);
        int level = LodWorld.KeyLevel(key);
        int wanted = LodWorld.WantedLevelForSq(NearestDistanceSqTo(key));

        if (level > 0 && level > wanted)
        {
            // Asked with or without a mesh of our own, as it always was, so the coarse-wait
            // counters describe the same population they did before stand-ins were split.
            int uncovered = UncoveredVisibleChildren(key);
            if (!hasMesh) return CollectChildren(key, skipQuarters: 0);

            if (uncovered == 0)
            {
                if (CollectChildren(key, skipQuarters: 0)) return true;
                // Every finer piece is ready but none of them is in view: this one is.
                drawList.Add(key);
                return true;
            }

            // Standing in for finer terrain that is not ready. A parent covers until its
            // children are ready, so this can be coarse but never a hole - and it covers only
            // the quarters that are waiting. The other quarters draw their own finer terrain,
            // and a kilometre-wide section no longer puts coarse columns over all the ground
            // around the player because one new piece appeared at the edge of what is loaded.
            // Only clustered opaque geometry can be limited to a quarter, so any other opaque path
            // - a diagnostic override - keeps whole stand-ins rather than drawing coarse ground
            // over the finer ground its ready quarters would then also draw.
            int quarters = LodStandInPolicy.StandInQuarters(uncovered,
                StandInFix && clusterDrawingThisFrame);
            if (quarters != 0)
            {
                CollectChildren(key, skipQuarters: quarters);
                standInQuarters[key] = (byte)quarters;
            }
            drawList.Add(key);
            return true;
        }

        if (level > 0 && !hasMesh) return CollectChildren(key, skipQuarters: 0);

        if (hasMesh)
        {
            drawList.Add(key);
            return true;
        }
        return false;
    }

    /// <summary>
    /// Walks the children that exist, except those in <paramref name="skipQuarters"/>, and says
    /// whether any of them drew.
    /// </summary>
    bool CollectChildren(long key, int skipQuarters)
    {
        bool anyChildDrew = false;
        for (int qz = 0; qz < 2; qz++)
        {
            for (int qx = 0; qx < 2; qx++)
            {
                if ((skipQuarters & LodStandInPolicy.QuarterBit(qx, qz)) != 0) continue;
                long ck = LodWorld.ChildKey(key, qx, qz);
                if (world.HasDataSet.Contains(ck)) anyChildDrew |= CollectDrawNodes(ck);
            }
        }
        return anyChildDrew;
    }

    /// <summary>
    /// This frame's stand-ins that cover only some quarters, keyed by section. A section absent
    /// from it draws whole, which is every ordinary draw. Cleared with the draw list.
    /// </summary>
    readonly Dictionary<long, byte> standInQuarters = new();

    /// <summary>The quarters this frame draws of a section, or zero for all of it.</summary>
    int StandInQuartersFor(long key) =>
        standInQuarters.TryGetValue(key, out byte quarters) ? quarters : 0;

    /// <summary>
    /// Two fixes for coarse distant terrain showing over the game's own ground, switchable
    /// together for comparison in the same session (<c>.dlstandin</c>, not saved):
    /// <list type="bullet">
    /// <item>a coarse section standing in for finer terrain that is not ready covers only the
    /// quarters that are waiting, rather than its whole area;</item>
    /// <item>a chunk the game has just announced is rechecked every frame until its mesh is up,
    /// rather than waiting for the interior sweep while both draw it.</item>
    /// </list>
    /// Off restores both behaviours as they were before 0.4.135.
    /// </summary>
    public bool StandInFix { get; set; } = true;

    /// <summary>
    /// Takes a cell off the fresh-chunk list for good: it cannot be handed to the game however
    /// often it is asked (out of range, or tesselated to nothing), so the ordinary sweeps own it
    /// again. Its announcement is dropped too, so a much later commit reports no wait.
    /// </summary>
    void EndFreshChunkWatch(VanillaChunkCell cell)
    {
        if (readiness == null) return;
        readiness.Unwatch(cell);
        readiness.TakeAnnouncement(cell);
    }

    /// <summary>A cell was just handed to the game: it leaves the list, and its announcement with it.</summary>
    void NoteFreshChunkConfirmed(VanillaChunkCell cell)
    {
        if (readiness == null) return;
        readiness.Unwatch(cell);
        readiness.TakeAnnouncement(cell);
    }

    /// <summary>
    /// Refresh the live tint table and snow line every 30 seconds. Each tint slot is sampled
    /// at two altitudes and interpolated per vertex, because the climate maps are keyed
    /// by temperature and temperature falls with height; the snow line extrapolates the
    /// same lapse rate to where it hits freezing.
    /// </summary>
    void RefreshSeasonalState()
    {
        long now = Environment.TickCount64;
        if (!seasonalRefreshActive)
        {
            bool normalRefreshDue = LodTintRegistry.RefreshDue(
                seasonalStateInitialized,
                tints.HasUnreadySlots,
                now - lastSeasonRefreshMs,
                SeasonalRefreshIntervalMs);
            bool snowObservationRefreshDue = seasonalStateInitialized
                && publishedSnowObservationRevision != world.SnowStateObservationRevision
                && now - world.LastSnowStateObservationChangeMs
                    >= SnowObservationRefreshDebounceMs;
            if (!normalRefreshDue && !snowObservationRefreshDue) return;

            seasonalRefreshActive = true;
            seasonalRefreshSlot = 1;   // skip slot 0, the identity tint
            seasonalRefreshX = (int)camPos.X;
            seasonalRefreshZ = (int)camPos.Z;
            seasonalRefreshTotalDays = capi.World.Calendar.TotalDays;
            seasonalRefreshSnowObservationRevision = world.SnowStateObservationRevision;
            // The engine's own season uniform, not Calendar.YearRel. The two differ by the
            // column jitter colormap.vsh applies and the CPU colour call does not; taking the
            // uniform means the mod reads the season map at the column the shader reads.
            tints.BeginRefresh(capi.World, capi.Render.ShaderUniforms.SeasonRel);
            climateField.BeginRefresh(seasonalRefreshX, seasonalRefreshZ,
                capi.World.SeaLevel - 64f, NoSnowLineY());
            pendingSnowLineY = CalculateSnowLine(seasonalRefreshX, seasonalRefreshZ);
        }

        // Game climate and colour-map APIs are owning-thread state. Keep them here, but
        // spread the calls over frames and publish only after the whole table is ready.
        //
        // One unit is one slot. The colour maps are no longer read per height - the shader
        // applies the lapse rate at the vertex's own height - so this is back to what it was
        // before the ramp, and the whole table republishes in as many frames as there are slots.
        int tintUnits = tints.RefreshUnitCount;
        if (seasonalRefreshSlot < tintUnits)
        {
            tints.RefreshUnit(capi.World, seasonalRefreshX, seasonalRefreshZ,
                seasonalRefreshSlot++);
            return;
        }

        // Then the climate field, one row of cells per frame on the same schedule. A row is
        // 128 map-region lookups of which only the loaded ones cost a climate read, so it is
        // cheaper per frame than a tint unit and it shares the tint's atomic publish.
        int fieldRow = seasonalRefreshSlot - tintUnits;
        if (fieldRow < climateField.RefreshRowCount)
        {
            climateField.RefreshRow(fieldRow, ClimateSampler);
            seasonalRefreshSlot++;
            return;
        }

        // What the refresh learned goes to disk, and anything the store answered while it ran
        // is folded in for the next one. Both are handed to the section worker; neither touches
        // SQLite from here.
        if (ClimateStorage != null)
        {
            while (ClimateStorage.TryTakeClimateResult(out LodClimateLoadResult loaded))
                ClimateMemory.Accept(loaded.Key, loaded.Data);
        }
        SaveClimateMemory();

        // Pages before the table: the table records which page each slot reads, so a page has
        // to exist before a slot is sent to it.
        tints.AdoptPages(climatePages, seasonPages);
        tints.CompleteRefresh();
        climateField.CompleteRefresh();
        snowLineY = pendingSnowLineY;
        publishedSnowObservationRevision = seasonalRefreshSnowObservationRevision;
        seasonalRefreshActive = false;
        seasonalStateInitialized = true;
        lastSeasonRefreshMs = now;
    }

    float CalculateSnowLine(int px, int pz)
    {
        float calculated = 99999;

        try
        {
            int seaLevel = capi.World.SeaLevel;
            climatePos.Set(px, seaLevel, pz);
            ClimateCondition? low = capi.World.BlockAccessor.GetClimateAt(climatePos);
            climatePos.Set(px, seaLevel + 150, pz);
            ClimateCondition? high = capi.World.BlockAccessor.GetClimateAt(climatePos);

            if (low == null || high == null || low.Temperature <= high.Temperature)
            {
                calculated = 99999; // no usable lapse rate → snow line disabled
            }
            else
            {
                float lapsePerBlock = (low.Temperature - high.Temperature) / 150f;
                calculated = seaLevel + (low.Temperature - (-1f)) / lapsePerBlock;
                calculated = GameMath.Clamp(calculated, seaLevel - 64, 99999);
            }
        }
        catch
        {
            calculated = 99999;
        }
        return calculated;
    }

    /// <summary>
    /// The current seasonal snow threshold for one known climate cell. The remembered field
    /// stores world-generation temperature and rainfall, not a seasonal result: that result
    /// changes with the calendar and must be rebuilt each refresh. The block-accessor overload
    /// used here is specifically for applying the game's climate events to previously obtained
    /// world-generation values, so remembered cells receive the same season and day/night
    /// adjustment as cells whose map region is currently loaded.
    /// </summary>
    float CalculateSnowLine(int px, int pz, byte rawTemperature, byte rawRainfall,
        double totalDays)
    {
        try
        {
            int seaLevel = capi.World.SeaLevel;
            PrepareSnowClimate(snowClimateLow, rawTemperature, rawRainfall, 0);
            PrepareSnowClimate(snowClimateHigh, rawTemperature, rawRainfall, 150);

            climatePos.Set(px, seaLevel, pz);
            ClimateCondition? low = capi.World.BlockAccessor.GetClimateAt(
                climatePos, snowClimateLow,
                EnumGetClimateMode.ForSuppliedDate_TemperatureOnly, totalDays);
            climatePos.Set(px, seaLevel + 150, pz);
            ClimateCondition? high = capi.World.BlockAccessor.GetClimateAt(
                climatePos, snowClimateHigh,
                EnumGetClimateMode.ForSuppliedDate_TemperatureOnly, totalDays);

            if (low == null || high == null || low.Temperature <= high.Temperature)
                return NoSnowLineY();

            float lapsePerBlock = (low.Temperature - high.Temperature) / 150f;
            float calculated = seaLevel + (low.Temperature - (-1f)) / lapsePerBlock;
            return GameMath.Clamp(calculated, seaLevel - 64f, NoSnowLineY());
        }
        catch
        {
            return NoSnowLineY();
        }
    }

    void PrepareSnowClimate(ClimateCondition climate,
        byte rawTemperature, byte rawRainfall, int blocksAboveSeaLevel)
    {
        climate.WorldGenTemperature = Climate.GetScaledAdjustedTemperatureFloatClient(
            rawTemperature, blocksAboveSeaLevel);
        climate.WorldgenRainfall = Climate.GetRainFall(
            rawRainfall, capi.World.SeaLevel + blocksAboveSeaLevel) / 255f;
    }

    float NoSnowLineY() => capi.World.BlockAccessor.MapSizeY + 24f;


    /// <summary>Transfer one planner or content-change obligation into exact dirty ownership.</summary>
    bool RequestMesh(long key)
    {
        if (meshJobInFlight.Contains(key)) return false;

        // RAM-evicted sections still count: HasDataSet says whether the subtree has
        // data at all; the scheduler reloads the row from disk when it picks the job.
        if (world.Sections.TryGetValue(key, out LodSection? section))
        {
            if (section.CapturedColumns == 0) return false;
        }
        else if (!world.HasDataSet.Contains(key))
        {
            return false;
        }

        return world.RenderDirty.Add(key);
    }

    bool RadialDemandPending(long key) => world.RenderDirty.Contains(key)
        || world.LoadsInFlight.Contains(key)
        || meshJobInFlight.Contains(key);

    bool RadialDemandTerminal(long key) => world.LoadFailed.Contains(key)
        || (world.Sections.TryGetValue(key, out LodSection? section)
            && section.CapturedColumns == 0);

    void PlanRadialDemand()
    {
        // The first configured transition is exactly the area whose final target is L0.
        // Prepare it under the player even while vanilla suppresses the cache: movement
        // may expose those meshes later, and discovering a coarse fallback at that point
        // looks like terrain regressed. Its independent planner lets this foundation keep
        // refining while outward coverage also advances.
        int foundationRadius = (int)Math.Ceiling(LodWorld.ThresholdForLevel(1));
        int outer = FarViewDistanceCap > 0
            ? FarViewDistanceCap
            : LodRadialDemandPlanner.UnlimitedPlanningRadius;
        foundationDemand.Pump(
            world.AvailableDataSet,
            world.AvailableDataRevision,
            LodWorld.DetailPolicyRevision,
            camPos.X,
            camPos.Z,
            0,
            foundationRadius,
            radialDemandReady,
            radialDemandPending,
            radialDemandTerminal,
            startRadialDemand);
        radialDemand.Pump(
            world.AvailableDataSet,
            world.AvailableDataRevision,
            LodWorld.DetailPolicyRevision,
            camPos.X,
            camPos.Z,
            foundationRadius,
            outer,
            radialDemandReady,
            radialDemandPending,
            radialDemandTerminal,
            startRadialDemand);
    }

    public string DescribeRadialDemand() => $"inner {foundationDemand.Describe()}; "
        + $"outward {radialDemand.Describe()}";

    public string DescribeTintReadiness() => seasonalRefreshActive
        ? $"{tints.ReadySlotCount}/{tints.SlotCount} slots ready (sampling)"
        : $"{tints.ReadySlotCount}/{tints.SlotCount} slots ready";

    void EvictStaleMeshes()
    {
        int available = meshEvictionOrder.Count;
        for (int n = 0; n < MeshEvictionChecksPerFrame && n < available; n++)
        {
            long key = meshEvictionOrder.Dequeue();
            if (!HasAnyMesh(key))
            {
                meshEvictionQueued.Remove(key);
                lastResidencyMs.Remove(key);
                continue;
            }
            if (!ShouldEvictMesh(key))
            {
                meshEvictionOrder.Enqueue(key);
                continue;
            }

            RemoveMeshes(key);
            meshEvictionQueued.Remove(key);
            lastResidencyMs.Remove(key);
            EvictedTotal++;
        }
    }

    bool ShouldEvictMesh(long key)
    {
        if (LodTraversalPolicy.WithinResidencyBand(key, camPos.X, camPos.Z))
        {
            lastResidencyMs[key] = Environment.TickCount64;
            return false;
        }

        return !lastResidencyMs.TryGetValue(key, out long last)
            || Environment.TickCount64 - last > EvictAfterMs;
    }

    void QueueMeshEvictionCheck(long key)
    {
        if (!meshEvictionQueued.Add(key)) return;
        meshEvictionOrder.Enqueue(key);
    }

    // ---- Mesh job scheduling + result upload ----

    /// <summary>
    /// Drop meaningless render-dirty entries: no live mesh AND finer than the level
    /// the walk wants there - meshing those wastes work. Entries at wanted level or
    /// COARSER must survive: they are draw targets or gate meshes the walk descends
    /// through (pruning gates stalls descent and freezes approached terrain at the
    /// coarse level it was first meshed at). New keys are indexed incrementally;
    /// existing priorities refresh only when the camera crosses a coarse cell.
    /// </summary>
    void PruneRenderDirty()
    {
        dirtyScheduler.Refresh(world.RenderDirty, camPos.X, camPos.Z,
            LodWorld.DetailPolicyRevision, keepRenderDirty);
    }

    bool KeepRenderDirty(long key) => HasAnyMesh(key)
        || LodWorld.KeyLevel(key) >= LodWorld.WantedLevelForSq(NearestDistanceSqTo(key));

    bool RenderDirtyBlocked(long key)
    {
        if (meshJobInFlight.Contains(key) || world.LoadsInFlight.Contains(key)) return true;

        // Palette resolution can discover a climate/season tint after the last completed
        // refresh. Keep this exact mesh obligation queued until every slot it uses has a
        // published value. A coarser parent remains live during replacement; on a cold
        // start, a few incremental tint frames are preferable to revealing white-tinted
        // ground and recolouring the whole horizon thirty seconds later.
        return world.Sections.TryGetValue(key, out LodSection? section)
            && !tints.SectionTintsReady(section);
    }

    void ScheduleMeshJobs()
    {
        if (world.RenderDirty.Count == 0 || worker.PendingMeshes >= maxWorkerMeshBacklog) return;

        bool includeTextureSlots = textureSlotsReady;

        // Walls draw at their own texture's colour - bark on a standing log, not the cut end
        // its block colour comes from. Resolved here because the registry is main-thread only.
        System.Func<int, (int X, int Z)>? wallColors = blockTextures == null
            ? null
            : wallColorsOf ??= blockId => blockTextures.WallColorsFor(capi.World.GetBlock(blockId));

        // Two budgets. Starting a background reload costs this thread almost nothing
        // (enqueue a key), whereas building a mesh snapshot is real work, so charging a
        // reload against the mesh budget throttled join fill-in badly: every section
        // needed two passes to appear and only four could be touched per frame.
        int meshBudget = MeshSchedulesPerFrame;
        int loadBudget = MeshLoadRequestsPerFrame;
        var snapshotBudget = new LodDrainBudget(
            MeshSnapshotMaxBytesPerFrame, MeshSnapshotMaxMillisecondsPerFrame);

        // Empty/missing candidates charge neither work budget, so keep a separate hard
        // examination limit. Priority lookup itself stays independent of dirty-set size.
        int examinations = MeshSchedulesPerFrame + MeshLoadRequestsPerFrame;
        int priorityExaminationBudget = examinations + meshJobInFlight.Count + world.LoadsInFlight.Count;
        while (meshBudget > 0 && loadBudget > 0 && examinations-- > 0
            && dirtyScheduler.TryTake(world.RenderDirty, keepRenderDirty,
                renderDirtyBlocked, priorityExaminationBudget,
                out long best))
        {
            // Non-blocking: an evicted section starts a background reload and is
            // re-requested by the radial planner once it lands, rather than stalling
            // this frame on a decompress.
            if (!world.TryGetForRender(best, out LodSection section))
            {
                if (world.LoadsInFlight.Contains(best))
                {
                    loadBudget--; // a reload is under way; radial demand retains responsibility
                }
                else
                {
                    RemoveMeshes(best);
                }
                continue;
            }

            if (section.CapturedColumns == 0)
            {
                RemoveMeshes(best);
                continue;
            }

            // Eight, not four. The mesher's side-face culling only ever asks about the four
            // edges, but cave culling floods light through a window around the section, and
            // an absent DIAGONAL leaves a section-sized block of assumed-open air touching
            // the corner. Light pours in from it and rescues caves that are genuinely dark -
            // measured at roughly half the saving. The corners are read for light only and
            // are never asked whether a side is covered.
            var neighborSections = new LodSection?[8];
            long estimatedBytes = SectionSnapshot.EstimateRetainedBytes(
                section, includeTextureSlots, includeWallColors: wallColors != null);

            // A neighbour that is absent from RAM is NOT automatically the edge of
            // explored space. HasDataSet is the question the mesher actually wants
            // answered - the shader's openEdges has always asked it - and residency was
            // standing in for it. Sections are meshed nearest-first, so the outward side
            // of nearly every one of them is scheduled before its neighbour has finished
            // loading, and an ocean answered that with a seabed-deep sheet of translucent
            // water down every section boundary that never healed.
            //
            // Deliberately no load request here: a neighbour outside the draw set is
            // meant to end in nothing, and forcing it resident to prove that would pull
            // the whole cache into memory one ring at a time.
            byte assumedCovered = 0;
            for (int d = 0; d < 4; d++)
            {
                long nk = LodWorld.NeighborKey(best, d == 0 ? -1 : d == 1 ? 1 : 0, d == 2 ? -1 : d == 3 ? 1 : 0);
                if (world.Sections.TryGetValue(nk, out LodSection? nb))
                {
                    neighborSections[d] = nb;
                    estimatedBytes = SectionSnapshot.SaturatingAdd(
                        estimatedBytes, SectionSnapshot.EstimateRetainedBytes(nb, includeTextureSlots));
                }
                else if (world.HasDataSet.Contains(nk) && !world.LoadFailed.Contains(nk))
                {
                    assumedCovered |= (byte)(1 << d);
                }
            }

            // The four corners, for light only.
            for (int d = 4; d < 8; d++)
            {
                int cornerX = (d == 4 || d == 6) ? -1 : 1;
                int cornerZ = d < 6 ? -1 : 1;
                long ck = LodWorld.NeighborKey(best, cornerX, cornerZ);
                if (!world.Sections.TryGetValue(ck, out LodSection? corner)) continue;
                neighborSections[d] = corner;
                estimatedBytes = SectionSnapshot.SaturatingAdd(
                    estimatedBytes, SectionSnapshot.EstimateRetainedBytes(corner, includeTextureSlots));
            }

            // TryTake transferred the exact dirty obligation to us. If this frame has
            // spent its snapshot time/byte allowance, restore that obligation before
            // stopping. The first eligible snapshot always progresses even if oversized.
            if (!snapshotBudget.TryStart(estimatedBytes))
            {
                world.RenderDirty.Add(best);
                break;
            }

            var neighbors = new SectionSnapshot?[neighborSections.Length];
            long contentRevision = world.ContentRevision(best);
            long renderRevision = world.RenderDirty.Revision(best);
            for (int d = 0; d < neighborSections.Length; d++)
            {
                if (neighborSections[d] != null)
                    neighbors[d] = SectionSnapshot.Of(neighborSections[d]!, includeTextureSlots);
            }

            SectionSnapshot self = SectionSnapshot.Of(section, includeTextureSlots, wallColors);

            // Same-world publication needs a finer identity than WorldEpoch. If a mesh-visible
            // capture or neighbour-edge invalidation landed while these immutable inputs were
            // copied, restore the obligation without submitting an already-stale snapshot.
            // ContentRevision is intentionally not the gate: provenance-only snow updates must
            // invalidate mip work without forcing an otherwise identical terrain mesh.
            if (renderRevision != world.RenderDirty.Revision(best))
            {
                world.RenderDirty.Add(best);
                continue;
            }

            meshBudget--;
            meshJobInFlight.Add(best);
            worker.EnqueueMesh(new MeshJob
            {
                WorldEpoch = currentWorldEpoch(),
                Key = best,
                ContentRevision = contentRevision,
                RenderRevision = renderRevision,
                Self = self,
                Neighbors = neighbors,
                EstimatedRetainedBytes = estimatedBytes,
                ReadyAtMilliseconds = Environment.TickCount64,
                AssumedCoveredSides = assumedCovered,
                CaveCullReach = CaveCulling ? caveCullReach : 0,
                HalfBlockLevels = halfBlockLevels,
                SurfaceHalfBlocksOnly = SurfaceHalfBlocksOnly,
                CutoutBlocksCover = CutoutBlocksCover,
                WaterAbsorption = WaterAbsorptionTable(),
                WaterSeamStripBlocks = waterSeamStripBlocks,
                SuppressUnknownWaterWalls = suppressUnknownWaterWalls,
                EmitTextureSlots = includeTextureSlots,
                TextureModeRevision = textureModeRevision,
            });
        }


        MeshSnapshotItems += snapshotBudget.Items;
        MeshSnapshotBytes = SectionSnapshot.SaturatingAdd(
            MeshSnapshotBytes, snapshotBudget.Bytes);
    }

    /// <summary>
    /// A section arrived in RAM. Any neighbour whose live mesh left the side facing it
    /// open - because this section's data was on disk rather than in memory when that
    /// mesh was built - now owes one re-mesh, and only that neighbour: the bit is cleared
    /// as the obligation is handed over so an arrival can never queue the same repair
    /// twice, and a section that guessed at nothing costs a dictionary miss.
    /// </summary>
    void OnSectionBecameResident(long key)
    {
        if (meshedWithoutNeighbor.Count == 0) return;

        for (int d = 0; d < 4; d++)
        {
            long nk = LodWorld.NeighborKey(key, d == 0 ? -1 : d == 1 ? 1 : 0, d == 2 ? -1 : d == 3 ? 1 : 0);
            if (!meshedWithoutNeighbor.TryGetValue(nk, out byte sides)) continue;

            // Our neighbour to the west has us to ITS east: W/E and N/S are the low bit
            // of the direction, so the facing side is d ^ 1.
            int facing = d ^ 1;
            if ((sides & (1 << facing)) == 0) continue;

            sides &= (byte)~(1 << facing);
            if (sides == 0) meshedWithoutNeighbor.Remove(nk);
            else meshedWithoutNeighbor[nk] = sides;

            world.RenderDirty.Add(nk);
            SeamRepairsQueued++;
        }
    }

    void UploadFinishedMeshes()
    {
        int itemBudget = MeshUploadsPerFrame;
        var uploadBudget = new LodDrainBudget(
            MeshUploadMaxBytesPerFrame, MeshUploadMaxMillisecondsPerFrame);
        while (itemBudget-- > 0
            && worker.MeshResults.TryPeek(out MeshResult? waiting)
            && uploadBudget.TryStart(waiting.EstimatedUploadBytes)
            && worker.MeshResults.TryDequeue(out MeshResult? result))
        {
            long currentEpoch = currentWorldEpoch();
            if (result.WorldEpoch != currentEpoch)
            {
                result.ReturnPooledArrays();
                continue;
            }

            meshJobInFlight.Remove(result.Key);

            // Keep the currently live mesh until a result built from the newest complete
            // render input arrives. Snow recapture made this race common: an older result
            // could otherwise replace the mesh for a frame or more, followed by the queued
            // correction, which appeared as a flash exactly when vanilla handed ownership
            // to the cache.
            if (result.RenderRevision != world.RenderDirty.Revision(result.Key))
            {
                if (!world.RenderDirty.Contains(result.Key))
                    world.RenderDirty.Add(result.Key);
                StaleMeshResultsRejected++;
                result.ReturnPooledArrays();
                continue;
            }

            // The blend master switch changes what word two of an L0 packed quad means.
            // Never publish work built for the other interpretation after the switch: put
            // the exact section obligation back and let the ordinary bounded scheduler
            // snapshot it again under the current identity.
            if (result.TextureModeRevision != textureModeRevision)
            {
                world.RenderDirty.Add(result.Key);
                result.ReturnPooledArrays();
                continue;
            }

            MeshRef? newOpaque = null;
            MeshRef? newWater = null;
            try
            {
                if (result.IndexCount > 0)
                {
                    newOpaque = Upload(result.Xyz, result.Rgba, result.Indices,
                        result.VertexCount, result.IndexCount);
                }

                if (result.WaterIndexCount > 0 && result.WaterXyz != null)
                {
                    newWater = Upload(result.WaterXyz, result.WaterRgba!, result.WaterIndices!,
                        result.WaterVertexCount, result.WaterIndexCount);
                }
            }
            catch
            {
                // A partial replacement is not a replacement. Keep the old opaque/water
                // pair live, free anything newly created, and restore the dirty obligation.
                DisposeMeshRef(newOpaque);
                DisposeMeshRef(newWater);
                world.RenderDirty.Add(result.Key);
                result.ReturnPooledArrays();
                throw;
            }

            try
            {
                renderPaths.Publish(
                    currentWorldEpoch(),
                    result.Key,
                    newOpaque,
                    newWater,
                    result.VertexCount,
                    result.IndexCount,
                    result.WaterVertexCount,
                    result.WaterIndexCount,
                    result.AssumedCoveredSides,
                    // Borrowed for the call only. The shadow arenas copy what they need before
                    // this returns; nothing may retain the mesher's arrays.
                    new LodRenderGeometry(
                        result.Xyz, result.Rgba, result.Indices,
                        result.PackedOpaqueQuads, result.PackedOpaqueQuadCount,
                        result.ClusteredPackedOpaqueQuads, result.PackedOpaqueClusters),
                    result.Heights,
                    result.HasTextureSlots,
                    result.OpaqueGroundCoverage);
            }
            finally
            {
                // Both uploads and the arena mirror have copied what they need by now, which
                // is what lets the mesh threads rent these (0.4.131).
                result.ReturnPooledArrays();
            }
        }

        MeshUploadItems += uploadBudget.Items;
    }

    void RemoveMeshes(long key)
    {
        renderPaths.Remove(currentWorldEpoch(), key);
    }

    void RemoveMeshesLegacy(long worldEpoch, long key)
    {
        meshedWithoutNeighbor.Remove(key);
        RemoveTemporalOcclusionQuery(key);
        if (!HasAnyMesh(key)) return;
        DisposeMeshRefs(key);
        meshBounds.Remove(key);
    }

    void DisposeMeshRefs(long key)
    {
        RemoveLiveMeshStats(key);
        if (sectionMeshes.Remove(key, out MeshRef? mesh)) DisposeMeshRef(mesh);
        if (waterMeshes.Remove(key, out MeshRef? water)) DisposeMeshRef(water);
    }

    void PublishLegacy(LodRenderPublication publication)
    {
        long key = publication.Identity.SectionKey;
        bool hadMesh = HasAnyMesh(key);
        PublishLiveMeshStats(publication);
        ReplaceMeshRefs(key, publication.Opaque, publication.Water);

        // Tracks the mesh that is actually live, so a repair can only ever be owed by
        // geometry that is on screen.
        if (publication.AssumedCoveredSides != 0)
            meshedWithoutNeighbor[key] = publication.AssumedCoveredSides;
        else
            meshedWithoutNeighbor.Remove(key);

        bool hasMesh = HasAnyMesh(key);
        if (!hadMesh && hasMesh) meshBounds.Include(key);
        else if (hadMesh && !hasMesh) meshBounds.Remove(key);

        // Fresh uploads get an age grace period. Later retention depends on distance,
        // never on whether the current camera happens to see the mesh.
        if (hasMesh)
        {
            lastResidencyMs[key] = Environment.TickCount64;
            QueueMeshEvictionCheck(key);
        }
    }

    void PublishLiveMeshStats(LodRenderPublication publication)
    {
        long key = publication.Identity.SectionKey;
        RemoveLiveMeshStats(key);
        var stats = new LodLiveMeshStats(
            Math.Max(0, publication.OpaqueVertices),
            Math.Max(0, publication.OpaqueIndices),
            Math.Max(0, publication.WaterVertices),
            Math.Max(0, publication.WaterIndices),
            publication.Heights,
            publication.HasTextureSlots,
            publication.OpaqueGroundCoverage);
        if (stats.OpaqueIndices == 0 && stats.WaterIndices == 0) return;

        liveMeshStats[key] = stats;
        // Both passes in one span: the traversal box precedes the split into opaque and
        // water draws, so it has to hold whichever of them the walk goes on to select.
        subtreeHeights.SetMesh(key, stats.Heights.Either);
        NoteSectionHeights(stats.Heights);
        LiveOpaqueVertices += stats.OpaqueVertices;
        LiveOpaqueIndices += stats.OpaqueIndices;
        LiveWaterVertices += stats.WaterVertices;
        LiveWaterIndices += stats.WaterIndices;
        LiveGpuMeshBytes += stats.Bytes;
        if (stats.HasTextureSlots) liveTexturedMeshes++;

        MaybeReleaseBlockTextureTable();
    }

    void RemoveLiveMeshStats(long key)
    {
        subtreeHeights.RemoveMesh(key);
        if (!liveMeshStats.Remove(key, out LodLiveMeshStats stats)) return;
        LiveOpaqueVertices -= stats.OpaqueVertices;
        LiveOpaqueIndices -= stats.OpaqueIndices;
        LiveWaterVertices -= stats.WaterVertices;
        LiveWaterIndices -= stats.WaterIndices;
        LiveGpuMeshBytes -= stats.Bytes;
        if (stats.HasTextureSlots) liveTexturedMeshes--;

        MaybeReleaseBlockTextureTable();
    }

    void ReplaceMeshRefs(long key, MeshRef? newOpaque, MeshRef? newWater)
    {
        sectionMeshes.TryGetValue(key, out MeshRef? oldOpaque);
        waterMeshes.TryGetValue(key, out MeshRef? oldWater);

        if (newOpaque != null) sectionMeshes[key] = newOpaque;
        else sectionMeshes.Remove(key);
        if (newWater != null) waterMeshes[key] = newWater;
        else waterMeshes.Remove(key);

        // Publish the complete replacement first. If an engine-side disposal ever
        // throws, the new mesh pair is still the live pair and will not leak or vanish.
        DisposeMeshRef(oldOpaque);
        DisposeMeshRef(oldWater);
        // The replaced section itself must draw until remeasured. Do not globally reset
        // everything behind it: active exploration can publish several meshes per frame,
        // which previously kept every query stale forever. Hidden terrain already performs
        // periodic exact probes, so a changed occluder converges without defeating culling.
        if (temporalOcclusionQueries.TryGetValue(key, out TemporalOcclusionQuery? query))
            query.State.Invalidate(temporalOcclusionEpoch);
    }

    void DisposeMeshRef(MeshRef? mesh)
    {
        if (mesh == null) return;
        LodPhaseStart phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        try
        {
            mesh.Dispose();
        }
        finally
        {
            MeshDisposeCost.Add(phaseStart);
        }
    }

    MeshRef Upload(float[] xyz, byte[] rgba, int[] indices, int vertCount, int indexCount)
    {
        // What UploadMesh is asked to transfer, not backing-array capacity. Mesher pools
        // can deliberately hand back arrays larger than the live vertex/index counts.
        MeshUploadBytes = SectionSnapshot.SaturatingAdd(
            MeshUploadBytes, vertCount * 16L + indexCount * sizeof(int));
        var mesh = new MeshData(false);
        mesh.SetVerticesCount(vertCount);
        mesh.SetIndicesCount(indexCount);
        mesh.xyz = xyz;
        mesh.Rgba = rgba;
        mesh.Indices = indices;
        LodPhaseStart phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        try
        {
            return capi.Render.UploadMesh(mesh);
        }
        finally
        {
            GlUploadCost.Add(phaseStart);
        }
    }

    // ---- Frame ----

    float ApprovedViewDistance()
    {
        var playerData = capi.World.Player.WorldData;
        float distance = playerData.DesiredViewDistance;
        if (playerData.LastApprovedViewDistance > 0)
            distance = Math.Min(distance, playerData.LastApprovedViewDistance);
        return Math.Max(0, distance);
    }

    /// <summary>
    /// Reads the engine's own range-cull inputs for this frame, or returns an unavailable
    /// state that keeps every caller on the conservative bound.
    ///
    /// <para>
    /// `IRenderAPI.DefaultFrustumCuller` is public API and hands back `ClientMain.frustumCuller`
    /// itself - the same instance `MeshDataPool.FrustumCull` consults microseconds later in the
    /// same opaque stage. `ClientMain` fills it in after the whole Before stage, immediately
    /// before triggering the opaque one. The culler's settings read here hold all frame, but the
    /// player's position does not: the engine moves the player INSIDE the Before stage
    /// (`EntityBehaviorPlayerPhysics`, order 1.0), so a capture taken by the preparation at
    /// 0.994 describes where the player stood before this frame's physics step. That one-frame
    /// lag was the handoff flash; `RunLateRangeCheck` captures again after the step.
    /// </para>
    /// <para>
    /// Two inputs are deliberately not the ones this renderer uses elsewhere:
    /// </para>
    /// <para>
    /// The anchor is the PLAYER's block position, because that is what `CalcFrustumEquations`
    /// is handed. Everything else here measures from `Entity.CameraPos`, which in third person
    /// trails the player by several blocks - and the engine does not care where the camera is
    /// when it decides how far it will draw.
    /// </para>
    /// <para>
    /// The distance is the culler's own `ViewDistanceSq`, which `UpdateViewDistance` fills from
    /// `ClientSettings.ViewDistance` - the CLIENT setting, not the server-approved value
    /// `ApprovedViewDistance` returns. On a server that caps view distance the two differ, and
    /// only the client's decides what the engine draws.
    /// </para>
    /// </summary>
    VanillaDrawRangeState CaptureDrawRange()
    {
        try
        {
            FrustumCulling? culler = capi.Render.DefaultFrustumCuller;
            if (culler == null || culler.ViewDistanceSq <= 0) return default;
            EntityPos pos = capi.World.Player.Entity.Pos;
            // `AsBlockPos` truncates rather than floors. Reproduced rather than corrected:
            // world coordinates here are non-negative, and matching the engine matters more
            // than being right about a case that cannot arise.
            return new VanillaDrawRangeState((int)pos.X, (int)pos.Z,
                culler.ViewDistanceSq, culler.lod0BiasSq, culler.lod2BiasSq);
        }
        catch
        {
            return default;
        }
    }

    /// <summary>
    /// Withdraws ownership from every committed cell the engine's range clause has stopped
    /// admitting, in the frame it stops admitting it.
    ///
    /// <para>
    /// This is the half of the exact rule that lets the overlap margin go to zero. Loss used to
    /// wait for a probe, and probes come round about once a second, so cached terrain had to
    /// give ground back early - 16 blocks of guessing at how far the player might travel before
    /// anyone checked. Nothing here needs a probe: the sphere was recorded when the cell was
    /// committed and the player's block position is already in hand, so the same decision can
    /// be re-made for the whole committed set immediately.
    /// </para>
    /// <para>
    /// It runs only when one of the engine's own quantised inputs has changed. That is not a
    /// budget or a cadence - it is the observation that none of those inputs varies
    /// continuously: `AsBlockPos` truncates to whole blocks, and the biases come from settings.
    /// A settled view sweeps zero times; walking sweeps once per block.
    /// </para>
    /// <para>
    /// Loss only. A cell whose range starts admitting it again is left for the ordinary probe
    /// path, which is the plan's fast-loss/cautious-gain rule: gaining early costs an overlap,
    /// losing late costs a hole.
    /// </para>
    /// </summary>
    void SweepRangeLosses(long frameCounter)
    {
        if (readiness == null || !ExactDrawRange || !ChunkMaskEnabled) return;
        if (!drawRange.Available) return;
        if (!rangeSweepNow && !drawRange.DiffersFrom(lastSweptDrawRange)) return;

        rangeSweepNow = false;
        lastSweptDrawRange = drawRange;
        ReadinessRangeSweeps++;

        long started = Stopwatch.GetTimestamp();
        readiness.CollectRangeLosses(drawRange, rangeLosses, OverlapMarginBlocks);
        foreach (VanillaChunkCell cell in rangeLosses)
        {
            // Straight through the ordinary publication handshake, so a sweep-driven loss is
            // indistinguishable downstream from a probe-driven one and cannot get ahead of the
            // texture the shader reads.
            if (!readiness.Observe(cell, rendered: false, frameCounter,
                out VanillaReadinessPublication publication)) continue;
            if (!readiness.ResolvePublication(publication, accepted: true)) continue;

            readiness.ClearDrawSphere(cell);
            readinessMask?.Set(publication.Cell, false);
            readinessHandoffStale = true;
            ReadinessLostTransitions++;
            ReadinessRangeSweepLosses++;
        }

        double elapsedMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
        if (elapsedMs > ReadinessRangeSweepPeakMs) ReadinessRangeSweepPeakMs = elapsedMs;
    }

    void UpdateReadinessShadow(float viewDistance)
    {
        if (readinessDisabled) return;

        // Read once per frame, before anything consults it, so every ownership decision and
        // the sweep all agree about where the player is and how far the engine draws. Reading
        // it per decision would let the player cross a block boundary halfway through a probe
        // batch and decide two halves of one frontier against two different anchors.
        drawRange = CaptureDrawRange();

        if (capi.World.Player.Entity.Pos.Dimension != 0)
        {
            readiness?.Clear();
            readinessSeedCursor = readinessBoundaryCursor = readinessInteriorCursor = 0;
            readinessOwnsHandoff = false;
            // Another dimension's spheres decide nothing here, and the sweep must not compare
            // this world's player position against them.
            lastSweptDrawRange = default;
            rangeSweepNow = true;
            // Ownership belongs to the world it was measured in. Leaving the default
            // dimension must return every cell to the cache rather than keep suppressing
            // terrain with another world's readiness.
            readinessMask?.Clear();
            readinessMaskActive = false;
            // The band is derived from ownership, so another dimension's ownership must not
            // keep correcting terrain here. Clearing costs the correction, never a pixel.
            seamField?.Clear();
            seamFieldActive = false;
            return;
        }

        try
        {
            var blockAccessor = capi.World.BlockAccessor;
            int verticalChunks = Math.Max(1, (blockAccessor.MapSizeY + 31) / 32);
            VanillaReadinessWindow window = VanillaRenderReadiness.CalculateWindow(
                camPos.X, camPos.Z, viewDistance,
                blockAccessor.MapSizeX, blockAccessor.MapSizeZ, ReadinessGuardChunks);

            if (readiness == null || readiness.VerticalChunks != verticalChunks
                || readiness.HorizontalCapacity < window.RequiredCapacity)
            {
                readiness = new VanillaRenderReadiness(window.RequiredCapacity, verticalChunks);
                readiness.ColumnEvicted = (chunkX, chunkZ) => readinessMask?.ClearColumn(chunkX, chunkZ);
                readinessSeedCursor = readinessBoundaryCursor = readinessInteriorCursor = 0;
                readinessHandoffStale = true;
                rangeSweepNow = true;
                ReadinessResizes++;
                if (chunkMaskRequested && !readinessMaskFailed)
                {
                    readinessMask = new VanillaReadinessMask(window.RequiredCapacity, verticalChunks);
                    DisposeReadinessMaskTexture();
                    // The field mirrors the mask's window, so a new mask means the stored
                    // distances describe a window that no longer exists.
                    seamField = null;
                    DisposeSeamFieldTexture();
                }
            }

            if (chunkMaskRequested && !readinessMaskFailed && readinessMask == null)
            {
                readinessMask = new VanillaReadinessMask(
                    readiness.HorizontalCapacity, readiness.VerticalChunks);
                readiness.WriteMask(readinessMask);
            }

            if (readiness.SetWindow(window.MinX, window.MinZ, window.Width, window.Depth,
                frameCounter))
            {
                // Deliberately not resetting the discovery cursor. Restarting it on every
                // chunk crossing meant that during flight it swept the same near cells over
                // and over and never reached the frontier, which SetWindow now queues
                // directly. The boundary and interior cursors are cheap ring scans and
                // may restart.
                // The boundary cursor may restart because the whole shell is swept below on
                // this same frame. The interior cursor may not: it is the safety net that
                // eventually revisits every cell, one slice per frame, and restarting it on
                // every chunk crossing meant that while flying it never advanced past its
                // first slice - so stale ownership behind the camera was never revisited by
                // the one sweep guaranteed to reach it.
                readinessBoundaryCursor = 0;
                readinessHandoffStale = true;
                readinessSweepShellNow = true;
                rangeSweepNow = true;
                ReadinessWindowChanges++;
                // Departing columns were cleared inside the tracker; rebuilding is what
                // keeps a reused ring slot from carrying another place's ownership.
                if (readinessMask != null) readiness.WriteMask(readinessMask);
            }

            readiness.SeedUnknownCells(
                ref readinessSeedCursor, ReadinessSeedCellsPerFrame, frameCounter);
            readiness.QueueReadyShell(window.CenterX, window.CenterZ,
                Math.Max(0, window.VanillaRadius - ReadinessGuardChunks), window.OuterRadius,
                ref readinessBoundaryCursor,
                readinessSweepShellNow
                    ? ReadinessBoundarySweepOnCrossing
                    : ReadinessBoundaryCellsPerFrame,
                frameCounter);
            readinessSweepShellNow = false;
            readiness.QueueMaintenanceCells(
                ref readinessInteriorCursor, ReadinessInteriorCellsPerFrame, frameCounter);

            long now = capi.ElapsedMilliseconds;
            if (now - readinessLastFullRevalidateMs >= ReadinessFullRevalidateMilliseconds)
            {
                readinessLastFullRevalidateMs = now;
                ReadinessFullRevalidations++;
                readiness.QueueAllReadyCells(frameCounter);
                ResyncReadinessMask();

                // The counts the whole-mesh skip trusts are derived state that nothing else
                // re-derives. A single drift high hides a section permanently, so they are
                // checked against the cell states each pass and any repair is reported: a
                // non-zero number here is a bug in this class, not a tuning problem.
                int repaired = readiness.AuditReadyCounts();
                if (repaired > 0)
                {
                    ReadinessAggregateRepairs += repaired;
                    if (!readinessAggregateReported)
                    {
                        readinessAggregateReported = true;
                        capi.Logger.Warning(
                            "[Landscapes] ownership audit repaired {0} section counts; a section "
                            + "was reporting itself fully covered by vanilla terrain when it was not. "
                            + "Please report this with the surrounding log.", repaired);
                    }
                }
            }
            // Freshly announced chunks go to the front of this frame's probes, ahead of the
            // sweeps, and ahead of the observations promoted just below so a watched cell's
            // second look is not queued behind them. Off, the list is only kept tidy.
            readiness.PromoteWatched(frameCounter, Environment.TickCount64,
                FreshChunkWatchMilliseconds, FreshChunkWatchesPerFrame, promote: StandInFix);
            readiness.PromoteObserved(frameCounter, ReadinessProbeCatchUpItemsPerFrame);

            bool catchingUp = readiness.PendingCandidates >= ReadinessProbeCatchUpQueueDepth;
            int maxProbes = catchingUp
                ? ReadinessProbeCatchUpItemsPerFrame
                : ReadinessProbeMaxItemsPerFrame;
            double maxMilliseconds = catchingUp
                ? ReadinessProbeCatchUpMillisecondsPerFrame
                : ReadinessProbeMaxMillisecondsPerFrame;

            long started = Stopwatch.GetTimestamp();
            long maxTicks = Math.Max(1,
                (long)Math.Ceiling(maxMilliseconds * Stopwatch.Frequency / 1000.0));
            int probed = 0;
            while (probed < maxProbes)
            {
                if (probed > 0 && Stopwatch.GetTimestamp() - started >= maxTicks) break;
                if (!readiness.TryDequeueCandidate(out VanillaChunkCell cell)) break;

                bool rendered;
                try
                {
                    readinessProbePos.Dimension = 0;
                    readinessProbePos.SetPos(
                        (cell.X + 0.5) * 32.0,
                        (cell.Y + 0.5) * 32.0,
                        (cell.Z + 0.5) * 32.0);
                    rendered = capi.IsChunkRendered(readinessProbePos);
                }
                catch
                {
                    // An unknown/error result must restore or retain cached ownership.
                    rendered = false;
                    ReadinessProbeErrors++;
                }

                probed++;
                ReadinessProbes++;
                if (rendered) ReadinessTrueResults++;
                else ReadinessFalseResults++;

                if (!rendered && readiness.State(cell) == VanillaReadinessState.VanillaReady)
                    ReadinessStaleCommittedFound++;

                // Measured, not acted on; see G40 for what happened the one time a rule
                // like this reached the draw path from reasoning instead of evidence.
                if (readinessNoGeometryByY.Length < readiness.VerticalChunks)
                    readinessNoGeometryByY = new int[readiness.VerticalChunks];

                // Both the measurement and the rule below need a chunk lookup, which takes
                // the client's chunk lock - the same lock IsChunkRendered just took, and the
                // same one the loader threads want while a world is coming up. Ask only when
                // this observation can actually change ownership: the second of the two true
                // observations a commit needs, or a re-confirmation of a committed cell.
                // Every other probe at join is a first sighting that decides nothing.
                VanillaReadinessState decisionState = readiness.State(cell);
                bool atOwnershipDecision = decisionState is VanillaReadinessState.ObservedRendered
                    or VanillaReadinessState.VanillaReady;

                // One lookup, three consumers. Each of the rules below used to fetch the
                // chunk for itself, so an ownership decision could take the client's chunk
                // lock three times over - the lock G45 is about, at the moment the probe
                // queue is longest. Fetching once here is what pays for the range rule
                // needing the chunk at all: it is a net reduction in lock traffic, not an
                // addition.
                IWorldChunk? decisionChunk = null;
                if (rendered && atOwnershipDecision)
                {
                    try { decisionChunk = capi.World.BlockAccessor.GetChunk(cell.X, cell.Y, cell.Z); }
                    catch { decisionChunk = null; }
                    MeasureDrawnChunk(cell, decisionChunk, camPos.X, camPos.Z);
                }

                // Beyond the view distance the engine draws nothing here, whatever every
                // other signal says. It range-culls each terrain pool location per frame
                // against the view distance, and the signals ownership is built on do not
                // follow: the drawn counter never resets, the mesh stays in the pool, Hide
                // stays clear, and the culler freezes its verdict entirely while the camera
                // holds still in one chunk. So a cell the player walked away from stays
                // committed forever, and the mask discards cached terrain there against
                // nothing - a band at the seam, on the trailing side only, that never heals.
                //
                // Two rules, not one, and which applies depends on whether the engine's own
                // measurement is readable for this chunk:
                //
                //   exact - the chunk holds drawable meshes, so its bounding sphere and its
                //           pool levels can be read and the engine's clause evaluated as
                //           written. No slack, because nothing is being guessed.
                //
                //   bound - air, an unmeshed chunk, or an unavailable binding. Falls back to
                //           the column, which is all there is without a sphere. Note this is
                //           NOT what 0.3.152 did: it ran the bounded rule on every probe
                //           including first sightings, and that is what deadlocked ownership
                //           at small view distances.
                //
                // Air is denied by the bound too, unlike the geometry rule below. Beyond this
                // range vanilla owns nothing at all, so whole columns must demote together;
                // that is also what releases the section aggregates the whole-mesh skip reads.
                //
                // Gated on the mask for the same reason as the geometry rule: the default
                // radial-handoff path is benchmarked as it stands, and demoting the annulus
                // would pull its single global radius in for every column.
                // Only where an observation can actually change ownership, and that gate is the
                // whole point rather than an optimisation.
                //
                // A first sighting moves a cell from Unknown to ObservedRendered and commits
                // nothing: no aggregate moves, no texel is written, nothing is suppressed. The
                // range rule exists to stop OWNERSHIP, so applying it here refuses nothing
                // useful - and it used to refuse the cell's whole future.
                //
                // 0.3.154 through 0.3.157 ran the bounded fallback on first sightings, because
                // the exact rule needs the chunk and a first sighting had not fetched one. The
                // bounded threshold is `viewDistance - 46`, and at a 64-block vanilla view
                // distance that is 18 blocks - so a chunk 28 blocks away, meshed, cull-visible
                // and well inside the 67 blocks the engine actually draws to, was denied, stayed
                // Pending, never became an ownership decision, and therefore never got the exact
                // rule that would have admitted it. A deadlock: the accurate rule was gated on
                // already being on the path that the inaccurate rule prevented reaching. At a
                // 32-block view distance the gate is zero and ownership is impossible outright.
                // `.dlexact audit` measured it: 7 of 15 drawn chunks uncovered, every one of
                // them Pending, the nearest at 28 blocks.
                // A first sighting still gets ONE denial, and only one that cannot deadlock:
                // the column's nearest face against the widest bound any LOD level draws to.
                // The centre is inside the column so it is at least this far away, which makes
                // everything refused here a strict subset of what the exact rule refuses - so
                // no cell is kept out of the ownership path that could have stayed on it.
                //
                // Without this, every loaded chunk outside the draw range costs two probes and
                // an observation slot per cycle instead of one, because it has to reach
                // ObservedRendered before anything is allowed to turn it away. That is most of
                // the window at a small view distance.
                if (rendered && ChunkMaskEnabled && !atOwnershipDecision
                    && ExactDrawRange && drawRange.Available
                    && drawRange.ColumnCertainlyNotDrawn(cell.X, cell.Z, OverlapMarginBlocks))
                {
                    rendered = false;
                    ReadinessOwnershipDeniedBeyondViewDistance++;
                    // Out of the engine's range, it will not be drawn however often it is
                    // asked. The range sweep and ordinary probes pick it up if the player nears.
                    EndFreshChunkWatch(cell);
                }

                if (rendered && ChunkMaskEnabled && atOwnershipDecision)
                {
                    float centreX = 0f, centreZ = 0f;
                    int lodMask = 0;
                    bool exact = ExactDrawRange && drawRange.Available
                        && VanillaChunkGeometry.TryReadDrawBounds(
                            decisionChunk, out centreX, out centreZ, out lodMask);

                    bool denied;
                    if (exact)
                    {
                        denied = !drawRange.DrawsWithMargin(
                            centreX, centreZ, lodMask, OverlapMarginBlocks);
                        // Recorded whether or not this cell is granted, because the sweep has
                        // to be able to re-decide it later without the chunk. Cleared when it
                        // is denied, so a demoted cell cannot be re-tested from a sphere that
                        // no longer decides anything.
                        if (denied) readiness.ClearDrawSphere(cell);
                        else readiness.SetDrawSphere(cell, centreX, centreZ, lodMask);
                        ReadinessExactRangeDecisions++;
                    }
                    else
                    {
                        // Air, an unmeshed chunk, or an unavailable binding: there is no sphere
                        // to read, so the column is all there is to go on.
                        //
                        // With the exact rule active that means the column's nearest face
                        // against the widest bound - the same sound test the first-sighting gate
                        // uses. `viewDistance - 46` is NOT used here any more: it is 18 blocks at
                        // a 64-block view distance, so it refused every air cell past half a
                        // chunk, and air cells are most of a column. That under-reported ownership
                        // in `ready per Y` badly enough to send this investigation after the wrong
                        // cause twice. Air can only ever cost accuracy here, never a pixel: an air
                        // cell is excluded from the mask (G40), so it suppresses nothing whatever
                        // it owns, and its only consumers are the column aggregate and telemetry.
                        denied = ExactDrawRange && drawRange.Available
                            ? drawRange.ColumnCertainlyNotDrawn(cell.X, cell.Z, OverlapMarginBlocks)
                            : VanillaRenderReadiness.BeyondVanillaDrawRange(
                                cell.X, cell.Z, camPos.X, camPos.Z, viewDistance, OverlapMarginBlocks);
                        readiness.ClearDrawSphere(cell);
                        ReadinessBoundedRangeDecisions++;
                    }

                    if (denied)
                    {
                        rendered = false;
                        ReadinessOwnershipDeniedBeyondViewDistance++;
                        EndFreshChunkWatch(cell);
                    }
                }

                // The engine's drawn counter advances before the completed mesh is uploaded.
                // Mesh presence is therefore useful while GAINING ownership: it prevents a
                // first handoff from running ahead of vanilla's actual replacement.
                //
                // It is NOT a valid loss signal. Ordinary retessellation and the culler's
                // current verdict are transient, and revoking an already-ready cell on one
                // such observation made the mask alternate between vanilla and cache. The
                // 0.4.48 owner log recorded 28 losses, zero false IsChunkRendered results on
                // committed cells and zero range losses: every loss came through this gate.
                // Actual chunk removal still returns false above and exact range loss is
                // swept every player-block crossing, so retaining committed ownership here
                // does not weaken either authoritative loss path.
                bool gainingOwnership = decisionState == VanillaReadinessState.ObservedRendered;
                if (rendered && gainingOwnership && ChunkMaskEnabled && GeometryOwnershipRule
                    && VanillaChunkGeometry.Available
                    && !VanillaChunkHoldsDrawableGround(decisionChunk))
                {
                    rendered = false;
                    ReadinessOwnershipDeniedNoGeometry++;
                    // A mesh still on its way is worth checking again next frame. One uploaded
                    // as nothing - rock with no exposed face - never will be, so it returns to
                    // the sweep, as does any chunk whose upload state cannot be read.
                    if (!VanillaChunkGeometry.TryMeshStillComing(decisionChunk, out bool coming) || !coming)
                        EndFreshChunkWatch(cell);
                }

                if (!readiness.Observe(cell, rendered, frameCounter,
                    out VanillaReadinessPublication publication)) continue;

                // Phase 1 has no texture. Accepting into this shadow model lets diagnostics
                // validate aggregate transitions without changing any pixel or draw call.
                if (!readiness.ResolvePublication(publication, accepted: true)) continue;
                // Recorded, not only applied. The mask is also rebuilt wholesale on every
                // window change, and a rebuild has no chunk to ask - so the exclusion has to
                // live in the tracker or every 32 blocks of travel puts air ownership back
                // into the atlas until the next resync scrubs it out again.
                bool air = VanillaChunkIsAir(publication.Cell);
                readiness.SetMaskExcluded(publication.Cell, air);
                readinessMask?.Set(publication.Cell, publication.Ready && !air);
                readinessHandoffStale = true;
                if (publication.Ready) ReadinessReadyTransitions++;
                else ReadinessLostTransitions++;
                if (publication.Ready) NoteFreshChunkConfirmed(publication.Cell);
            }

            // After the probes, so a cell committed this frame is swept with the sphere the
            // probe just recorded rather than a frame later.
            SweepRangeLosses(frameCounter);

            // Measures this frame's final ownership against what the engine is drawing, and
            // under `.dlgap release` withdraws ground nothing draws, before the handoff radius
            // and the mask are published below. Off unless `.dlgap` is on.
            ScanHandoffGaps(frameCounter);

            // Ownership the tracker can actually prove, converted into the radius the
            // fragment shader already understands. Uncertainty anywhere - an unowned
            // column, an untracked window edge, an unconverged join - pulls this in and
            // returns coverage to the cache rather than opening a hole.
            //
            // The measurement only changes when committed ownership changes or the camera
            // moves, so a settled view reuses it instead of rescanning every column every
            // frame. Both triggers are conservative: either one rescans.
            if (readinessHandoffStale
                || Math.Abs(camPos.X - readinessHandoffCameraX) >= ReadinessHandoffRecheckBlocks
                || Math.Abs(camPos.Z - readinessHandoffCameraZ) >= ReadinessHandoffRecheckBlocks)
            {
                double nearestIncomplete = readiness.NearestIncompleteColumnBlocks(
                    camPos.X, camPos.Z, out _);
                readinessOwnedRadius =
                    (float)Math.Max(0, nearestIncomplete - ReadinessHandoffMarginBlocks);
                readinessHandoffCameraX = camPos.X;
                readinessHandoffCameraZ = camPos.Z;
                readinessHandoffStale = false;
            }

            readinessHandoffDistance = nearHandoff.Update(
                Math.Min(readinessOwnedRadius, viewDistance), capi.ElapsedMilliseconds);
            readinessOwnsHandoff = true;
            PublishReadinessMask();
            PublishSeamField();
        }
        catch (Exception e)
        {
            readiness?.Clear();
            readiness = null;
            readinessDisabled = true;
            readinessOwnsHandoff = false;
            // Everything derived from the exact rule has to fall back with it. Leaving a
            // usable range state behind would keep the refused ring at zero - and water's
            // footprint with it - while nothing is measuring anything any more.
            drawRange = default;
            lastSweptDrawRange = default;
            DisableReadinessMask();
            if (!readinessFailureReported)
            {
                readinessFailureReported = true;
                capi.Logger.Warning(
                    "[Landscapes] vanilla-readiness shadow tracker disabled; radial handoff remains active: {0}",
                    e.Message);
            }
        }
    }

    /// <summary>
    /// Per-cell ownership on or off. Turning it off restores the measured handoff radius
    /// immediately and releases the texture; turning it on rebuilds the mask from state the
    /// tracker already holds, so neither direction waits for readiness to reconverge.
    /// A failed mask stays off: the failure disabled it for a reason.
    /// </summary>
    public bool ChunkMaskEnabled
    {
        get => chunkMaskRequested && !readinessMaskFailed;
        set
        {
            chunkMaskRequested = value;
            if (value) return;
            readinessMask = null;
            DisposeReadinessMaskTexture();
            // The band is derived from per-cell ownership. Without the mask there is no
            // seam to measure from, so it goes with it rather than holding stale distances.
            seamField = null;
            DisposeSeamFieldTexture();
        }
    }

    public bool ChunkMaskFailed => readinessMaskFailed;

    /// <summary>
    /// What the player asked for, regardless of whether this session's mask is working.
    /// Persistence must read this and never <see cref="ChunkMaskEnabled"/>: that getter
    /// reports the effective state, so saving it would write `false` after any transient
    /// texture failure and turn the feature off for every future session.
    /// </summary>
    public bool ChunkMaskRequested => chunkMaskRequested;

    /// <summary>
    /// Walks the line of sight and reports the first cell whose ownership would produce a
    /// hole, with the distance it was found at. Asking about one fixed distance was not
    /// usable: the hole a player is looking at is wherever it happens to be, and a report
    /// about some other chunk looks identical to a report about the right one.
    ///
    /// Sampling stops at the first cell the mask suppresses while the engine says it is not
    /// drawing that chunk, because that is the combination that leaves nothing on screen.
    /// If the ray finds no such cell, the first place with no cached section is reported
    /// instead, which is the other way a hole appears and is not the mask's doing.
    /// </summary>
    public string ExplainViewRay(double startX, double startY, double startZ,
        float lookX, float lookY, float lookZ, int maxBlocks)
    {
        // Captured once: the field is nullable and can be cleared by the failure path, and
        // a diagnostic must not be the thing that throws while explaining a problem.
        VanillaRenderReadiness? tracker = readiness;
        if (tracker == null) return "no readiness tracker; the distance handoff is drawing";

        var previous = new VanillaChunkCell(int.MinValue, int.MinValue, int.MinValue);
        string? firstEmpty = null;
        int examined = 0;
        int verticalChunks = Math.Max(1, (worldHeight + 31) / 32);
        float viewDistance = ApprovedViewDistance();

        for (int distance = 0; distance <= maxBlocks && examined < 4096; distance += 4)
        {
            double x = startX + lookX * distance;
            double y = startY + lookY * distance;
            double z = startZ + lookZ * distance;
            if (y < 0 || y >= worldHeight) break;

            var cell = new VanillaChunkCell(
                VanillaRenderReadiness.ChunkCoordinate(x),
                VanillaRenderReadiness.ChunkCoordinate(y),
                VanillaRenderReadiness.ChunkCoordinate(z));
            if (cell == previous) continue;
            previous = cell;
            examined++;
            if (cell.Y < 0 || cell.Y >= verticalChunks) continue;

            bool rendered;
            try
            {
                readinessProbePos.Dimension = 0;
                readinessProbePos.SetPos((cell.X + 0.5) * 32.0, (cell.Y + 0.5) * 32.0, (cell.Z + 0.5) * 32.0);
                rendered = capi.IsChunkRendered(readinessProbePos);
            }
            catch
            {
                rendered = false;
            }

            bool suppressed = readinessMaskActive && readinessMask != null && readinessMask.IsReady(cell);

            // "Rendered" is not "drawing". The engine's counter advances once for a chunk of
            // air and is never reset, so a chunk it claims can hold no terrain at all: freed
            // on overload, or never meshed because nothing in it has an exposed face. Without
            // this the search walks straight past the one case that leaves ground undrawn
            // while we suppress the cache over it, and answers "nothing wrong".
            IWorldChunk? vanillaChunk = null;
            try { vanillaChunk = capi.World.BlockAccessor.GetChunk(cell.X, cell.Y, cell.Z); }
            catch { /* a diagnostic never throws out of itself */ }
            // Any reason the engine will not submit this chunk counts, not just a missing
            // mesh: it can hold one and still be flagged not to draw. Frustum state is
            // deliberately excluded - a chunk behind the camera is not a hole.
            bool engineHoldsNothing = vanillaChunk is { Empty: false }
                && VanillaChunkGeometry.TryIsVanillaDrawing(vanillaChunk, out bool drawingHere)
                && !drawingHere;

            // The other way the engine declines to draw a chunk it still holds, and the one
            // that used to make this report say "engine is drawing this chunk" about ground
            // nothing was drawing: every terrain pool location is range-culled per frame
            // against the view distance, and none of the flags above follow.
            //
            // Answered by whichever rule ownership itself used, never by the conservative
            // bound alone. The two differ by up to 62 blocks, so quoting the bound while
            // ownership ran on the measurement would report a hole in every column inside
            // that ring - a search whose whole job is finding real ones (G72).
            bool beyondDrawRange = ExactDrawRange && drawRange.Available
                && VanillaChunkGeometry.TryReadDrawBounds(
                    vanillaChunk, out float holeCentreX, out float holeCentreZ, out int holeLods)
                ? !drawRange.DrawsWithMargin(holeCentreX, holeCentreZ, holeLods, OverlapMarginBlocks)
                : VanillaRenderReadiness.BeyondVanillaDrawRange(
                    cell.X, cell.Z, startX, startZ, viewDistance, OverlapMarginBlocks);

            // Ground can be drawn by any level, so report all of them. A single verdict has
            // twice now been read as evidence when it was describing a different level than
            // the one that would actually have drawn the ground in question.
            bool resident = false;
            bool meshed = false;
            bool skipped = false;
            levelReport.Clear();
            for (int level = 0; level <= LodWorld.MaxLevel; level++)
            {
                int footprint = LodWorld.KeyFootprintBlocks(LodWorld.SectionKey(level, 0, 0));
                long key = LodWorld.SectionKey(level,
                    (int)Math.Floor(x / footprint), (int)Math.Floor(z / footprint));
                bool hasSection = world.Sections.ContainsKey(key);
                bool hasMesh = HasAnyMesh(key);
                bool wasSkipped = skippedLastFrame.Contains(key);
                resident |= hasSection;
                meshed |= hasMesh;
                skipped |= wasSkipped;

                if (!hasSection && !hasMesh) continue;
                if (levelReport.Length > 0) levelReport.Append(", ");
                levelReport.Append('L').Append(level).Append(' ')
                    .Append(hasSection ? "resident" : "absent")
                    .Append(hasMesh ? " meshed" : " unmeshed");
                levelReport.Append(' ').Append(tracker.Classify(key));
                if (wasSkipped) levelReport.Append(" SKIPPED-AS-OWNED");
            }
            if (levelReport.Length == 0) levelReport.Append("no cached section at any level");

            if ((!rendered || engineHoldsNothing || beyondDrawRange) && (suppressed || skipped))
            {
                string engineSays = !rendered
                    ? "engine not drawing"
                    : beyondDrawRange
                        ? $"engine claims this chunk but it is beyond the {viewDistance:0}-block draw range"
                        : "engine claims this chunk but is not drawing it";
                return $"OURS at {distance} blocks, chunk {cell.X},{cell.Y},{cell.Z}: {engineSays}, "
                    + $"we say {tracker.State(cell)}, mask {(suppressed ? "suppresses" : "allows")}, "
                    + $"draw {(skipped ? "skipped as owned" : "not skipped")} | "
                    + $"{VanillaChunkGeometry.DescribeSignals(vanillaChunk)} | {levelReport}";
            }

            if (firstEmpty == null && !rendered && !meshed)
            {
                firstEmpty = $"NO CACHED TERRAIN at {distance} blocks, chunk {cell.X},{cell.Y},{cell.Z}: "
                    + $"engine not drawing, we say {tracker.State(cell)}, "
                    + $"mask {(suppressed ? "suppresses" : "allows")} | "
                    + $"{VanillaChunkGeometry.DescribeSignals(vanillaChunk)} | {levelReport}";
            }
        }

        if (firstEmpty != null) return firstEmpty;
        return $"nothing wrong along {maxBlocks} blocks of view: every chunk sampled is either "
            + "drawn by the engine or backed by cached terrain that is allowed to draw.";
    }

    /// <summary>
    /// What the mod believes about the ground at a world position, against what the client
    /// says right now. Written for standing in front of a hole and asking why it is there:
    /// if ownership reads committed while the engine reports the chunk unrendered, the mask
    /// is suppressing cached terrain nothing else draws, and that is ours. If ownership is
    /// already cache-owned, the hole is missing cached data or mesh, and the mask is not
    /// involved.
    /// </summary>
    public string DescribeOwnershipAt(double worldX, double worldY, double worldZ)
    {
        if (readiness == null) return "no readiness tracker";

        var cell = new VanillaChunkCell(
            VanillaRenderReadiness.ChunkCoordinate(worldX),
            VanillaRenderReadiness.ChunkCoordinate(worldY),
            VanillaRenderReadiness.ChunkCoordinate(worldZ));

        bool renderedNow;
        try
        {
            readinessProbePos.Dimension = 0;
            readinessProbePos.SetPos((cell.X + 0.5) * 32.0, (cell.Y + 0.5) * 32.0, (cell.Z + 0.5) * 32.0);
            renderedNow = capi.IsChunkRendered(readinessProbePos);
        }
        catch
        {
            renderedNow = false;
        }

        VanillaReadinessState state = readiness.State(cell);
        bool suppressed = readinessMaskActive && readinessMask != null && readinessMask.IsReady(cell);
        long sectionKey = LodWorld.SectionKey(0,
            (int)Math.Floor(worldX / LodSection.SectionBlocks),
            (int)Math.Floor(worldZ / LodSection.SectionBlocks));

        // What the engine actually holds here, which "rendered" does not answer: its drawn
        // counter advances for a chunk of air and is never reset, so it reports whether the
        // chunk was ever tesselated, not whether terrain exists in the buffers right now.
        IWorldChunk? vanillaChunk = null;
        try { vanillaChunk = capi.World.BlockAccessor.GetChunk(cell.X, cell.Y, cell.Z); }
        catch { /* a diagnostic never throws out of itself */ }

        bool geometryKnown = VanillaChunkGeometry.TryIsVanillaDrawing(vanillaChunk, out bool drawingNow);
        bool holdsGeometry = drawingNow;

        // Distance outranks every flag on the chunk. The engine range-culls each terrain
        // pool location per frame, so a chunk this far out is not drawn however loaded,
        // meshed, unhidden and cull-visible it reports itself to be - which is exactly what
        // this report used to describe as "engine is drawing this chunk".
        //
        // Answered by whichever rule ownership itself would use, and the report says which.
        // The two disagree by up to 62 blocks, so a report that quoted one while ownership
        // used the other would describe a boundary that is not the one on screen.
        float viewDistance = ApprovedViewDistance();
        string rangeDetail;
        bool beyondDrawRange;
        if (ExactDrawRange && drawRange.Available
            && VanillaChunkGeometry.TryReadDrawBounds(
                vanillaChunk, out float sphereX, out float sphereZ, out int cellLodMask))
        {
            beyondDrawRange = !drawRange.DrawsWithMargin(
                sphereX, sphereZ, cellLodMask, OverlapMarginBlocks);
            double centreDistance = Math.Sqrt(VanillaDrawRangeState.HorizontalDistanceSq(
                drawRange.PlayerBlockX, drawRange.PlayerBlockZ, sphereX, sphereZ));
            rangeDetail = $"measured: the game draws this chunk from a point {centreDistance:0.0} "
                + $"blocks away and stops at {Math.Sqrt(drawRange.ViewDistanceSq):0}, "
                + $"less a {OverlapMarginBlocks:0}-block early handover "
                + $"(pools {VanillaChunkGeometry.LodMaskText(cellLodMask)})";
        }
        else
        {
            beyondDrawRange = VanillaRenderReadiness.BeyondVanillaDrawRange(
                cell.X, cell.Z, camPos.X, camPos.Z, viewDistance, OverlapMarginBlocks);
            double faceDistance = VanillaRenderReadiness.ColumnDistanceBlocksFrom(
                cell.X, cell.Z, camPos.X, camPos.Z);
            rangeDetail = $"estimated: nearest face {faceDistance:0.0} blocks against a "
                + $"{Math.Max(0f, viewDistance - VanillaRenderReadiness.SphereCentreSlackBlocks - Math.Max(0f, OverlapMarginBlocks)):0}-block "
                + "limit, because this chunk holds no readable mesh";
        }
        string vanillaHolds = vanillaChunk == null ? "chunk not loaded"
            : vanillaChunk.Empty ? "chunk is empty (air)"
            : beyondDrawRange
                ? $"chunk is beyond the draw range, so the engine range-culls it -- {rangeDetail}"
            : !geometryKnown ? "engine draw state unavailable"
            : drawingNow ? "engine is drawing this chunk"
            : "engine is NOT drawing this chunk";

        return $"chunk {cell.X},{cell.Y},{cell.Z}: we say {state}, engine says "
            + $"{(renderedNow ? "rendered" : "not rendered")}, {vanillaHolds}, "
            + $"mask {(suppressed ? "suppresses" : "allows")} "
            + $"cached terrain here; L0 section {(world.Sections.ContainsKey(sectionKey) ? "resident" : "absent")}, "
            + $"{(HasAnyMesh(sectionKey) ? "meshed" : "no mesh")}"
            + (state == VanillaReadinessState.VanillaReady && !renderedNow
                ? " -- STALE OWNERSHIP: the mask is hiding cached terrain the engine is not drawing"
                : "")
            // The boundary case worth naming outright: we suppress the cache, the engine
            // says it drew this chunk, the chunk is not air, and the engine holds no mesh
            // for it. Nothing draws that ground, and nothing will correct it on its own.
            + (suppressed && beyondDrawRange
                ? " -- NOTHING DRAWS THIS: suppressed cache beyond the range the engine draws at all"
                : "")
            + (suppressed && !beyondDrawRange && renderedNow && vanillaChunk is { Empty: false }
                    && geometryKnown && !holdsGeometry
                ? " -- NOTHING DRAWS THIS: suppressed cache, and the engine claims the chunk without holding terrain"
                : "")
            // Printed for every cell, not only the ones that fail. The question this report is
            // asked at a seam is "why does the boundary sit here", and the range clause is the
            // one term whose answer is a number rather than a flag.
            + $"; range {rangeDetail}";
    }

    /// <summary>
    /// Uploads the whole mask when it differs from what the GPU holds. The public client
    /// API has no subregion update, so partial changes still cost a full upload; coalescing
    /// to one upload per frame is what keeps that affordable. A failed upload disables the
    /// mask outright and leaves the measured radius as the only ownership decision, because
    /// a stale mask would suppress cached terrain vanilla has not replaced.
    /// </summary>
    int waterEdgeChunks = -1;

    /// <summary>
    /// How far water's footprint is grown past the ground footprint, in whole chunk columns.
    /// Negative derives it; zero or more overrides it. See `.dlwater edge`.
    /// </summary>
    public int WaterEdgeChunks
    {
        get => waterEdgeChunks;
        set => waterEdgeChunks = Math.Clamp(value, -1, 8);
    }

    /// <summary>
    /// The derived growth, and the subtraction in it is the whole point.
    ///
    /// Ownership is refused for <c>SphereCentreSlackBlocks</c> plus
    /// <see cref="OverlapMarginBlocks"/> - 62 blocks when this was written, and ZERO at
    /// today's defaults, because the exact rule removed the slack and 0.3.163 took the margin
    /// to nothing - so cached ground keeps covering chunks the engine may already have
    /// dropped. The worked example below is the 62-block case it was derived from; the
    /// arithmetic is what matters, not the number it was first run on. The obvious reading is
    /// that water must grow back all 62 blocks, two chunks, and that is what 0.3.151 did. It
    /// overshot by exactly one chunk, measured in play as a chunk-wide strip with no water in
    /// it at all.
    ///
    /// The reason is which FACE the refusal measures from. It compares the column's NEAREST
    /// face against the threshold, so the outermost column that still owns anything has its
    /// near face just inside it and its FAR face a whole chunk further out again. One chunk of
    /// the ring is therefore already inside the ground footprint, and only what remains has to
    /// be grown back: 62 - 32 = 30 blocks, which is one chunk.
    ///
    /// Rounding UP on what remains, because falling short leaves cached water inside live
    /// terrain and overshooting leaves a gap with no water in it - and of the two, the strip
    /// of missing water is the one the owner can see from anywhere.
    /// </summary>
    int WaterFootprintDilationChunks()
    {
        if (waterEdgeChunks >= 0) return waterEdgeChunks;

        float ring = RefusedOwnershipRingBlocks();
        float remaining = ring - VanillaReadinessMask.ChunkBlocks;
        return Math.Clamp(
            (int)Math.Ceiling(Math.Max(0f, remaining) / VanillaReadinessMask.ChunkBlocks), 0, 8);
    }

    /// <summary>
    /// How far inside vanilla's real edge ground ownership currently stops, in blocks.
    ///
    /// <para>
    /// Under the exact rule this is the early handover ALONE - because the 46 blocks of
    /// sphere-placement slack are what reading the sphere removes - and since 0.3.163 that
    /// handover is ZERO, so the ring is zero and ownership reaches vanilla's true edge.
    ///
    /// The handover was 16 blocks and was believed to cover vanilla unloading a chunk that is
    /// still inside its draw range, which neither the sphere nor the player position can see
    /// coming. It was never shown to be covering that. What it was actually covering were two
    /// defects in the seam itself, both since fixed (G136, G137), and with those gone it buys
    /// nothing and costs a permanently doubled band. `.dlband overlap <blocks>` puts it back
    /// live, and if a hole ever returns at the seam, that it helps is a clue about where to
    /// look rather than the answer.
    /// </para>
    /// <para>
    /// Everything derived from the ring has to follow it, or the derivation describes a
    /// boundary that is no longer there. Water is the one that shows: its footprint is the
    /// ground one grown by whatever the ring refused, so leaving the growth at a chunk while
    /// ground reaches vanilla's true edge would put cached water a chunk INSIDE vanilla's own -
    /// the flat sheet standing in live terrain that 0.3.152 removed. At a 16-block ring the
    /// growth derives to zero, which is the correct answer and not a coincidence: the refusal
    /// measures to the nearest face, so the outermost owned chunk already reaches a full chunk
    /// past the threshold (G132).
    /// </para>
    /// <para>
    /// The three conditions are exactly what the exact path needs to be reachable at all: the
    /// switch, the engine's culler, and the mesh binding. A chunk can still fall back
    /// individually - air, or no mesh - but those hold no water for the footprint to be wrong
    /// about.
    /// </para>
    /// </summary>
    float RefusedOwnershipRingBlocks()
    {
        float margin = Math.Max(0f, OverlapMarginBlocks);
        if (ExactDrawRange && drawRange.Available && VanillaChunkGeometry.Available) return margin;
        return VanillaRenderReadiness.SphereCentreSlackBlocks + margin;
    }

    void PublishReadinessMask(bool rebuildWaterFootprint = true)
    {
        if (readinessMask == null || readinessMaskFailed) return;

        // Water's footprint is derived from the ground one, so it is rebuilt whenever
        // ownership has moved and before the atlas is uploaded. Keyed on the mask's revision
        // rather than on Dirty: upload clears Dirty, and a rebuild that ran after the upload
        // would sit in the buffer unseen until something else happened to dirty it.
        //
        // The dilation itself is the second trigger, and it needs its own: it changes when the
        // refused ring changes - `.dlexact`, `.dlband overlap`, `.dlwater edge`, or the exact
        // path becoming unavailable - and none of those touches ownership, so a revision-only
        // gate would leave water on the old growth until something unrelated moved the mask
        // (G66: a switch that changes how the renderer decides must invalidate what it decided).
        int dilation = WaterFootprintDilationChunks();
        bool footprintRebuilt = false;
        // The late range check passes false. The cached-water depth pass has already read this
        // frame's footprint by then, so the change waits for the next preparation and every
        // water pass within one frame reads the same footprint.
        if (rebuildWaterFootprint && readiness != null
            && (waterFootprintRevision != readinessMask.Revision || waterFootprintDilation != dilation))
        {
            readinessMask.RebuildWaterFootprint(
                readiness.ActiveMinChunkX, readiness.ActiveMinChunkZ,
                readiness.ActiveWidth, readiness.ActiveDepth, dilation);
            waterFootprintRevision = readinessMask.Revision;
            waterFootprintDilation = dilation;
            footprintRebuilt = true;
        }

        // A footprint the late check deferred is rebuilt after the ground it follows was already
        // uploaded, so it has to go up even when nothing else changed.
        if (!readinessMask.Dirty && readinessMaskActive && !footprintRebuilt) return;

        long started = Stopwatch.GetTimestamp();
        try
        {
            // The client creates when TextureId is 0 or the pixel count disagrees with the
            // stored size, and otherwise does a plain TexSubImage2D of the same extent, so
            // a correctly sized LoadedTexture makes one call cover both. Mipmaps are built
            // on the creation branch only, and texelFetch ignores filtering regardless.
            LoadedTexture texture = readinessMaskTexture
                ?? new LoadedTexture(capi, 0, readinessMask.Width, readinessMask.Height);
            UploadOwnTexture(readinessMask.Texels, ReadinessMaskUnit, ref texture);
            if (texture.TextureId <= 0)
                throw new InvalidOperationException("Readiness mask texture was not created.");
            readinessMaskTexture = texture;
            if (SkyMaskVerify) VerifySkyMaskUpload(texture, readinessMask.Texels, ReadinessMaskUnit);

            readinessMask.MarkUploaded();
            readinessMaskActive = true;
            // Mixed-ownership seam sections bypass temporal hiding altogether. A mask
            // upload therefore needs no global reset: newly mixed sections draw normally,
            // while cache-only hidden sections keep their periodic exact probes.
        }
        catch (Exception e)
        {
            DisableReadinessMask();
            capi.Logger.Warning(
                "[Landscapes] chunk readiness mask disabled; the measured handoff radius remains: {0}",
                e.Message);
        }
        finally
        {
            readinessMaskUploadUs =
                (Stopwatch.GetTimestamp() - started) * 1_000_000 / Stopwatch.Frequency;
        }
    }

    /// <summary>
    /// Changes the saved master blend setting. The seam resource can disappear immediately;
    /// packed L0 geometry cannot, because an already-published textured quad needs the table
    /// even when atlas sampling is off in the fragment shader. Those sections are replayed
    /// through the ordinary bounded remesh path, and the table is released after the final
    /// textured live mesh has been replaced.
    /// </summary>
    public int SetBlendPipelineEnabled(bool enabled)
    {
        if (blendPipelineEnabled == enabled) return 0;

        blendPipelineEnabled = enabled;
        int queued = 0;
        if (!enabled)
        {
            ReleaseBlendField();
            queued = SetTextureSlotEmission(false);
            MaybeReleaseBlockTextureTable();
        }
        else if (textureSlotSourceReady && BlockTextureTableCurrent)
        {
            queued = SetTextureSlotEmission(true);
        }

        capi.Logger.Notification(
            "[Landscapes] seam blending and L0 textures {0}; {1} live L0 section(s) "
            + "queued for bounded re-meshing.", enabled ? "on" : "off", queued);
        return queued;
    }

    /// <summary>The saved request, plus any short-lived off transition still being remeshed.</summary>
    public string DescribeBlendPipeline()
    {
        if (blendPipelineEnabled)
        {
            if (blockTableFailed)
                return "on (saved): seam blending is enabled; L0 textures failed for this session.";
            if (!textureSlotSourceReady)
                return "on (saved): seam blending is enabled; resident L0 texture slots are being prepared.";
            return "on (saved): seam blending and L0 block textures are enabled"
                + (textureSlotsReady ? "." : "; L0 textures are waiting for their GPU table.");
        }

        return liveTexturedMeshes > 0
            ? $"off (saved): {liveTexturedMeshes} textured L0 mesh(es) are still being replaced; "
                + "atlas sampling is already off."
            : "off (saved): seam blending and L0 block textures are fully disabled.";
    }

    bool BlockTextureTableCurrent => blockTexturesLive && blockTableTexture != 0
        && blockTextures != null && blockTableVersion == blockTextures.Version;

    int SetTextureSlotEmission(bool enabled)
    {
        if (textureSlotsReady == enabled) return 0;

        textureSlotsReady = enabled;
        unchecked { textureModeRevision++; }

        int queued = 0;
        foreach (long key in sectionMeshes.Keys)
        {
            if (LodWorld.KeyLevel(key) != 0) continue;
            world.RenderDirty.Add(key);
            queued++;
        }
        return queued;
    }

    void MaybeReleaseBlockTextureTable()
    {
        if (textureSlotsReady || liveTexturedMeshes > 0) return;
        if (WaterSurfaceTexture && WaterSurfaceTextureSlot > 0 && !blockTableFailed) return;
        if (blendPipelineEnabled && !blockTableFailed) return;
        DisposeBlockTableTexture();
    }

    void ReleaseBlendField()
    {
        seamField = null;
        seamFieldBuiltRevision = -1;
        seamFieldBuiltMarginCells = -1;
        DisposeSeamFieldTexture();
    }

    /// <summary>
    /// Uploads the block texture table when it has grown, and turns texture-slot mesh
    /// emission on only once the table is actually on the GPU.
    ///
    /// Ordering is the whole point of doing this at the top of the frame: a textured quad
    /// carries a block slot INSTEAD of its colour, so a slot the table does not yet describe
    /// would draw as black ground. Slots are handed out on the game tick as sections resolve;
    /// uploading before any draw means every slot a mesh can be carrying is already here.
    ///
    /// Until the first successful upload no mesh job carries a slot at all, so every failure
    /// path here is simply the terrain that was drawn before this feature existed.
    /// </summary>
    void PublishBlockTextureTable()
    {
        bool waterNeedsTable = WaterSurfaceTexture && WaterSurfaceTextureSlot > 0;
        if (!blendPipelineEnabled && !waterNeedsTable)
        {
            MaybeReleaseBlockTextureTable();
            return;
        }
        if (blendPipelineEnabled && !textureSlotSourceReady)
        {
            SetTextureSlotEmission(false);
            if (!waterNeedsTable) return;
        }
        if (blockTextures == null || blockTableFailed) return;
        if (BlockTextureTableCurrent)
        {
            SetTextureSlotEmission(blendPipelineEnabled && textureSlotSourceReady);
            return;
        }

        int replacementTexture = 0;
        try
        {
            // Ten rows per slot: the top's atlas rectangle, the averaged colour the quad no
            // longer carries, the mean to normalise by, the grass-style overlay's rectangle,
            // a rectangle and a mean for each wall axis, then each wall axis's colour (see
            // RowsPerSlot).
            // Float, because an atlas rectangle in eight bits would land a fragment on the
            // wrong block's texture.
            var pixels = new float[LodBlockTextureRegistry.MaxSlots
                * LodBlockTextureRegistry.FloatsPerSlot];
            IReadOnlyList<float> table = blockTextures.Table;
            for (int slot = 0; slot < LodBlockTextureRegistry.MaxSlots; slot++)
            {
                int from = slot * LodBlockTextureRegistry.FloatsPerSlot;
                for (int row = 0; row < LodBlockTextureRegistry.RowsPerSlot; row++)
                {
                    int at = (row * LodBlockTextureRegistry.MaxSlots + slot) * 4;
                    pixels[at + 0] = table[from + row * 4 + 0];
                    pixels[at + 1] = table[from + row * 4 + 1];
                    pixels[at + 2] = table[from + row * 4 + 2];
                    pixels[at + 3] = table[from + row * 4 + 3];
                }
            }

            replacementTexture = GL.GenTexture();
            if (replacementTexture == 0)
                throw new InvalidOperationException("block texture table was not created.");

            // Upload to a new object. Re-defining the live texture in place makes a failed
            // refresh destroy the only table old slot-bearing meshes can still read.
            GL.BindTexture(TextureTarget.Texture2D, replacementTexture);
            GL.TexParameter(TextureTarget.Texture2D,
                TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D,
                TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D,
                TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            GL.TexParameter(TextureTarget.Texture2D,
                TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba32f,
                LodBlockTextureRegistry.MaxSlots, LodBlockTextureRegistry.RowsPerSlot, 0,
                PixelFormat.Rgba, PixelType.Float, pixels);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            int retiredTexture = blockTableTexture;
            blockTableTexture = replacementTexture;
            replacementTexture = 0;
            blockTableVersion = blockTextures.Version;
            blockTexturesLive = true;
            if (retiredTexture != 0)
            {
                try { GL.DeleteTexture(retiredTexture); }
                catch { /* the replacement is already live; retirement is best effort */ }
            }
            SetTextureSlotEmission(blendPipelineEnabled && textureSlotSourceReady);
        }
        catch (Exception e)
        {
            try { GL.BindTexture(TextureTarget.Texture2D, 0); }
            catch { /* preserve the original failure */ }
            if (replacementTexture != 0)
            {
                try { GL.DeleteTexture(replacementTexture); }
                catch { /* preserve the original failure */ }
            }

            blockTableFailed = true;
            // Keep a previous successful table live for already-published textured meshes.
            // The revision flip below prevents any queued result carrying newer slots from
            // publishing, and the table retires once the old live count reaches zero.
            blockTexturesLive = blockTableTexture != 0 && blockTexturesLive;
            SetTextureSlotEmission(false);
            MaybeReleaseBlockTextureTable();
            capi.Logger.Warning(
                "[Landscapes] block textures disabled for this session; distant ground "
                + "keeps its averaged colours: {0}", e.Message);
        }
    }

    void DisposeBlockTableTexture()
    {
        if (blockTableTexture == 0) return;
        try { GL.DeleteTexture(blockTableTexture); }
        catch { /* teardown must not throw over a texture the driver already released */ }
        blockTableTexture = 0;
        blockTexturesLive = false;
    }

    /// <summary>
    /// Rebuilds and uploads the seam distance field the blend band reads.
    ///
    /// The band is derived entirely from committed ownership, so it needs no signal of its
    /// own and cannot disagree with the mask about where vanilla is. It also decides no
    /// pixel's visibility: every failure here costs the correction and leaves cached terrain
    /// drawing exactly as it does today, which is why none of this can open a hole.
    /// </summary>
    void PublishSeamField()
    {
        if (seamFieldFailed) return;
        if (!blendPipelineEnabled || !BandEnabled
            || !readinessMaskActive || readinessMask == null || readiness == null)
        {
            seamFieldActive = false;
            return;
        }

        int marginCells = SeamFieldMarginCells();
        int required = NextPowerOfTwo(Math.Max(
            readiness.ActiveWidth * VanillaSeamDistanceField.CellsPerChunk + marginCells * 2,
            readiness.ActiveDepth * VanillaSeamDistanceField.CellsPerChunk + marginCells * 2));
        if (required > BandMaxCapacityCells)
        {
            // Not permanent: lowering the vanilla view distance brings this back under the
            // limit, and lowering it is the whole point of the mod.
            seamFieldActive = false;
            if (!seamFieldCapacityWarned)
            {
                seamFieldCapacityWarned = true;
                capi.Logger.Notification(
                    "[Landscapes] blend band off: a {0}-cell seam field is past the {1}-cell limit "
                    + "at this vanilla view distance.", required, BandMaxCapacityCells);
            }
            return;
        }
        seamFieldCapacityWarned = false;

        try
        {
            if (seamField == null || seamField.Capacity != required)
            {
                seamField = new VanillaSeamDistanceField(required);
                DisposeSeamFieldTexture();
                seamFieldBuiltRevision = -1;
            }

            // A moved window rebuilds on the spot. The ring hands a reused slot to different
            // world coordinates, so anything else would address another place's distance.
            //
            // A changed margin does too, and it needs saying separately: the margin follows
            // the band's inner offset, which follows the refused ring, which `.dlexact` moves
            // by 46 blocks in one step. The capacity above only reacts when that crosses a
            // power of two, so without this the field would keep serving distances built for
            // a reach it no longer has - a switch that changed how the renderer decides
            // without invalidating what it decided (G66).
            bool windowMoved = seamFieldWindowMinChunkX != readiness.ActiveMinChunkX
                || seamFieldWindowMinChunkZ != readiness.ActiveMinChunkZ
                || seamFieldWindowWidth != readiness.ActiveWidth
                || seamFieldWindowDepth != readiness.ActiveDepth
                || seamFieldBuiltMarginCells != marginCells;
            long now = capi.ElapsedMilliseconds;
            bool ownershipMoved = seamFieldBuiltRevision != readinessMask.Revision;

            if (windowMoved
                || (ownershipMoved && now - seamFieldLastBuildMs >= BandRebuildIntervalMilliseconds))
            {
                long buildStarted = Stopwatch.GetTimestamp();
                seamField.Build(readinessMask,
                    readiness.ActiveMinChunkX, readiness.ActiveMinChunkZ,
                    readiness.ActiveWidth, readiness.ActiveDepth, marginCells);
                seamFieldBuildUs =
                    (Stopwatch.GetTimestamp() - buildStarted) * 1_000_000 / Stopwatch.Frequency;
                seamFieldBuiltRevision = readinessMask.Revision;
                seamFieldLastBuildMs = now;
                seamFieldWindowMinChunkX = readiness.ActiveMinChunkX;
                seamFieldWindowMinChunkZ = readiness.ActiveMinChunkZ;
                seamFieldWindowWidth = readiness.ActiveWidth;
                seamFieldWindowDepth = readiness.ActiveDepth;
                seamFieldBuiltMarginCells = marginCells;
            }

            if (!seamField.Dirty && seamFieldActive) return;

            long uploadStarted = Stopwatch.GetTimestamp();
            try
            {
                LoadedTexture texture = seamFieldTexture
                    ?? new LoadedTexture(capi, 0, seamField.Width, seamField.Height);
                UploadOwnTexture(seamField.Texels, SeamFieldUnit, ref texture);
                if (texture.TextureId <= 0)
                    throw new InvalidOperationException("Seam distance field texture was not created.");
                seamFieldTexture = texture;
                seamField.MarkUploaded();
                seamFieldActive = true;
            }
            finally
            {
                seamFieldUploadUs =
                    (Stopwatch.GetTimestamp() - uploadStarted) * 1_000_000 / Stopwatch.Frequency;
            }
        }
        catch (Exception e)
        {
            DisableSeamField();
            capi.Logger.Warning(
                "[Landscapes] blend band disabled for this session; cached terrain keeps its "
                + "current colouring: {0}", e.Message);
        }
    }

    /// <summary>
    /// How far past the mask's own window the field must reach, in 8-block cells. The mask
    /// tracks only a couple of chunks past vanilla, and the band lives entirely outside
    /// vanilla, so the field needs the band's full reach plus one cell of slack.
    ///
    /// The reach is the band's width PLUS its inner offset, and leaving the inner offset out
    /// was a visible defect rather than a rounding question. The shader ramps on
    /// <c>seamBlocks = seamCells * 8 - bandInnerBlocks</c>, so the weight only reaches zero
    /// <c>BandWidthBlocks + BandInnerBlocks</c> past the last owned column - 174 blocks at the
    /// defaults, against a margin sized for 136. Every cell past the margin is outside the
    /// field's window and reads no band at all, so the ramp was being cut while it still
    /// carried a weight of 0.31 and the correction fell to zero in a single step, on the
    /// window's own straight edge. That is a hard line across the world, in the one place the
    /// whole design exists to avoid having one.
    ///
    /// The capacity clamp can still truncate it at extreme settings, where the same step
    /// returns; `.dlband map` reports the weight the field actually ends on.
    /// </summary>
    int SeamFieldMarginCells()
    {
        float reach = Math.Max(0f, BandWidthBlocks) + Math.Max(0f, BandInnerBlocks);
        int cells = (int)Math.Ceiling(reach / VanillaSeamDistanceField.CellBlocks) + 1;
        return Math.Clamp(cells, 1, BandMaxCapacityCells / 4);
    }

    static int NextPowerOfTwo(int value)
    {
        int result = 1;
        while (result < value)
        {
            if (result > 1 << 29) return int.MaxValue;
            result <<= 1;
        }
        return result;
    }

    void DisableSeamField()
    {
        seamFieldFailed = true;
        seamFieldActive = false;
        seamField = null;
        DisposeSeamFieldTexture();
    }

    void DisposeSeamFieldTexture()
    {
        if (seamFieldTexture == null) return;
        try { capi.Render.GLDeleteTexture(seamFieldTexture.TextureId); }
        catch { /* teardown must not throw over a texture the driver already released */ }
        seamFieldTexture = null;
        seamFieldActive = false;
    }

    /// <summary>
    /// The near-field floor described on <see cref="MaskNearFloorBlocks"/>. Zero unless the
    /// camera's own ownership cell is committed ready, so uncertainty still fails toward
    /// cached coverage.
    /// </summary>
    float MaskNearFloor()
    {
        if (readiness == null) return 0f;

        var cell = new VanillaChunkCell(
            VanillaRenderReadiness.ChunkCoordinate(camPos.X),
            VanillaRenderReadiness.ChunkCoordinate(camPos.Y),
            VanillaRenderReadiness.ChunkCoordinate(camPos.Z));
        return readiness.State(cell) == VanillaReadinessState.VanillaReady
            ? Math.Max(0f, NearSuppressionBlocks)
            : 0f;
    }

    void DisableReadinessMask()
    {
        readinessMaskFailed = true;
        readinessMaskActive = false;
        readinessMask = null;
        DisposeReadinessMaskTexture();
        seamField = null;
        DisposeSeamFieldTexture();
    }

    void DisposeReadinessMaskTexture()
    {
        if (readinessMaskTexture == null) return;
        try { capi.Render.GLDeleteTexture(readinessMaskTexture.TextureId); }
        catch { /* teardown must not throw over a texture the driver already released */ }
        readinessMaskTexture = null;
        readinessMaskActive = false;
    }

    /// <summary>
    /// Whether the engine holds this chunk with no blocks in it. Separate from the rendered
    /// query on purpose: that one cannot distinguish "drew terrain" from "drew nothing".
    /// A chunk the engine does not hold at all is not empty, it is absent, and the rendered
    /// query has already answered for that case.
    /// </summary>
    /// <summary>
    /// Separates the two ways a chunk can report drawn while drawing nothing.
    ///
    /// <para>
    /// "Empty" is the ordinary one and is not a fault: the flag arrives from the server in
    /// the chunk packet, the tesselator reads exactly it to skip meshing, and every column
    /// has sky above it. Refusing ownership to those cells is what broke 0.3.3 - a column
    /// counts as owned only when all of its chunks do, so excluding air excluded every
    /// column at once.
    /// </para>
    /// <para>
    /// The one worth finding is the second: a chunk that is not empty, that the engine
    /// says it drew, and for which the engine holds no mesh at all. That is ground the
    /// player should be able to see and nothing is drawing, and it is the only shape of
    /// discrepancy that can leave a hole standing still. Zero here retires the theory.
    /// </para>
    /// </summary>
    /// <summary>
    /// Whether the engine holds this chunk with no blocks in it.
    ///
    /// <para>
    /// An air chunk keeps its ownership in the tracker, because a column counts as owned
    /// only when every one of its chunks does and every column has sky above it - refusing
    /// air there is what collapsed 0.3.3. But it must never suppress a fragment. Cached
    /// terrain is an approximation and stands taller than the real world in places, so its
    /// geometry sits inside air cells; vanilla draws nothing in an air cell, so discarding
    /// there removes the only thing that was drawing and cuts the top off the cached
    /// horizon. That is the band: painted red by `.dlpaint`, invisible to `.dlholes`
    /// because the tracker is right, and unaffected by every ownership rule tried, because
    /// ownership was never wrong.
    /// </para>
    /// <para>
    /// The cost is the overlap this was originally meant to remove: cached terrain may show
    /// above real ground where the approximation overshoots. A hole outranks an overlap.
    /// </para>
    /// </summary>
    bool VanillaChunkIsAir(VanillaChunkCell cell)
    {
        try
        {
            IWorldChunk? chunk = capi.World.BlockAccessor.GetChunk(cell.X, cell.Y, cell.Z);
            return chunk is { Empty: true };
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Rewrites the whole ownership atlas from the tracker's committed state.
    ///
    /// <para>
    /// The atlas is fed incrementally, by publication, and an incremental mirror is only
    /// ever as correct as the completeness of its update paths. One missing path - a column
    /// scrolling out of the ring without clearing its texels - held cached terrain suppressed
    /// over ground nothing was drawing, and no CPU-side check could see it, because every CPU
    /// structure was right and only the mirror was wrong. `.dlholes` in particular cannot
    /// find this class of fault at all: it walks cells the tracker owns, and the fault is a
    /// texel claiming ownership the tracker does not have.
    /// </para>
    /// <para>
    /// Rather than keep hunting for the next missing path, the mirror is rebuilt from the
    /// authority once a second. It costs one pass over the window and at most one upload that
    /// was already budgeted, and it bounds any future desync to a second by construction.
    /// Disagreements are counted, because a non-zero figure here means an update path is
    /// still missing and the resync is masking it.
    /// </para>
    /// </summary>
    void ResyncReadinessMask()
    {
        if (readiness == null || readinessMask == null || readinessMaskFailed) return;

        for (int z = readiness.ActiveMinChunkZ; z < readiness.ActiveMinChunkZ + readiness.ActiveDepth; z++)
        for (int x = readiness.ActiveMinChunkX; x < readiness.ActiveMinChunkX + readiness.ActiveWidth; x++)
        for (int y = 0; y < readiness.VerticalChunks; y++)
        {
            var cell = new VanillaChunkCell(x, y, z);
            // The chunk lookup stays authoritative here. The stored bit is a cache of this
            // answer for the paths that cannot ask - a rebuild, and the shader mirror - and
            // a chunk that stops being air after its publication would otherwise keep the
            // stale bit forever. Refreshing it here bounds that to one resync interval.
            bool air = VanillaChunkIsAir(cell);
            readiness.SetMaskExcluded(cell, air);
            bool owned = readiness.State(cell) == VanillaReadinessState.VanillaReady && !air;
            if (readinessMask.Set(cell, owned)) ReadinessMaskResyncs++;
        }
    }

    /// <summary>
    /// Sweeps every committed cell in the tracked window and reports the ones the engine is
    /// not actually drawing. Written because aiming a ray at a hole is unreliable - a band
    /// behind the player is not something a view ray finds, and the ground visible through
    /// a hole answers for itself. This needs no aiming and no hole in view.
    ///
    /// One pass over the window on demand, never on a frame budget.
    /// </summary>
    public string ExplainOwnedButNotDrawn(double cameraX, double cameraZ, int maxListed)
    {
        VanillaRenderReadiness? tracker = readiness;
        if (tracker == null) return "no readiness tracker; the distance handoff is drawing";

        int owned = 0;
        int notDrawn = 0;
        int unknown = 0;
        int beyondRange = 0;
        float viewDistance = ApprovedViewDistance();
        var worst = new List<(double Distance, string Text)>();

        for (int z = tracker.ActiveMinChunkZ; z < tracker.ActiveMinChunkZ + tracker.ActiveDepth; z++)
        for (int x = tracker.ActiveMinChunkX; x < tracker.ActiveMinChunkX + tracker.ActiveWidth; x++)
        for (int y = 0; y < tracker.VerticalChunks; y++)
        {
            var cell = new VanillaChunkCell(x, y, z);
            if (tracker.State(cell) != VanillaReadinessState.VanillaReady) continue;
            owned++;
            double distance = VanillaRenderReadiness.ColumnDistanceBlocksFrom(x, z, cameraX, cameraZ);

            // Tested before the chunk is even looked up. The engine range-culls every
            // terrain pool location per frame, so out here its answer is no regardless of
            // what the chunk's own flags say, and reading them would only produce the
            // reassuring verdict that hid this case in the first place.
            if (VanillaRenderReadiness.BeyondVanillaDrawRange(x, z, cameraX, cameraZ, viewDistance))
            {
                notDrawn++;
                beyondRange++;
                if (worst.Count < maxListed)
                {
                    worst.Add((distance, $"chunk {x},{y},{z} at {distance:0} blocks | "
                        + $"beyond the {viewDistance:0}-block draw range"));
                }
                continue;
            }

            IWorldChunk? chunk;
            try { chunk = capi.World.BlockAccessor.GetChunk(x, y, z); }
            catch { continue; }

            if (!VanillaChunkGeometry.TryIsVanillaDrawing(chunk, out bool drawing)) { unknown++; continue; }
            if (drawing) continue;

            notDrawn++;
            if (worst.Count < maxListed)
            {
                worst.Add((distance, $"chunk {x},{y},{z} at {distance:0} blocks | "
                    + VanillaChunkGeometry.DescribeSignals(chunk)));
            }
        }

        if (owned == 0) return "nothing is owned yet; the tracker has not committed a single cell.";
        if (notDrawn == 0)
        {
            return $"all {owned} owned cells are being drawn by the engine"
                + (unknown > 0 ? $" ({unknown} unmeasurable)" : "")
                + ". Ownership is not suppressing anything the engine has abandoned.";
        }

        worst.Sort((a, b) => a.Distance.CompareTo(b.Distance));
        return $"{notDrawn} of {owned} owned cells are NOT being drawn by the engine"
            + (beyondRange > 0 ? $" ({beyondRange} beyond the {viewDistance:0}-block draw range)" : "")
            + (unknown > 0 ? $" ({unknown} unmeasurable)" : "")
            + " -- cached terrain is suppressed there and nothing replaces it. Nearest: "
            + string.Join(" ;; ", worst.Select(w => w.Text));
    }

    /// <summary>
    /// Whether the engine holds uploaded terrain geometry here. An empty chunk qualifies:
    /// there is nothing to draw and nothing for the cache to cover, and refusing it ownership
    /// is what collapsed 0.3.3 because every column has sky. An unmeasurable chunk qualifies,
    /// so a game update can only cost the gain gate, never invent a confident refusal.
    ///
    /// This intentionally does not ask TryIsVanillaDrawing. That broader diagnostic also
    /// includes CullVisible and Hide, which can change while the pool geometry remains live;
    /// using it as this predicate was the ownership oscillation fixed in 0.4.49.
    /// </summary>
    bool VanillaChunkHoldsDrawableGround(IWorldChunk? chunk)
    {
        try
        {
            if (chunk == null) return true;
            if (chunk.Empty) return true;
            return !VanillaChunkGeometry.TryHoldsGeometry(chunk, out bool holdsGeometry)
                || holdsGeometry;
        }
        catch
        {
            return true;
        }
    }

    void MeasureDrawnChunk(VanillaChunkCell cell, IWorldChunk? chunk, double cameraX, double cameraZ)
    {
        try
        {
            if (chunk == null) return;

            if (chunk.Empty)
            {
                ReadinessEmptyChunksSeen++;
                return;
            }

            if (!VanillaChunkGeometry.TryHoldsGeometry(chunk, out bool holdsGeometry)) return;
            if (holdsGeometry) return;

            ReadinessDrawnWithoutGeometry++;

            // Only a cell we have actually committed can suppress cached terrain, so this
            // is the subset that could be showing the player a hole. Where it sits decides
            // which explanation survives: a buried chunk with no exposed faces also holds
            // no mesh, and that is invisible underground, while the same condition out at
            // the handoff ring is ground the player can see with nothing drawing it.
            if (readiness == null || readiness.State(cell) != VanillaReadinessState.VanillaReady) return;

            ReadinessOwnedWithoutGeometry++;
            if (cell.Y >= 0 && cell.Y < readinessNoGeometryByY.Length) readinessNoGeometryByY[cell.Y]++;

            double distance = VanillaRenderReadiness.ColumnDistanceBlocksFrom(
                cell.X, cell.Z, cameraX, cameraZ);
            if (ReadinessOwnedWithoutGeometryNearest < 0 || distance < ReadinessOwnedWithoutGeometryNearest)
                ReadinessOwnedWithoutGeometryNearest = distance;
            if (distance > ReadinessOwnedWithoutGeometryFarthest)
                ReadinessOwnedWithoutGeometryFarthest = distance;
        }
        catch
        {
            // A diagnostic must never be able to change ownership, including by throwing
            // into a handler that treats failure as "vanilla is not drawing here".
        }
    }

    public string DescribeReadiness()
    {
        if (readinessDisabled) return "shadow disabled (radial fallback active)";
        if (readiness == null) return "shadow not initialized";
        readiness.GetStateCounts(out int unknown, out int pending, out int observed, out int ready);
        long candidateFrame = readiness.OldestCandidateFrame();
        long observedFrame = readiness.OldestObservedFrame();
        long oldestAge = Math.Max(
            candidateFrame < 0 ? 0 : frameCounter - candidateFrame,
            observedFrame < 0 ? 0 : frameCounter - observedFrame);
        if (readinessColumnReady.Length < readiness.VerticalChunks)
            readinessColumnReady = new int[readiness.VerticalChunks];
        readiness.GetColumnReadiness(readinessColumnReady, out int columnsTracked,
            out int fullColumns, out int partialColumns, out int maxReadyPerColumn);
        readinessColumnText.Clear();
        readinessNoGeometryText.Clear();
        for (int y = 0; y < readiness.VerticalChunks; y++)
        {
            if (y > 0) readinessColumnText.Append('/');
            readinessColumnText.Append(readinessColumnReady[y]);
            if (y > 0) readinessNoGeometryText.Append('/');
            readinessNoGeometryText.Append(readinessNoGeometryByY[y]);
        }

        double nearestIncomplete = readiness.NearestIncompleteColumnBlocks(
            camPos.X, camPos.Z, out double nearestUnready);
        // Kept out of the format literal: an interpolation hole containing a quoted string
        // is not machine-readable, and the benchmark gate parses this exact line.
        float radialHandoff = LodNearHandoff.InnerDiscardRadius(ApprovedViewDistance());
        // Report what the shader actually receives. A live mask drives the radius to zero,
        // and printing the radius it would otherwise have used reads as suppression that
        // is not happening.
        bool maskOwnsPixels = readinessMaskActive && readinessMask != null;
        float appliedHandoff = maskOwnsPixels ? 0f
            : readinessOwnsHandoff ? readinessHandoffDistance : radialHandoff;
        string handoffSource = maskOwnsPixels ? "mask"
            : readinessOwnsHandoff ? "readiness" : "radial";
        string maskState = readinessMaskFailed ? "failed"
            : readinessMask == null ? "off"
            : readinessMaskActive
                ? $"{readinessMask.ReadyTexels} owned/{readinessMask.WaterColumns} water columns/{readinessMask.Bytes / 1024} KiB/"
                  + $"{readinessMask.Uploads} uploads/last {readinessMaskUploadUs}us"
                : "pending";

        string fresnelState = WaterFresnel ? "on" : "off";
        string maskPaintState = MaskDebugPaint ? "ON" : "off";
        string bandPaintState = BandDebugPaint ? "ON" : "off";
        return $"shadow {readiness.ActiveWidth}x{readiness.VerticalChunks}x{readiness.ActiveDepth}, "
            + $"{ready} ready/{observed} observed/{pending} pending/{unknown} unknown, "
            + $"{readiness.PendingCandidates} probes/{readiness.PendingObservations} observations queued, "
            + $"oldest work {oldestAge} frames, {readiness.TrackedArrayBytes / 1024.0:0.0} KiB arrays, "
            + $"interval {ReadinessProbes} probes ({ReadinessTrueResults} true/{ReadinessFalseResults} false), "
            + $"{ReadinessReadyTransitions} ready/{ReadinessLostTransitions} lost transitions, "
            + $"{ReadinessProbeErrors} errors, {ReadinessWindowChanges} window changes/{ReadinessResizes} resizes, "
            + $"{ReadinessFullRevalidations} full revalidations, "
            + $"range {ExactRangeState()}, "
            + $"{ReadinessStaleCommittedFound} stale committed found/{ReadinessAggregateRepairs} count repairs, "
            + $"{ReadinessEmptyChunksSeen} drawn-but-empty/{ReadinessDrawnWithoutGeometry} drawn-without-geometry chunks, "
            + $"owned without geometry {ReadinessOwnedWithoutGeometry} at {ReadinessOwnedWithoutGeometryNearest:0}-{ReadinessOwnedWithoutGeometryFarthest:0} blocks per Y {readinessNoGeometryText}, {ReadinessOwnershipDeniedNoGeometry} ownership denied no-geometry/{ReadinessOwnershipDeniedBeyondViewDistance} denied beyond view distance/{ReadinessMaskResyncs} mask resyncs, "
            + $"events {readiness.CandidateEventsAccepted} accepted/{readiness.CandidateEventsCoalesced} coalesced/"
            + $"{readiness.CandidateEventsDropped} dropped, "
            + $"sweeps {readiness.ScheduledCandidatesAccepted} accepted/"
            + $"{readiness.ScheduledCandidatesCoalesced} coalesced, "
            + $"columns {columnsTracked} tracked/{fullColumns} full/{partialColumns} partial, "
            + $"max {maxReadyPerColumn}/{readiness.VerticalChunks} ready per column, "
            + $"ready per Y {readinessColumnText}, "
            + $"nearest incomplete {nearestIncomplete:0} blocks/unready {nearestUnready:0} blocks, "
            + $"handoff {appliedHandoff:0} blocks ({handoffSource}, radial {radialHandoff:0}), "
            + $"mask {maskState}, {VanillaOwnedDrawsSkipped} owned draws skipped, "
            + $"band {BandState()}, textures {BlockTextureState()}, "
            + $"near suppression {NearSuppressionBlocks:0} blocks, "
            // Water settings and debug paints ride the periodic report because a screenshot
            // cannot show which of them is active, and every square, ring and line this work
            // produced was attributed to the wrong one at least once for want of these.
            //
            // Composed into locals first, never inline conditionals: the benchmark runner
            // parses this line by extracting the interpolated literal, and a nested quote
            // inside a hole terminates that extraction. A static check holds it.
            + $"water near {waterNearChunks} chunks/fresnel {fresnelState}"
            + $"/glint {waterGlint:0.00}/foam {waterFoam:0.00}, "
            + $"paints mask {maskPaintState}/band shape {bandPaintState}";
    }

    /// <summary>
    /// How much ground the exact rule is actually taking back, measured on the running client
    /// and written to the log.
    ///
    /// <para>
    /// This exists because the change it reports on is invisible from inside the game. Both
    /// rules draw terrain everywhere; they differ only in how far cached terrain keeps drawing
    /// on top of vanilla's, and a seam that overlaps by two chunks and one that does not look
    /// like the same seam from anywhere except a spot the player has to find. The map prints
    /// the disagreement directly: which columns the old bound would have refused and the
    /// engine's own measurement grants.
    /// </para>
    /// <para>
    /// Read entirely from tracker state - the recorded spheres and the committed cells - so it
    /// takes no chunk lock however large the window is, and reports what ownership actually
    /// did rather than what a fresh probe would say now.
    /// </para>
    /// </summary>
    public string LogExactRangeMap()
    {
        if (readiness == null) return "no readiness tracker: " + ExactRangeState();

        float viewDistance = ApprovedViewDistance();
        double exactEdge = drawRange.Available
            ? Math.Sqrt(drawRange.ViewDistanceSq) - Math.Max(0f, OverlapMarginBlocks)
            : 0;
        double boundedEdge = Math.Max(0f, viewDistance
            - VanillaRenderReadiness.SphereCentreSlackBlocks - Math.Max(0f, OverlapMarginBlocks));

        int camChunkX = VanillaRenderReadiness.ChunkCoordinate(camPos.X);
        int camChunkZ = VanillaRenderReadiness.ChunkCoordinate(camPos.Z);
        int stride = Math.Max(1, readiness.ActiveWidth / 96);

        int reclaimed = 0, measured = 0, bounded = 0, ownedColumns = 0;
        double nearestReclaimed = double.MaxValue, farthestReclaimed = 0;

        var map = new StringBuilder();
        for (int z = 0; z < readiness.ActiveDepth; z += stride)
        {
            map.Append('\n').Append("    ");
            for (int x = 0; x < readiness.ActiveWidth; x += stride)
            {
                int chunkX = readiness.ActiveMinChunkX + x;
                int chunkZ = readiness.ActiveMinChunkZ + z;
                if (chunkX == camChunkX && chunkZ == camChunkZ) { map.Append('+'); continue; }

                bool owned = false, hasSphere = false;
                for (int y = 0; y < readiness.VerticalChunks; y++)
                {
                    var cell = new VanillaChunkCell(chunkX, y, chunkZ);
                    if (readiness.State(cell) != VanillaReadinessState.VanillaReady) continue;
                    owned = true;
                    if (readiness.TryGetDrawSphere(cell, out _, out _, out _)) { hasSphere = true; break; }
                }

                if (!owned) { map.Append('.'); continue; }
                ownedColumns++;

                bool boundWouldDeny = VanillaRenderReadiness.BeyondVanillaDrawRange(
                    chunkX, chunkZ, camPos.X, camPos.Z, viewDistance, OverlapMarginBlocks);

                if (hasSphere && boundWouldDeny)
                {
                    // The whole point: ground the old rule handed back to cached terrain and
                    // the engine's own measurement says vanilla is still drawing.
                    reclaimed++;
                    double distance = VanillaRenderReadiness.ColumnDistanceBlocksFrom(
                        chunkX, chunkZ, camPos.X, camPos.Z);
                    if (distance < nearestReclaimed) nearestReclaimed = distance;
                    if (distance > farthestReclaimed) farthestReclaimed = distance;
                    map.Append('X');
                }
                else if (hasSphere) { measured++; map.Append('#'); }
                else { bounded++; map.Append('B'); }
            }
        }

        string reclaimedRing = reclaimed == 0
            ? "none"
            : $"{nearestReclaimed:0}-{farthestReclaimed:0} blocks out";

        var report = new StringBuilder();
        report.Append("[Landscapes] draw-range ownership map: ")
            .Append(readiness.ActiveWidth).Append('x').Append(readiness.ActiveDepth)
            .Append(" chunk columns, ").Append(ownedColumns).Append(" owned (")
            .Append(measured).Append(" by the engine's own sphere, ")
            .Append(bounded).Append(" by the conservative bound, ")
            .Append(reclaimed).Append(" reclaimed from the old overlap ring at ")
            .Append(reclaimedRing).Append("). ")
            .Append("The old rule stopped ownership at ").Append($"{boundedEdge:0}")
            .Append(" blocks; the engine actually draws to ").Append($"{exactEdge:0}")
            .Append(" blocks, a ring ").Append($"{Math.Max(0, exactEdge - boundedEdge):0}")
            .Append(" blocks wide. '#' measured, 'X' reclaimed, 'B' bounded, '.' not owned, ")
            .Append("'+' the camera's own column.")
            .Append(map);

        capi.Logger.Notification(report.ToString());
        return $"draw-range map written to the client log: {measured} columns owned by the engine's "
            + $"own measurement, {reclaimed} reclaimed from the overlap ring ({reclaimedRing}), "
            + $"{bounded} still on the conservative bound. " + ExactRangeState();
    }

    /// <summary>
    /// Why cached meshes are not being dropped before they are drawn, measured over the
    /// sections actually resident right now and written to the log.
    ///
    /// <para>
    /// The whole-mesh skip fires when a section is classified vanilla-owned, and it has never
    /// fired - not once, in any log. The classification asks whether every 32-block cell in the
    /// section's box is owned, bedrock to sky, and the bottom of that box is owned by nothing:
    /// solid stone with no exposed face holds no mesh, and chunks the player has never been
    /// near were never tesselated, so neither reports drawn. One such cell anywhere in the
    /// column disqualifies the section permanently.
    /// </para>
    /// <para>
    /// The cached mesh does not occupy those cells. So this reports both answers side by side:
    /// what the current rule classifies, and what a rule counting only the vertical band the
    /// mesh actually occupies WOULD classify. The gap between the two columns is exactly the
    /// value of making that change, measured before building it rather than argued for.
    /// </para>
    /// </summary>
    public string LogSectionOwnershipMap()
    {
        if (readiness == null) return "no readiness tracker";
        if (!readinessMaskActive) return "the ownership mask is not live, so nothing can be skipped yet";

        int total = 0, cacheOnly = 0, mixed = 0, vanillaOnly = 0, noSpan = 0, bandOwned = 0;
        var gainedByLevel = new int[LodWorld.MaxLevel + 1];
        var totalByLevel = new int[LodWorld.MaxLevel + 1];

        foreach (KeyValuePair<long, LodLiveMeshStats> entry in liveMeshStats)
        {
            long key = entry.Key;
            int level = LodWorld.KeyLevel(key);
            if (level < 0 || level > LodWorld.MaxLevel) continue;
            total++;
            totalByLevel[level]++;

            VanillaSectionOwnership current = readiness.Classify(key);
            switch (current)
            {
                case VanillaSectionOwnership.CacheOnly: cacheOnly++; break;
                case VanillaSectionOwnership.Mixed: mixed++; break;
                default: vanillaOnly++; break;
            }

            LodHeightSpan span = entry.Value.Heights.Either;
            if (!span.HasGeometry) { noSpan++; continue; }

            int minChunkY = (int)Math.Floor(span.MinY / VanillaReadinessMask.ChunkBlocks);
            int maxChunkY = (int)Math.Floor(span.MaxY / VanillaReadinessMask.ChunkBlocks);
            if (!readiness.AllOwnedInBand(key, minChunkY, maxChunkY)) continue;

            bandOwned++;
            if (current != VanillaSectionOwnership.VanillaOnly) gainedByLevel[level]++;
        }

        int gained = 0;
        var perLevel = new StringBuilder();
        for (int level = 0; level <= LodWorld.MaxLevel; level++)
        {
            gained += gainedByLevel[level];
            if (totalByLevel[level] == 0) continue;
            if (perLevel.Length > 0) perLevel.Append(' ');
            perLevel.Append('L').Append(level).Append(' ')
                .Append(gainedByLevel[level]).Append('/').Append(totalByLevel[level]);
        }

        string verdict = total == 0
            ? "no cached meshes are resident, so there is nothing to classify."
            : gained == 0
                ? "counting only the occupied band would change NOTHING - the cells the cached "
                    + "mesh does occupy are not owned either, so the vertical box is not what is "
                    + "blocking the skip. Look at ownership itself, not at the classification."
                : $"counting only the occupied band would let {gained} of {total} resident "
                    + $"sections be dropped before drawing, where {vanillaOnly} qualify today.";

        string overshoot = DescribeGeometryOvershoot();

        var report = new StringBuilder();
        report.Append("[Landscapes] section ownership: ").Append(total)
            .Append(" resident cached meshes -- ").Append(cacheOnly).Append(" cache-only/")
            .Append(mixed).Append(" mixed/").Append(vanillaOnly)
            .Append(" vanilla-owned under the current whole-column rule; ")
            .Append(bandOwned).Append(" have every cell owned across the vertical band their own ")
            .Append("geometry occupies (").Append(noSpan).Append(" have no measured span). ")
            .Append(verdict);
        if (perLevel.Length > 0) report.Append(" Gained per level: ").Append(perLevel).Append('.');
        report.Append(' ').Append(overshoot);

        capi.Logger.Notification(report.ToString());
        return $"section ownership written to the client log: {vanillaOnly} of {total} resident "
            + $"sections qualify today, {bandOwned} would qualify counting only the band their "
            + $"geometry occupies ({gained} newly).";
    }

    /// <summary>
    /// The mask against the engine's own answer, one cell at a time. The audit that says
    /// whether ownership is under-covering, and for every cell it misses, why.
    ///
    /// <para>
    /// Every other report here measures the mod against itself: what the tracker believes, what
    /// it published, what it would classify. None of them can see the failure the player
    /// actually reports, which is cached terrain drawn on top of terrain the engine is drawing
    /// right now - because that is a disagreement between the mask and the ENGINE, and nothing
    /// was asking the engine. This walks the tracked window, evaluates the engine's whole
    /// predicate per cell, and counts the cells where the two disagree.
    /// </para>
    /// <para>
    /// The direction that matters is UNCOVERED: the engine is drawing this chunk and the mask
    /// does not hide cached terrain there. Every one of those is visible overlap. The reasons
    /// are broken out because they want completely different fixes - a cell still Pending is a
    /// latency problem, a cell refused for range is an arithmetic one, and an air cell is a
    /// deliberate rule (G40) that no amount of extra ownership will change.
    /// </para>
    /// <para>
    /// One chunk lookup per cell, so this takes the client's chunk lock several hundred times
    /// (G45). It is a command, run while standing at the problem, and must never be called from
    /// the frame path.
    /// </para>
    /// </summary>
    public string LogCoverageAudit()
    {
        if (readiness == null) return "no readiness tracker";
        if (!readinessMaskActive || readinessMask == null)
            return "the ownership mask is not live, so there is nothing to audit";
        // Without the culler there is no range clause, so every cell would fall out of the loop
        // before being counted and the report would say "the engine is drawing 0 chunks" - which
        // reads as a healthy world rather than as a broken measurement.
        if (!drawRange.Available)
            return "the engine's culler could not be read, so what it is drawing is unknown; "
                + ExactRangeState();

        int cells = 0, engineDraws = 0, covered = 0;
        int missAir = 0, missPending = 0, missRange = 0, missGeometry = 0, missOther = 0;
        int overOwned = 0;
        double nearestMiss = -1;
        string nearestMissText = "none";

        // Bounded to the columns the engine could possibly be drawing, plus a chunk of margin,
        // rather than the whole tracked window. The window is sized from the view distance plus
        // guard chunks and grows as its square, so at a 256-block view distance an unbounded
        // audit is fourteen thousand chunk lookups - each taking the client's chunk lock (G45) -
        // for an answer that is entirely decided inside the draw radius. This keeps the cost
        // roughly constant instead of quartic in the view distance.
        int reachChunks = (int)Math.Ceiling(
            (drawRange.Available ? drawRange.WidestDrawBoundBlocks() : ApprovedViewDistance())
            / VanillaReadinessMask.ChunkBlocks) + 1;
        int camChunkX = VanillaRenderReadiness.ChunkCoordinate(camPos.X);
        int camChunkZ = VanillaRenderReadiness.ChunkCoordinate(camPos.Z);

        int minX = Math.Max(readiness.ActiveMinChunkX, camChunkX - reachChunks);
        int minZ = Math.Max(readiness.ActiveMinChunkZ, camChunkZ - reachChunks);
        int maxX = Math.Min(readiness.ActiveMinChunkX + readiness.ActiveWidth - 1, camChunkX + reachChunks);
        int maxZ = Math.Min(readiness.ActiveMinChunkZ + readiness.ActiveDepth - 1, camChunkZ + reachChunks);

        for (int z = minZ; z <= maxZ; z++)
        for (int x = minX; x <= maxX; x++)
        for (int y = 0; y < readiness.VerticalChunks; y++)
        {
            var cell = new VanillaChunkCell(x, y, z);
            cells++;

            IWorldChunk? chunk;
            try { chunk = capi.World.BlockAccessor.GetChunk(cell.X, cell.Y, cell.Z); }
            catch { continue; }
            if (chunk == null || chunk.Empty) continue;

            // The engine's own predicate, in full: it holds unhidden mesh, the ray culler
            // reached it, and the range clause admits it. Anything short of all three and the
            // engine submits nothing, so cached terrain there is not an overlap.
            if (!VanillaChunkGeometry.TryIsVanillaDrawing(chunk, out bool drawing) || !drawing) continue;
            if (!VanillaChunkGeometry.TryReadDrawBounds(chunk, out float cx, out float cz, out int lods))
                continue;
            if (!drawRange.Available) continue;
            if (!drawRange.DrawsWithMargin(cx, cz, lods, OverlapMarginBlocks)) continue;

            engineDraws++;
            if (readinessMask.IsReady(cell)) { covered++; continue; }

            // Uncovered. Which rule let it through?
            VanillaReadinessState state = readiness.State(cell);
            if (state == VanillaReadinessState.VanillaReady && readiness.IsMaskExcluded(cell)) missAir++;
            else if (state != VanillaReadinessState.VanillaReady)
            {
                if (!VanillaChunkHoldsDrawableGround(chunk)) missGeometry++;
                // The gate that would ACTUALLY run for this cell, in its current state - not
                // whichever one is convenient to evaluate. Two earlier versions of this line got
                // it wrong in opposite directions: the first was dead code that filed everything
                // under "not committed yet", and the second asked the bounded rule about cells
                // the bounded rule no longer decides. A reason bucket that names a gate the cell
                // never passes through is worse than no bucket, because it is believed.
                else if (ExactDrawRange && drawRange.Available
                    && drawRange.ColumnCertainlyNotDrawn(cell.X, cell.Z, OverlapMarginBlocks))
                    missRange++;
                else if (!ExactDrawRange && VanillaRenderReadiness.BeyondVanillaDrawRange(
                    cell.X, cell.Z, camPos.X, camPos.Z, ApprovedViewDistance(), OverlapMarginBlocks))
                    missRange++;
                else missPending++;
            }
            else missOther++;

            double distance = VanillaRenderReadiness.ColumnDistanceBlocksFrom(
                cell.X, cell.Z, camPos.X, camPos.Z);
            if (nearestMiss < 0 || distance < nearestMiss)
            {
                nearestMiss = distance;
                nearestMissText = $"chunk {cell.X},{cell.Y},{cell.Z} at {distance:0} blocks "
                    + $"(we say {state}{(state == VanillaReadinessState.VanillaReady && readiness.IsMaskExcluded(cell) ? ", air-excluded" : "")}) "
                    + $"| {VanillaChunkGeometry.DescribeSignals(chunk)}";
            }
        }

        // The other direction, which is a hole rather than an overlap: we hide cached terrain
        // where the engine is drawing nothing. Counted from the mask outward so a stale texel
        // cannot hide behind the uncovered count.
        // The over-owned pass keeps the FULL window: a stale texel that nothing draws is a hole,
        // and the place it is most likely to survive is exactly the ground the camera moved away
        // from - outside the radius the pass above bounds itself to.
        for (int z = 0; z < readiness.ActiveDepth; z++)
        for (int x = 0; x < readiness.ActiveWidth; x++)
        for (int y = 0; y < readiness.VerticalChunks; y++)
        {
            var cell = new VanillaChunkCell(
                readiness.ActiveMinChunkX + x, y, readiness.ActiveMinChunkZ + z);
            if (!readinessMask.IsReady(cell)) continue;
            IWorldChunk? chunk;
            try { chunk = capi.World.BlockAccessor.GetChunk(cell.X, cell.Y, cell.Z); }
            catch { continue; }
            bool draws = chunk is { Empty: false }
                && VanillaChunkGeometry.TryIsVanillaDrawing(chunk, out bool d) && d
                && VanillaChunkGeometry.TryReadDrawBounds(chunk, out float sx, out float sz, out int sl)
                && drawRange.Available
                && drawRange.DrawsWithMargin(sx, sz, sl, OverlapMarginBlocks);
            if (!draws) overOwned++;
        }

        int uncovered = engineDraws - covered;
        double coverage = engineDraws == 0 ? 0 : 100.0 * covered / engineDraws;

        var report = new StringBuilder();
        report.Append("[Landscapes] coverage audit: of ").Append(cells)
            .Append(" cells inside the draw radius the engine is drawing ").Append(engineDraws)
            .Append("; the mask covers ").Append(covered).Append(" of them (")
            .Append($"{coverage:0}").Append("%), leaving ").Append(uncovered)
            .Append(" chunks where the engine draws and cached terrain is NOT hidden. ")
            .Append("Reasons: ").Append(missAir).Append(" air-excluded/")
            .Append(missPending).Append(" not committed yet/")
            .Append(missRange).Append(" refused for range/")
            .Append(missGeometry).Append(" refused for no geometry/")
            .Append(missOther).Append(" other. ")
            .Append(overOwned).Append(" cells are hidden where the engine draws nothing (holes). ")
            .Append("Nearest uncovered: ").Append(nearestMissText);

        capi.Logger.Notification(report.ToString());
        return $"coverage audit written to the client log: the engine draws {engineDraws} chunks, "
            + $"the mask covers {covered} ({coverage:0}%), {uncovered} uncovered "
            + $"({missAir} air/{missPending} uncommitted/{missGeometry} no-geometry), "
            + $"{overOwned} hidden with nothing drawing.";
    }

    /// <summary>
    /// How much cached geometry stands ABOVE the highest cell that is allowed to hide it, in
    /// columns vanilla is drawing.
    ///
    /// <para>
    /// This separates the two remaining explanations for cached terrain visibly overlapping the
    /// game's own, and they want opposite fixes. Either the mask is not owning the ground
    /// (ownership is too sparse, and the answer is more ownership), or it owns the ground and
    /// the cached surface simply sticks up past it into the air cell above - which is excluded
    /// from the mask on purpose, because cached terrain overshoots the real surface and
    /// suppressing air shears the top off the horizon (G40). The second is a deliberate trade
    /// and no amount of extra ownership touches it.
    /// </para>
    /// <para>
    /// Measured per section against the top of its own footprint's suppressing band, so a
    /// column vanilla is not drawing at all contributes nothing either way.
    /// </para>
    /// </summary>
    string DescribeGeometryOvershoot()
    {
        if (readiness == null) return string.Empty;

        int overColumns = 0, coveredColumns = 0, sectionsOvershooting = 0;
        int worstBands = 0;
        double worstDistance = -1;

        foreach (KeyValuePair<long, LodLiveMeshStats> entry in liveMeshStats)
        {
            LodHeightSpan span = entry.Value.Heights.Either;
            if (!span.HasGeometry) continue;

            long key = entry.Key;
            int level = LodWorld.KeyLevel(key);
            if (level < 0 || level > LodWorld.MaxLevel) continue;

            // L0 only, and the first version of this without that line reported 63 columns
            // where 9 exist - every column counted once per level that covers it, seven times
            // over. Worse than the arithmetic: a coarse section's height span is the maximum
            // over its whole footprint, four thousand blocks across at L6, so comparing it to
            // one column's ground measures the terrain's slope and calls it overshoot. Only the
            // finest level is narrow enough for the comparison to mean anything, and even at 64
            // blocks it is an upper bound rather than a measurement.
            if (level != 0) continue;

            int edgeCells = 2 << level;
            int baseX = LodWorld.KeySx(key) * edgeCells;
            int baseZ = LodWorld.KeySz(key) * edgeCells;
            int topGeometryY = (int)Math.Floor(span.MaxY / VanillaReadinessMask.ChunkBlocks);
            bool sectionOvershoots = false;

            for (int dz = 0; dz < edgeCells; dz++)
            for (int dx = 0; dx < edgeCells; dx++)
            {
                int chunkX = baseX + dx;
                int chunkZ = baseZ + dz;
                int suppressingTop = readiness.HighestSuppressingY(chunkX, chunkZ);
                // A column vanilla is not drawing has nothing to overlap, so it is neither a
                // covered column nor an overshoot - counting it either way would make a scene
                // with no ownership look like a scene full of overshoot.
                if (suppressingTop < 0) continue;

                coveredColumns++;
                if (topGeometryY <= suppressingTop) continue;

                overColumns++;
                sectionOvershoots = true;
                int bands = topGeometryY - suppressingTop;
                if (bands > worstBands) worstBands = bands;
                double distance = VanillaRenderReadiness.ColumnDistanceBlocksFrom(
                    chunkX, chunkZ, camPos.X, camPos.Z);
                if (worstDistance < 0 || distance < worstDistance) worstDistance = distance;
            }

            if (sectionOvershoots) sectionsOvershooting++;
        }

        if (coveredColumns == 0)
        {
            return "Overshoot: nothing to measure - not one column under a cached mesh has a cell "
                + "that is allowed to hide anything, so every overlap you can see is missing "
                + "ownership rather than cached terrain standing too tall.";
        }

        return $"Overshoot (L0 only, upper bound): {overColumns} of {coveredColumns} columns that "
            + $"vanilla owns carry cached geometry ABOVE the highest cell allowed to hide it, across "
            + $"{sectionsOvershooting} sections, worst {worstBands} cell(s) over, nearest at "
            + $"{worstDistance:0} blocks. "
            + (overColumns == 0
                ? "So where vanilla owns the ground, the cached surface stays under it: any overlap "
                    + "you can see is in columns vanilla does not own."
                : "That geometry sits in air cells, which are excluded from the mask on purpose "
                    + "(G40), so it is drawn whatever ownership says.");
    }

    /// <summary>
    /// The exact draw-range rule in one clause of the periodic line.
    ///
    /// <para>
    /// The figure that matters is the exact/bounded split. Nothing on screen distinguishes
    /// "the sphere is being read" from "the sphere could not be read and every cell fell back
    /// to the sphere-slack ring" - both draw terrain, one just overlaps more - so if this rule
    /// ever goes quietly unavailable, this clause is the only place it shows.
    /// </para>
    /// </summary>
    public string DescribeExactRange() => "draw range: " + ExactRangeState();

    /// <summary>
    /// Deliberately comma-free. `scripts/bench-windows.ps1` parses the readiness line clause by
    /// clause on commas, and a comma inside one clause silently ends the parse there - so every
    /// separator here is a slash and the player position is written `x/z`. A static check holds
    /// the runner's pattern against this line, and it fails on exactly that mistake.
    /// </summary>
    string ExactRangeState()
    {
        if (!ExactDrawRange) return $"bounded by request (ring {RefusedOwnershipRingBlocks():0} blocks)";
        if (!VanillaChunkGeometry.Available)
            return $"BOUNDED - mesh binding unavailable (ring {RefusedOwnershipRingBlocks():0} blocks)";
        if (!drawRange.Available)
            return $"BOUNDED - engine culler unreadable (ring {RefusedOwnershipRingBlocks():0} blocks)";

        long decisions = ReadinessExactRangeDecisions + ReadinessBoundedRangeDecisions;
        double exactShare = decisions == 0 ? 0 : 100.0 * ReadinessExactRangeDecisions / decisions;
        double bound = Math.Sqrt(drawRange.ViewDistanceSq) - Math.Max(0f, OverlapMarginBlocks);
        return $"exact to {bound:0} blocks from {drawRange.PlayerBlockX}/{drawRange.PlayerBlockZ}"
            + $" -- {ReadinessExactRangeDecisions} exact/{ReadinessBoundedRangeDecisions} bounded"
            + $" ({exactShare:0}% exact)/{readiness?.CellsWithDrawSphere() ?? 0} cells measured/"
            + $"{ReadinessRangeSweeps} sweeps/{ReadinessRangeSweepLosses} losses/"
            + $"peak {ReadinessRangeSweepPeakMs:0.000} ms";
    }

    /// <summary>
    /// The blend band in one clause of the periodic line, and the body of `.dlband`.
    /// Build and upload cost are included because "nearly free" is this design's claim and
    /// an unmeasured claim is not one.
    /// </summary>
    string BandState()
    {
        if (!blendPipelineEnabled) return "master off";
        if (!BandEnabled) return "off";
        if (seamFieldFailed) return "failed";
        if (seamField == null || !seamFieldActive) return "pending";

        return $"{BandWidthBlocks:0}-block reach/{BandStepBlocks:0}-block steps/"
            + $"inner {BandInnerBlocks:0} {(BandInnerDerived ? "derived" : "pinned")}/strength {BandStrength:0.00}/"
            + $"speckle {BandSpeckle:0.00}/darken {BandDarken:0.00}/saturate {BandSaturate:0.00}/"
            + $"sink {BandSinkVoxels}-16ths/overlap +{OverlapMarginBlocks:0}/"
            + $"{seamField.WidthCells}x{seamField.DepthCells} cells in {seamField.Capacity}/"
            + $"{seamField.Bytes / 1024} KiB/{seamField.Builds} builds/{seamField.Uploads} uploads/"
            + $"last build {seamFieldBuildUs}us/upload {seamFieldUploadUs}us"
            + (BandDebugPaint ? "/SHAPE VIEW ON" : "");
    }

    /// <summary>
    /// The block texture system in one clause. The refusal counts matter: a block with no
    /// texture, or one the game put on a second atlas page, keeps its averaged colour, and
    /// the only way to know how much ground that covers is to count it.
    /// </summary>
    string BlockTextureState()
    {
        if (!blendPipelineEnabled) return "master off";
        if (!BlockTexturesEnabled) return "off";
        if (blockTextures == null) return "unavailable";
        if (blockTableFailed) return "failed";
        if (!blockTexturesLive) return "pending";
        return $"strength {BlockTextureStrength:0.00}/mips {BlockTextureMipLimit:0.0}/"
            + $"detail {BlockTextureDetail:0.0}/" + blockTextures.Describe();
    }

    /// <summary>
    /// Answers `.dltexture`. Logged as well as returned, because game chat cannot be copied
    /// out and the refusal counts are the part worth keeping.
    /// </summary>
    public string DescribeBlockTextures()
    {
        string state = BlockTextureState();
        capi.Logger.Notification("[Landscapes] block textures {0}", state);
        return "block textures " + state + ".";
    }

    /// <summary>
    /// Answers `.dlband` with no argument. Written to the log as well as returned, because
    /// game chat cannot be copied out and this line is the one worth keeping after a
    /// tuning pass.
    /// </summary>
    public string DescribeBlendBand()
    {
        string state = BandState();
        capi.Logger.Notification("[Landscapes] band {0}", state);
        return "blending band " + state + ".";
    }

    /// <summary>
    /// Writes the live seam field to the client log as an ASCII map of the weight the shader
    /// would compute, plus the weight the field's own outer edge ends on.
    ///
    /// The band's shape was judged from a screenshot for three builds and the screenshot could
    /// not answer the question. `.dlband shape on` paints the weight, but a ramp seen at a
    /// shallow angle from altitude is a few pixels tall, so a gradient and a cliff look alike -
    /// and the number that decides which it is, the weight where the field's window stops, is
    /// not on screen at all. This puts both in the log, where they can be read exactly.
    ///
    /// Deliberately a whole-map dump on demand rather than a line in the periodic report: the
    /// question is what the SHAPE is, and no summary statistic answers that.
    /// </summary>
    public string LogBlendBandMap()
    {
        if (seamField == null || !seamFieldActive)
            return "no blend band to map: the seam field is not live. " + BandState();

        int stride = Math.Max(1, seamField.WidthCells / 96);
        float width = Math.Max(1f, BandWidthBlocks);
        float step = Math.Max(VanillaSeamDistanceField.CellBlocks, BandStepBlocks);
        float inner = Math.Max(0f, BandInnerBlocks);

        float WeightAt(int cellX, int cellZ)
        {
            int index = VanillaSeamDistanceField.TexelIndex(cellX, cellZ, seamField.Capacity);
            float cells = (seamField.Texels[index] >> 16) & 0xFF;
            float blocks = Math.Max(0f, cells * VanillaSeamDistanceField.CellBlocks - inner);
            return Math.Clamp(1f - MathF.Floor(blocks / step) * step / width, 0f, 1f);
        }

        // Ownership first, because the band is derived from it and a wrong band with a right
        // ownership map is a different bug from a wrong band with a wrong one. Offline
        // reproduction of this exact configuration produces a clean ring with a full gradient,
        // so if the shape on screen is not that, the difference is in this map and nowhere
        // else. Printed per chunk column - "any cell owned", the same question the field asks.
        var owned = new StringBuilder();
        if (readiness != null && readinessMask != null)
        {
            int chunkStride = Math.Max(1, readiness.ActiveWidth / 96);
            owned.Append("[Landscapes] vanilla ownership footprint: ")
                .Append(readiness.ActiveWidth).Append('x').Append(readiness.ActiveDepth)
                .Append(" chunk columns (").Append(readiness.ActiveWidth * 32)
                .Append(" blocks across), ").Append(readinessMask.ReadyTexels)
                .Append(" cells owned. ")
                // Which boundary the footprint below actually has, because the band is
                // measured from it and the two rules put it up to 62 blocks apart. Reading
                // this map against the wrong one is how the band's circularity was
                // mis-attributed twice.
                .Append(RefusedOwnershipRingBlocks() <= 0f
                    ? "Ownership follows vanilla's own chunk footprint exactly (.dlexact on), "
                        + "so this IS where vanilla's terrain ends. "
                    : $"Ownership is denied past "
                        + $"{Math.Max(0f, ApprovedViewDistance() - RefusedOwnershipRingBlocks()):0}"
                        + " blocks by the draw-range rule, so the band's inner edge is that "
                        + "boundary and not vanilla's real chunk footprint. ")
                .Append("'O' owned, '.' not. '+' is the camera's own column.");
            int camChunkX = VanillaRenderReadiness.ChunkCoordinate(camPos.X);
            int camChunkZ = VanillaRenderReadiness.ChunkCoordinate(camPos.Z);
            for (int z = 0; z < readiness.ActiveDepth; z += chunkStride)
            {
                owned.Append('\n').Append("    ");
                for (int x = 0; x < readiness.ActiveWidth; x += chunkStride)
                {
                    int chunkX = readiness.ActiveMinChunkX + x;
                    int chunkZ = readiness.ActiveMinChunkZ + z;
                    owned.Append(chunkX == camChunkX && chunkZ == camChunkZ ? '+'
                        : readinessMask.ColumnHasReady(chunkX, chunkZ) ? 'O' : '.');
                }
            }
            capi.Logger.Notification(owned.ToString());
        }

        var report = new StringBuilder();
        report.Append("[Landscapes] blend band map: ")
            .Append(seamField.WidthCells).Append('x').Append(seamField.DepthCells)
            .Append(" cells of 8 blocks (")
            .Append(seamField.WidthCells * VanillaSeamDistanceField.CellBlocks)
            .Append(" blocks across), margin ").Append(SeamFieldMarginCells())
            .Append(" cells, reach ").Append($"{inner:0}+{width:0}")
            .Append(" blocks, mask ").Append(readinessMask?.ReadyTexels ?? 0).Append(" ground cells owned, ")
            .Append(readinessMask?.WaterColumns ?? 0).Append(" columns in water's own footprint (")
            .Append(WaterFootprintDilationChunks()).Append(" chunks wider). ")
            .Append("'.' none, 1-9 partial, '#' full.");

        // How the ramp ends. A field whose outermost ring still carries weight is a field that
        // stops mid-ramp, and its window edge is a straight line across the world.
        float edgeMax = 0f;
        for (int x = 0; x < seamField.WidthCells; x++)
        {
            edgeMax = Math.Max(edgeMax, WeightAt(seamField.MinCellX + x, seamField.MinCellZ));
            edgeMax = Math.Max(edgeMax, WeightAt(
                seamField.MinCellX + x, seamField.MinCellZ + seamField.DepthCells - 1));
        }
        for (int z = 0; z < seamField.DepthCells; z++)
        {
            edgeMax = Math.Max(edgeMax, WeightAt(seamField.MinCellX, seamField.MinCellZ + z));
            edgeMax = Math.Max(edgeMax, WeightAt(
                seamField.MinCellX + seamField.WidthCells - 1, seamField.MinCellZ + z));
        }
        report.Append(" Outer edge ends at weight ").Append($"{edgeMax:0.00}")
            .Append(edgeMax > 0.02f
                ? " - the ramp is CUT, and that edge is a straight line on screen."
                : " - the ramp completes inside the window.");

        for (int z = 0; z < seamField.DepthCells; z += stride)
        {
            report.Append('\n').Append("    ");
            for (int x = 0; x < seamField.WidthCells; x += stride)
            {
                float weight = WeightAt(seamField.MinCellX + x, seamField.MinCellZ + z);
                report.Append(weight <= 0f ? '.'
                    : weight >= 1f ? '#'
                    : (char)('0' + (int)(weight * 9f)));
            }
        }

        capi.Logger.Notification(report.ToString());
        return $"blend band map written to the client log: {seamField.WidthCells} cells across, "
            + $"outer edge ends at weight {edgeMax:0.00}"
            + (edgeMax > 0.02f ? " (the ramp is cut - that is a line on screen)." : ".");
    }

    /// <summary>
    /// Times the whole frame around the real work. The body below has several early
    /// returns, and every one of them still has to close the frame, or the mod's share
    /// would be recorded only for the frames where it did the most.
    /// </summary>
    public void OnRenderFrame(float deltaTime, EnumRenderStage stage)
    {
        // Normally prepared before vanilla's liquid-depth pass. A missing Before callback
        // can still draw normally, without extending liquid depth for that frame.
        if (!beforeStageRan) PrepareBeforeFrame(deltaTime);
        long drawStart = Stopwatch.GetTimestamp();
        LodHitchMarks.Scope range = LodHitchMarks.Enter(HitchBeforeLod, HitchLod);
        try
        {
            RenderFrame(deltaTime, stage);
            DrawSilhouettes();
        }
        finally
        {
            range.Leave();
            beforeStageTicks += Stopwatch.GetTimestamp() - drawStart;
            waterFrame.FinishOpaque(frameCounter, currentWorldEpoch());
            beforeStageRan = false;
        }
    }

    const string HitchBeforeLod = "before vh:lod";
    const string HitchLod = "vh:lod";

    /// <summary>Per-frame measurement of the client's own frame interval against ours.</summary>
    public LodFrameTimeline FrameTimeline => frameTimeline;

    readonly LodFrameTimeline frameTimeline = new();

    void PrepareRenderFrame(float deltaTime)
    {
        waterFrame.Invalidate();
        drawList.Clear();
        standInQuarters.Clear();
        if (AutoUnpause && capi.IsGamePaused) capi.PauseGame(false);
        if (ContinueLoadingWhilePaused && capi.IsGamePaused) PausedLoadingPump?.Invoke();

        var rapi = capi.Render;
        gpuTelemetry.BeginFrame();
        if (rapi.FrameWidth == 0) return;

        camPos = capi.World.Player.Entity.CameraPos;
        frameCounter++;
        ConfigureRenderPaths();
        // Before anything reads a threshold: radial demand, residency and traversal all do.
        UpdateAutoThresholds();
        ApplyGpuShadowRequest();
        ApplyGpuDrawFailureInjection();
        float viewDistance = ApprovedViewDistance();

        // Fixed before anything reads it, and before the occlusion queries are resolved:
        // the two paths disagree about whether per-section queries are issued at all, so
        // the choice must not change between resolving a query and acting on it.
        // The packed renderer is the product path under development. Expanded indirect and
        // per-section MeshRef drawing are retained only as source scaffolding while their
        // resource ownership is removed; neither is allowed to become a visible fallback.
        bool packedWanted = PackedDrawAvailable
            && (!ClusterDrawEnabled || ClusterDrawAvailable);
        bool clustersWanted = packedWanted && ClusterDrawEnabled;
        bool indirectWanted = packedWanted;
        if (indirectWanted != indirectDrawingThisFrame)
        {
            // Entering the batched path also gives up the query objects themselves. They are
            // allocated lazily, one GL query per section that was ever tested, and while
            // batching is on not one of them is issued or read - so they are pure retention:
            // driver objects, plus a dictionary the per-tick eviction sweep still walks. They
            // come back lazily if suppression ever returns to the CPU.
            //
            // Released BEFORE the invalidation below, because the release clears the pending
            // list too and there is nothing left to mark stale afterwards.
            if (indirectWanted) DisposeTemporalOcclusionQueries();

            // Every previous-frame occlusion answer was taken under the other path, and
            // under batching no new ones are taken at all. Keeping them across the switch
            // would let a section that was hidden several seconds and one camera move ago
            // stay hidden after switching back - which would look like the fast path
            // losing terrain, in the exact A/B this switch exists for.
            InvalidateTemporalOcclusionScene();
        }

        if (packedShaderOk && !reportedPackedShaderOk)
        {
            reportedPackedShaderOk = true;
            capi.Logger.Notification(
                "[Landscapes] the packed lodterrain variant compiled after {0} "
                + "attempt(s); 12-byte quad drawing is available",
                shaderLoadAttempts);
        }
        indirectDrawingThisFrame = indirectWanted;
        packedDrawingThisFrame = packedWanted;
        clusterDrawingThisFrame = clustersWanted;
        depthSplitThisFrame = LateDepthPyramid
            && GpuCullEnabled
            && DepthPyramidEnabled
            && indirectDrawingThisFrame
            && ActiveFarBuilder != null;

        LodPhaseStart phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        UpdateReadinessShadow(viewDistance);
        // Before any draw: a quad can carry a block slot instead of a colour, and a slot the
        // table does not describe yet would draw as black ground.
        PublishBlockTextureTable();
        ReadinessCost.Add(phaseStart);

        if (prog == null || !shaderOk || prog.LoadError) return;

        // Demand exists independently of draw visibility. This runs before the empty-mesh
        // return below, breaking the old no-mesh -> no-walk -> no-request join cycle.
        PlanRadialDemand();

        // Timed apart: pruning/index refresh is normally incremental but deliberately
        // rebuilds when the camera crosses a coarse cell; scheduling then consumes a
        // bounded nearest-first queue without scanning the complete dirty set.
        phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        PruneRenderDirty();
        PruneCost.Add(phaseStart);

        phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        ScheduleMeshJobs();
        ScheduleCost.Add(phaseStart);

        phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        UploadFinishedMeshes();
        UploadCost.Add(phaseStart);

        phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        EvictStaleMeshes();
        EvictCost.Add(phaseStart);

        phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        RefreshSeasonalState();
        SeasonalCost.Add(phaseStart);
        if (sectionMeshes.Count == 0 && waterMeshes.Count == 0)
        {
            // Nothing to draw, so traversal is skipped. Radial demand and scheduling have
            // already run above; this count now distinguishes a bounded warm-up from an
            // unexpected failure to produce the first mesh.
            FramesWithoutMeshes++;
            return;
        }

        phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        UpdateEffectiveFarDistance(viewDistance);
        FarDistanceCost.Add(phaseStart);
        ApplyZFar();

        // The traversal and draw cull use the exact matrices handed to the shader. This
        // must happen after ApplyZFar, because that can rebuild the projection matrix.
        worldHeight = capi.World.BlockAccessor.MapSizeY;
        frustum.Update(rapi.CurrentProjectionMatrix, rapi.CameraMatrixOriginf);
        UpdateTemporalOcclusionView(rapi.CameraMatrixOriginf, rapi.CurrentProjectionMatrix,
            camPos.X, camPos.Y, camPos.Z);
        ResolveTemporalOcclusionQueries();
        PrepareTemporalOcclusionQueries();

        traversalCulledThisFrame = 0;
        subtreeCulledThisFrame = 0;
        subtreeCulledMeshesThisFrame = 0;
        culledThisFrame = 0;
        verticalCulledThisFrame = 0;
        LastTemporalOcclusionDrawsSkipped = 0;
        LastTemporalOcclusionSeamDraws = 0;
        LastTemporalOcclusionEdgeDraws = 0;

        phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);
        drawList.Clear();
        standInQuarters.Clear();
        foreach (long top in world.TopLevelKeys) CollectDrawNodes(top);
        WalkCost.Add(phaseStart);
        LastDrawCount = drawList.Count;
        LastTraversalCulledCount = traversalCulledThisFrame;
        LastSubtreeCulledCount = subtreeCulledThisFrame;
        SubtreeCulledNodes += subtreeCulledThisFrame;
        SubtreeCulledMeshes += subtreeCulledMeshesThisFrame;
        if (subtreeCulledThisFrame > SubtreeCulledMax)
            SubtreeCulledMax = subtreeCulledThisFrame;
        if (drawList.Count == 0)
        {
            LastCulledCount = 0;
            LastVerticalCulledCount = 0;
            return;
        }

        PrepareWaterSurfaceStage();
    }

    void RenderFrame(float deltaTime, EnumRenderStage stage)
    {
        if (!preparedFrameReady || prog == null) return;
        var rapi = capi.Render;
        float viewDistance = ApprovedViewDistance();
        LodPhaseStart phaseStart;
        // Distant anti-aliasing freezes the depth buffer before a single cached draw, so the
        // resolve at the end of this callback can tell exactly which pixels cached terrain drew.
        BeginTaaFrame();
        // The water capture already separated opaque ground from cached water, but that
        // still left two possible authors for a band in the opaque image: vanilla terrain
        // that was present before this callback, and cached ground added below. Freeze both
        // colour and depth here, before either the HZB copy or a cached draw, so one requested
        // frame can attribute that difference without relying on two screenshots or a moving
        // water surface.
        CaptureWaterImage("02b-before-cached-ground-depth", EnumFrameBuffer.Primary, depth: true);
        CaptureWaterImage("02c-before-cached-ground-color", EnumFrameBuffer.Primary, depth: false);
        // Here, and not later, because "after vanilla terrain" is what makes the depth
        // buffer worth copying: this renderer runs in the opaque stage after the engine has
        // drawn its own world, so what sits in the depth attachment right now is exactly
        // the set of near occluders a distant section would have to be behind. Copied
        // before we draw a single cached section, so cached terrain cannot occlude itself
        // by accident - that is Phase 6 and it is a separate decision.
        // The split still needs this first picture for its near bucket. A second build after
        // the near draw adds cached occluders for the far bucket; both are same-frame.
        BuildDepthPyramid();
        hzbBoxes.Clear();
        hzbCollecting = DepthPyramidEnabled;

        prog.Use();
        if (OpaqueBackfaceCulling)
        {
            // The public API exposes the enable switch but not the concrete client's
            // GlCullFaceBack helper. The opaque stage owns the ordinary back-face mode;
            // use that state rather than binding to an internal render implementation.
            rapi.GlEnableCullFace();
        }
        else
        {
            rapi.GlDisableCullFace();
        }

        ApplyFrameUniforms(prog, viewDistance);

        float cullDistSq = float.MaxValue;
        if (FarViewDistanceCap > 0)
        {
            float cull = FarViewDistanceCap + LodSection.SectionBlocks;
            cullDistSq = cull * cull;
        }

        renderPaths.PrepareFrame(new LodRenderFrame(
            currentWorldEpoch(), frameCounter, drawList.Count, cullDistSq));

        phaseStart = LodPhaseCost.Start(TrackPhaseAllocations);

        skippedLastFrame.Clear();

        // Pass 1: opaque terrain. The experimental order computes distance once per
        // selected section and reuses list capacity; sorting is included in DrawCost so
        // its CPU price stays visible beside any GPU-side gain. The GPU timer ends after
        // command submission and is read only after a later frame reports it available.
        renderPaths.DrawOpaque();

        // `.dlsky`: count sky still showing below the horizon now that all opaque terrain,
        // vanilla's and ours, is in the depth buffer. Off by default; draws nothing visible.
        RunSkyWitness();

        CaptureWaterImage("03-ground-depth", EnumFrameBuffer.Primary, depth: true);
        CaptureWaterImage("04-before-cached-water", EnumFrameBuffer.Primary, depth: false);

        LastCulledCount = culledThisFrame; // opaque pass only: water covers a subset

        // Pass 2: water and thin cover remain two-sided. Water must be visible from below,
        // and a plant mat has no opposite face to take over when the camera is underneath.
        renderPaths.DrawWater();
        CaptureWaterImage(waterFrame.UseOit ? "05-opaque-stage-end" : "05-after-cached-water",
            EnumFrameBuffer.Primary, depth: false);

        // The split's mid-frame picture deliberately excludes water. It excluded it because
        // water was blended and still wrote depth, so treating it as an occluder could remove
        // visible ground beneath it; the water pass no longer writes depth at all, so it can
        // no longer occlude anything and the exclusion is now simply correct by construction.
        // Classification reads the unchanged private pyramid, not the live depth attachment.
        hzbCollecting = false;
        ClassifyAgainstDepthPyramid();

        // Both passes cull, so this is summed after the second one rather than beside
        // LastCulledCount, which is deliberately opaque-only.
        LastVerticalCulledCount = verticalCulledThisFrame;
        VerticalCulledSections += verticalCulledThisFrame;
        if (verticalCulledThisFrame > VerticalCulledMax)
            VerticalCulledMax = verticalCulledThisFrame;

        // Submission only. RenderMesh queues work for the GPU and returns, so this
        // measures the CPU cost of the draw loop -- the uniform uploads, the culling and
        // the dictionary probes -- and not what the GPU then does with it.
        DrawCost.Add(phaseStart);

        rapi.GlEnableCullFace();
        prog.Stop();

        // After thin cover as well as opaque ground: both are drawn by the terrain program, so
        // both were nudged, and both have to be blended back or they would wobble.
        ResolveTaa();
    }

    /// <summary>
    /// The texture units this renderer owns, named rather than written as literals in ten
    /// places. Two separate defects have already come from a unit number: the engine rebinds
    /// its own samplers on every Use() and can land on one of these (G130), and the upload path
    /// below binds to whichever unit is current. Both are answered by one place saying which
    /// unit a texture belongs to.
    ///
    /// The engine's automatic bindings are explicitly assigned by LodTerrainSamplerLayout;
    /// their declaration-order defaults do NOT reserve the low units for shared includes.
    /// </summary>
    const int ReadinessMaskUnit = LodTerrainSamplerLayout.ReadinessMask;
    const int SeamFieldUnit = LodTerrainSamplerLayout.SeamField;
    const int BlockTableUnit = LodTerrainSamplerLayout.BlockTable;
    const int BlockAtlasUnit = LodTerrainSamplerLayout.BlockAtlas;
    const int SlotTableUnit = LodTerrainSamplerLayout.SlotTable;
    const int ClimateFieldUnit = LodTerrainSamplerLayout.ClimateField;
    const int ClimateMapsUnit = LodTerrainSamplerLayout.ClimateMaps;
    const int SeasonColumnUnit = LodTerrainSamplerLayout.SeasonColumn;

    /// <summary>
    /// Upload one of this renderer's own CPU-built tables, with its destination unit made
    /// current first.
    ///
    /// `LoadOrUpdateTextureFromBgra` reaches `GL.BindTexture` with no `GL.ActiveTexture` of its
    /// own and never puts back what it displaced, so the texture it writes is left bound to
    /// whichever unit happened to be current. Called from the middle of a binding sequence,
    /// that quietly takes a unit away from the sampler just bound to it: the tint ramp's upload
    /// sat one line after the block atlas was bound to unit 9, so every fragment INSIDE the
    /// blend band - the only place the atlas is sampled - drew the 64x96 tint ramp as its block
    /// texture and came out several times too bright, while all the terrain outside the band
    /// was perfect. Making the destination current turns that stray bind into the bind that was
    /// wanted anyway.
    ///
    /// Nearest and clamp for all three: each is read with texelFetch, which ignores filtering,
    /// and none of them may wrap.
    /// </summary>
    void UploadOwnTexture(int[] texels, int unit, ref LoadedTexture texture)
    {
        GL.ActiveTexture(TextureUnit.Texture0 + unit);
        capi.Render.LoadOrUpdateTextureFromBgra(texels, linearMag: false, clampMode: 0,
            ref texture);
    }

    /// <summary>
    /// Everything the shader needs that is the same for every section this frame.
    ///
    /// Taken as a parameter rather than read from the field because the indirect path
    /// draws through a second program built from the same source, and both must be handed
    /// identical values: any divergence here would show up as a pixel difference and be
    /// read as a bug in the fast path rather than as a missing upload.
    /// </summary>
    void ApplyFrameUniforms(IShaderProgram prog, float viewDistance)
    {
        var rapi = capi.Render;
        prog.UniformMatrix("viewMatrix", rapi.CameraMatrixOriginf);
        prog.UniformMatrix("projectionMatrix", rapi.CurrentProjectionMatrix);

        prog.Uniform("sunPosition", capi.World.Calendar.SunPositionNormalized);
        prog.Uniform("sunColor", capi.World.Calendar.SunColor);
        prog.Uniform("dayLight", Math.Max(0, capi.World.Calendar.DayLightStrength));

        prog.Uniform("rgbaFogIn", capi.Ambient.BlendedFogColor);
        prog.Uniform("fogDensityIn", capi.Ambient.BlendedFogDensity);
        prog.Uniform("fogMinIn", capi.Ambient.BlendedFogMin);
        prog.Uniform("horizonFog", capi.Ambient.BlendedCloudDensity);

        prog.Uniform("viewDistance", viewDistance);
        prog.Uniform("farViewDistance", EffectiveFarDistance);
        // The uniform exists in the shared body but only the packed decoder reads it.
        // Keeping it tied to the established switch makes `.dlbackface off` disable both
        // the early exact rejection and OpenGL's later winding rejection together.
        prog.Uniform("packedBackfaceCull", OpaqueBackfaceCulling ? 1 : 0);
        // Distant anti-aliasing's nudge. Declared in the shared vertex inputs, because the water
        // programs receive this same upload and the engine throws on a name a program never
        // declared; only the terrain body reads them. A zero scale is "off" in the shader. The
        // nudge's ramp is the second half of the resolve's, both from the one reach setting, so
        // geometry only moves where history is already at full share.
        prog.Uniform("vhTaaBlockScale", taaActiveThisFrame ? taaBlockScale : 0f);
        prog.Uniform("vhTaaJitterX", taaJitterX);
        prog.Uniform("vhTaaJitterY", taaJitterY);
        prog.Uniform("vhTaaStartPixels", LodTaaMath.NudgeStartPixels(taaReachPixels));
        prog.Uniform("vhTaaFullPixels", LodTaaMath.FullPixels(taaReachPixels));
        // Per-cell ownership supersedes the radius entirely: the mask knows which
        // individual cells vanilla owns, so leaving a radius active would suppress cached
        // terrain the mask says is still ours. With the mask off, the measured radius (or
        // the established constant) remains the only ownership decision.
        bool maskOwnsPixels = readinessMaskActive && readinessMask != null;
        prog.Uniform("cacheHandoffDistance", maskOwnsPixels
            ? MaskNearFloor()
            : readinessOwnsHandoff
                ? readinessHandoffDistance
                : LodNearHandoff.InnerDiscardRadius(viewDistance));
        prog.Uniform("maskEnabled", maskOwnsPixels ? 1 : 0);
        prog.Uniform("waterNearChunks", waterNearChunks);
        // Two integer scalars, never the Vec2i overload: that one reaches glUniform2f, which
        // an integer uniform rejects, and the camera chunk would silently stay at zero (G42).
        prog.Uniform("camChunkX", (int)VanillaRenderReadiness.ChunkCoordinate((double)camPos.X));
        prog.Uniform("camChunkZ", (int)VanillaRenderReadiness.ChunkCoordinate((double)camPos.Z));
        prog.Uniform("waterFresnel", WaterFresnel ? 1 : 0);
        prog.Uniform("waterGroundMurk", WaterGroundMurk ? 1 : 0);
        prog.Uniform("waterSurfaceVisible", WaterSurfaceVisible ? 1 : 0);
        bool waterSurfaceTextureOn = WaterSurfaceTextureReady;
        prog.Uniform("waterTextureEnabled", waterSurfaceTextureOn ? 1 : 0);
        prog.Uniform("waterTextureSlot", WaterSurfaceTextureSlot);
        prog.Uniform("waterTextureTintSlot", WaterSurfaceTintSlot);
        prog.Uniform("waterGlint", waterGlint);
        prog.Uniform("waterFoam", waterFoam);
        prog.Uniform("waterShallowness", Math.Clamp(WaterShallowness, 0f, 1f));
        prog.Uniform("waterMaxAlpha", Math.Clamp(WaterMaxAlpha, 0.05f, 2f));

        // Everything the engine's own liquid shader reads, under mod-owned names so no
        // engine-managed uniform can be shadowed. All public API: nothing here reaches into
        // the client. The counters animate, so they upload every frame with the rest.
        DefaultShaderUniforms engineUniforms = capi.Render.ShaderUniforms;
        prog.Uniform("vhSunPosRel", engineUniforms.SunPosition3D);
        prog.Uniform("vhReflectColor", capi.World.Calendar.ReflectColor);
        prog.Uniform("vhSunSpecular", engineUniforms.SunSpecularIntensity);
        prog.Uniform("vhWaterWave", engineUniforms.WaterWaveCounter);
        prog.Uniform("vhWaterStill", engineUniforms.WaterStillCounter);
        prog.Uniform("vhWindWave", engineUniforms.WindWaveCounter);
        prog.Uniform("vhWaterFlow", engineUniforms.WaterFlowCounter);
        prog.Uniform("vhWindSpeed", engineUniforms.WindSpeed);
        prog.Uniform("vhFoamOrigin", engineUniforms.PlayerPosForFoam);

        // What vertexwarp.vsh's applyLiquidWarping reads, under the engine's own names. The
        // include is already in our vertex body, so the warp itself is the game's code rather
        // than a copy of it; these are the inputs it would otherwise find at zero.
        prog.Uniform("playerpos", new Vec3f(
            (float)engineUniforms.playerReferencePos.X,
            (float)engineUniforms.playerReferencePos.Y,
            (float)engineUniforms.playerReferencePos.Z));
        prog.Uniform("waterWaveCounter", engineUniforms.WaterWaveCounter);
        prog.Uniform("windWaveCounter", engineUniforms.WindWaveCounter);
        prog.Uniform("waterWaveIntensity", engineUniforms.WaterWaveIntensity);
        prog.Uniform("windWaveIntensity", engineUniforms.WindWaveIntensity);
        prog.Uniform("windSpeed", engineUniforms.WindSpeed);

        // Held to the ring where a seam can be. Between the per-block strips water is still
        // merged into wide quads, and waving those would facet the surface rather than ripple
        // it - see vhWaveReach in the vertex body.
        prog.Uniform("vhWaveReach", WaterSeamStripBlocks > 0 ? WaterWaveReachBlocks : 0f);
        prog.Uniform("maskDebug", MaskDebugPaint ? 1 : 0);
        prog.Uniform("flatTopLight", FlatTopLight ? 1 : 0);
        prog.Uniform("lightMoonDir", LightMoonDirection ? 1 : 0);
        prog.Uniform("lightVanillaRamp", LightVanillaRamp ? 1 : 0);
        prog.Uniform("lightAmbientColor", LightAmbientColor ? 1 : 0);
        prog.Uniform("lightDayBoost", LightDayBoost ? 1 : 0);
        // `lightPosition` and `shadowIntensity` need no upload here: the shader includes
        // fogandlight.fsh, and the engine's own Use() uploads both to any program that
        // does. Only the ambient colour has no such carrier.
        prog.Uniform("rgbaAmbientIn", capi.Ambient.BlendedAmbientColor);
        // Match the engine's brightness floor set by the client settings
        prog.Uniform("minBrightness", Math.Max(0f, capi.Settings.Float.Get("minbrightness", 0f)));
        // 1 is exactly what the shader assumed before this correction, so the switch needs
        // no branch of its own on the GPU side.
        prog.Uniform("sunLightLevel", LightSunLevel ? SunLightLevel() : 1f);
        prog.Uniform("lightSkyDayLight", LightSkyDayLight ? 1 : 0);
        prog.Uniform("skyDayLight", SkyDayLight());
        prog.Uniform("silhouetteEnabled", SilhouetteFadeTargetActive ? 1 : 0);
        prog.Uniform("silhouetteSkyTint", SilhouetteSkyTint);
        prog.Uniform("silhouetteColorTint", SilhouetteColorTint);
        prog.Uniform("silhouetteLightLevelBias", SilhouetteLightLevelBias);
        prog.Uniform("silhouetteSeaLevel", capi.World.SeaLevel);
        if (maskOwnsPixels && readiness != null && readinessMaskTexture != null)
        {
            prog.Uniform("maskMinX", readiness.ActiveMinChunkX);
            prog.Uniform("maskMinZ", readiness.ActiveMinChunkZ);
            prog.Uniform("maskWidth", readiness.ActiveWidth);
            prog.Uniform("maskDepth", readiness.ActiveDepth);
            prog.Uniform("maskCapacity", readinessMask!.Width);
            prog.Uniform("maskVerticalChunks", readinessMask.VerticalChunks);
            prog.BindTexture2D("readinessMask", readinessMaskTexture.TextureId,
                ReadinessMaskUnit);
        }

        // The sink is not part of the band and does not wait for it: it is a constant
        // geometric offset that exists to stop two opaque surfaces fighting for depth in the
        // overlap, and it applies whenever cached terrain draws.
        prog.Uniform("bandSinkBlocks", Math.Max(0, BandSinkVoxels) / 16f);

        bool bandOwnsPixels = blendPipelineEnabled && maskOwnsPixels && seamFieldActive
            && seamField != null && seamFieldTexture != null;
        prog.Uniform("bandEnabled", bandOwnsPixels ? 1 : 0);
        prog.Uniform("bandDebug", BandDebugPaint ? 1 : 0);
        if (bandOwnsPixels)
        {
            // Scalars, never a packed integer vector: the client's only pair setter reaches
            // glUniform2f, which an integer uniform rejects outright and silently. See G42.
            prog.Uniform("bandMinCellX", seamField!.MinCellX);
            prog.Uniform("bandMinCellZ", seamField.MinCellZ);
            prog.Uniform("bandWidthCells", seamField.WidthCells);
            prog.Uniform("bandDepthCells", seamField.DepthCells);
            prog.Uniform("bandCapacity", seamField.Capacity);
            prog.Uniform("bandWidthBlocks", Math.Max(1f, BandWidthBlocks));
            prog.Uniform("bandStepBlocks", BandStepBlocks);
            prog.Uniform("bandStrength", BandStrength);
            prog.Uniform("bandInnerBlocks", Math.Max(0f, BandInnerBlocks));
            prog.Uniform("bandDarken", BandDarken);
            prog.Uniform("bandSaturate", BandSaturate);
            prog.Uniform("bandSpeckle", Math.Max(0f, BandSpeckle));
            // Unit 7, one past the mask's own.
            prog.BindTexture2D("seamField", seamFieldTexture!.TextureId, SeamFieldUnit);
        }

        // The table is bound whenever any mesh can be carrying a block slot, and NOT only
        // when textures are switched on: a textured quad's colour lives in the table too, so
        // an unbound sampler here would leave `.dltexture off` reading colours out of
        // whatever texture happened to be on unit 0.
        if (blockTexturesLive && blockTableTexture != 0)
            prog.BindTexture2D("blockTable", blockTableTexture, BlockTableUnit);

        bool blockTexturesOn = blendPipelineEnabled && blockTexturesLive && blockTextures != null
            && blockTextures.AtlasTextureId > 0 && BlockTexturesEnabled;
        prog.Uniform("blockTextureEnabled", blockTexturesOn ? 1 : 0);
        if (blockTexturesOn || waterSurfaceTextureOn)
        {
            // Both axes: the atlas is 4096 x 2048 on a fresh install, square only on upgraded ones.
            prog.Uniform("blockAtlasSize",
                (float)capi.BlockTextureAtlas.Size.Width, (float)capi.BlockTextureAtlas.Size.Height);
            prog.BindTexture2D("blockAtlas", blockTextures!.AtlasTextureId, BlockAtlasUnit);
        }
        if (blockTexturesOn)
        {
            prog.Uniform("blockTextureStrength", Math.Clamp(BlockTextureStrength, 0f, 1f));
            prog.Uniform("blockTextureMipLimit", Math.Max(0f, BlockTextureMipLimit));
            prog.Uniform("blockTextureDetail", Math.Max(0f, BlockTextureDetail));
        }

        UploadClimateTables();
        // The slot table is bound whenever anything can be carrying a slot, not only when the
        // field is on: it also carries the flat fallback every slot falls back to, and an unbound
        // sampler reads zero, which is a plausible-looking black tint.
        if (slotTableTexture != null)
            prog.BindTexture2D("slotTable", slotTableTexture.TextureId, SlotTableUnit);

        // The two rows every vertex reads, as uniform arrays rather than texels. Unconditional
        // and outside the climate switch: the fallback is what draws when the climate path is
        // off or cannot finish, so it must be present in exactly the cases the climate block
        // below is skipped. See LodTintRegistry.FallbackUniforms.
        prog.Uniforms4("tintFallback", LodTintRegistry.MaxSlots, tints.FallbackUniforms);
        prog.Uniforms4("tintPages", LodTintRegistry.MaxSlots, tints.PageUniforms);

        // The field itself is also the snow source. It stays live when biome colouring is
        // switched off or limited to nearby LOD levels; otherwise those performance settings
        // would silently put snow back on the old player-local global threshold.
        bool climateFieldOn = climateField.Active && climateFieldTexture != null;
        bool climateTintOn = ClimateFieldTint && climateFieldOn
            && climateMapTexture != null && slotTableTexture != null
            && climatePages.PageCount > 0;
        prog.Uniform("climateFieldEnabled", climateFieldOn ? 1 : 0);
        prog.Uniform("climateTintEnabled", climateTintOn ? 1 : 0);
        if (climateFieldOn)
        {
            prog.BindTexture2D("climateField", climateFieldTexture!.TextureId, ClimateFieldUnit);
            // Two integer scalars, never the Vec2i overload, which reaches glUniform2f. See G42.
            prog.Uniform("climateMinCellX", climateField.MinCellX);
            prog.Uniform("climateMinCellZ", climateField.MinCellZ);
            prog.Uniform("climateCapacity", climateField.Capacity);
            prog.Uniform("climateCellBlocks", (float)LodClimateField.CellBlocks);
            prog.Uniform("climateSnowMinY", climateField.SnowMinY);
            prog.Uniform("climateSnowStepY", climateField.SnowStepY);
            prog.Uniform("climateSeaLevel", (float)capi.World.SeaLevel);
        }
        if (climateTintOn)
        {
            prog.BindTexture2D("climateMaps", climateMapTexture!.TextureId, ClimateMapsUnit);
            if (seasonColumnTexture != null)
                prog.BindTexture2D("seasonColumn", seasonColumnTexture.TextureId,
                    SeasonColumnUnit);
            prog.Uniform("leafSeasonVariation",
                seasonColumnTexture != null ? Math.Clamp(LeafSeasonVariation, 0f, 1f) : 0f);
            prog.Uniform("leafSeasonFadeBlocks", Math.Max(2f, LeafSeasonFadeBlocks));
            prog.Uniform("leafSeasonCheapRow", LeafSeasonCheapRow ? 1 : 0);
            // Zero means every level pays. Anything else is the voxel size past which a
            // vertex takes the flat fallback instead of working out its own climate.
            prog.Uniform("climateFadeBlocks", Math.Max(0f, ClimateFadeBlocks));
            // Rainfall's height terms are keyed on Climate.Sealevel, which is a separate
            // static from the world map's sea level the lapse rate uses. Passed rather than
            // assumed equal - in the owner's world they are not.
            prog.Uniform("climateRainSeaLevel", Climate.Sealevel);
        }
        prog.Uniform("snowLineY", snowLineY);
        prog.Uniform("seasonTemperature", capi.Render.ShaderUniforms.SeasonTemperature);
    }


    /// <summary>
    /// What the climate field path computes for one block, term by term, on the CPU using the
    /// same arithmetic the vertex shader runs.
    ///
    /// The point is comparability: `.dltint` already prints the game's own answer for a block,
    /// and until now the only mod number beside it came from the height ramp. Reasoning about
    /// the difference from the ramp's number while the field is what draws is how two rounds
    /// of guessing happened. Every term is printed rather than the result alone, because the
    /// result agreeing is not evidence when four terms can cancel.
    /// </summary>
    internal List<string> DescribeClimateTint(int x, int y, int z, int slot)
    {
        var lines = new List<string>();
        if (slot <= LodTintRegistry.SlotNone || slot >= LodTintRegistry.MaxSlots)
        {
            lines.Add("D field: slot has no tint, so the field path does not apply");
            return lines;
        }

        int pageWord = tints.SlotTable[LodTintRegistry.SlotTableTexel(slot, LodTintRegistry.PageRow)];
        int page = (pageWord >> 16) & 0xFF;
        bool on = ClimateFieldTint && climateField.Active && climatePages.PageCount > 0;
        lines.Add($"D field: switch {(ClimateFieldTint ? "on" : "off")}, "
            + $"field {(climateField.Active ? "live" : "NOT LIVE")} "
            + $"({climateField.KnownCells} of {climateField.Capacity * climateField.Capacity} "
            + $"cells real), {climatePages.PageCount} map page(s), this slot uses page "
            + (page == LodClimateMapPages.NoPage ? "NONE so it draws from the ramp" : $"{page}"));

        if (!on || page == LodClimateMapPages.NoPage) return lines;
        if (!climateField.TryClimateAt(x + 0.5, z + 0.5,
                out float rawTemp, out float rawRain, out int spread))
        {
            lines.Add("D field: no climate cell here, which should not happen while it is live");
            return lines;
        }

        float seaLevelDist = y - capi.World.SeaLevel;
        float adjusted = Math.Clamp(rawTemp - seaLevelDist / 1.5f, 0f, 255f);
        int rainfall = LodClimateMath.RainfallAt(
            (int)MathF.Round(rawRain), y, Climate.Sealevel);
        float weight = LodSeasonWeight.Vanilla(adjusted, seaLevelDist);

        lines.Add($"D cell: raw temperature {rawTemp:F1}, raw rainfall {rawRain:F1}, "
            + $"spread {spread} " + (spread == 0
                ? "(real climate, measured here)"
                : "(NO climate here; taken from the nearest cell that had one)"));
        lines.Add($"D here: adjusted temperature {adjusted:F1}, rainfall {rainfall}, "
            + $"season weight {weight:F3}");

        if (!climatePages.TrySample(page, adjusted, rainfall,
                out float mr, out float mg, out float mb))
        {
            lines.Add("D field: the climate map page could not be sampled");
            return lines;
        }

        int seasonWord = tints.SlotTable[LodTintRegistry.SlotTableTexel(slot, LodTintRegistry.SeasonRow)];
        float sa = ((seasonWord >> 24) & 0xFF) / 255f;
        float sr = ((seasonWord >> 16) & 0xFF) / 255f;
        float sg = ((seasonWord >> 8) & 0xFF) / 255f;
        float sb = (seasonWord & 0xFF) / 255f;
        int shareWord = tints.SlotTable[LodTintRegistry.SlotTableTexel(slot, LodTintRegistry.ShareRow)];
        float hr = ((shareWord >> 16) & 0xFF) / 255f;
        float hg = ((shareWord >> 8) & 0xFF) / 255f;
        float hb = (shareWord & 0xFF) / 255f;

        lines.Add($"D climate map: R{mr:F3} G{mg:F3} B{mb:F3}");
        lines.Add($"D season: R{sr:F3} G{sg:F3} B{sb:F3}"
            + (sa > 0.5f ? "" : "  (no season blend on this slot)"));

        float w = weight * (sa > 0.5f ? 1f : 0f);
        float br = mr + (sr - mr) * w, bg = mg + (sg - mg) * w, bb = mb + (sb - mb) * w;
        lines.Add($"D blended: R{br:F3} G{bg:F3} B{bb:F3}");
        lines.Add($"D what the shader draws: R{Math.Clamp(hr + (1 - hr) * br, 0f, 1f):F3} "
            + $"G{Math.Clamp(hg + (1 - hg) * bg, 0f, 1f):F3} "
            + $"B{Math.Clamp(hb + (1 - hb) * bb, 0f, 1f):F3}  "
            + "(compare against C, the game's own answer above)");
        return lines;
    }

    /// <summary>
    /// Publish the three tables the shader finishes a tint from. All three go up through
    /// UploadOwnTexture, so none of them can take a unit away from a neighbour.
    /// </summary>
    void UploadClimateTables()
    {
        if (uploadedSlotTableVersion != tints.Version)
        {
            LoadedTexture texture = slotTableTexture ?? new LoadedTexture(
                capi, 0, LodTintRegistry.SlotTableWidth, LodTintRegistry.SlotTableHeight);
            UploadOwnTexture(tints.SlotTable, SlotTableUnit, ref texture);
            if (texture.TextureId > 0)
            {
                slotTableTexture = texture;
                uploadedSlotTableVersion = tints.Version;
            }
        }

        // Only the live column reaches the GPU - two kilobytes rather than the whole 136 KiB
        // store, because the column is a per-frame constant and only the row varies per vertex.
        // The condition is therefore the COLUMN as well as the version: it moves 128 times a
        // game year, and nothing is uploaded on the frames between.
        int liveColumn = LodSeasonMapPages.ColumnFor(tints.SeasonRel);
        if (seasonPages.PageCount > 0
            && (uploadedSeasonColumn != liveColumn
                || uploadedSeasonMapVersion != seasonPages.Version))
        {
            seasonPages.RefreshLiveColumn(liveColumn);

            // A newly adopted page makes the table taller, so the LoadedTexture's stored size
            // would disagree with the pixels and it has to be replaced. A column move keeps the
            // size and updates in place.
            bool resized = uploadedSeasonMapVersion != seasonPages.Version
                || seasonColumnTexture == null;
            LoadedTexture texture = resized
                ? new LoadedTexture(capi, 0,
                    seasonPages.LiveColumnWidth, seasonPages.LiveColumnHeight)
                : seasonColumnTexture!;
            UploadOwnTexture(seasonPages.LiveColumnTexels, SeasonColumnUnit, ref texture);
            if (texture.TextureId > 0)
            {
                if (!ReferenceEquals(texture, seasonColumnTexture))
                {
                    LoadedTexture? retiredSeason = seasonColumnTexture;
                    seasonColumnTexture = texture;
                    try { retiredSeason?.Dispose(); }
                    catch { /* the replacement is already live; retirement is best effort */ }
                }
                uploadedSeasonColumn = liveColumn;
                uploadedSeasonMapVersion = seasonPages.Version;
            }
        }

        if (uploadedClimateMapVersion != climatePages.Version && climatePages.PageCount > 0)
        {
            // A new page changes the texture's height, so the old LoadedTexture cannot be
            // updated in place - its stored size would disagree with the pixels. The one it
            // replaces is released rather than dropped: pages are only appended, but a
            // texture leaked per append is still a texture leaked.
            var texture = new LoadedTexture(
                capi, 0, climatePages.Width, climatePages.Height);
            UploadOwnTexture(climatePages.Texels, ClimateMapsUnit, ref texture);
            if (texture.TextureId > 0)
            {
                LoadedTexture? retired = climateMapTexture;
                climateMapTexture = texture;
                uploadedClimateMapVersion = climatePages.Version;
                try { retired?.Dispose(); }
                catch { /* the replacement is already live; retirement is best effort */ }
            }
        }

        if (climateField.Dirty && climateField.Active)
        {
            LoadedTexture texture = climateFieldTexture ?? new LoadedTexture(
                capi, 0, climateField.Width, climateField.Height);
            UploadOwnTexture(climateField.Texels, ClimateFieldUnit, ref texture);
            if (texture.TextureId > 0)
            {
                climateFieldTexture = texture;
                climateField.MarkUploaded();
            }
        }
    }

    void ConfigureRenderPaths()
    {
        if (renderPathConfigured || !gpuTelemetry.ProbeAttempted) return;
        renderPathConfigured = true;
        LodRenderPathSelection selection = LodRenderPathPolicy.Evaluate(
            gpuRendererPreference, gpuTelemetry.Decision);
        renderPaths.Configure(selection);
        capi.Logger.Notification(
            "[Landscapes] Opaque renderer: packed GPU path required; per-section and "
            + "expanded fallbacks disabled. Capability result: {0}.",
            selection.Reason);
        if (selection.ShadowEnabled)
        {
            requiredTerrainCapabilityFailure = null;
            AttachShadowArenas();
        }
        else if (LodRenderPathPolicy.RequestsRuntimeValidation(gpuRendererPreference))
        {
            requiredTerrainCapabilityFailure = DescribeGpuValidationFailure();
            capi.ShowChatMessage(
                "[Landscapes] Distant terrain's GPU check did not pass, so cached terrain "
                + "is hidden. Run .dlgpu on to re-check; details are in the client log.");
        }
    }

    (int Width, int Height) GpuFrameSize()
    {
        FrameBufferRef? primary = PrimaryFrameBuffer();
        int apiWidth = capi.Render.FrameWidth;
        int apiHeight = capi.Render.FrameHeight;
        (int width, int height) = LodGpuFrameSizePolicy.Select(
            apiWidth,
            apiHeight,
            primary?.Width ?? 0,
            primary?.Height ?? 0);
        if (!reportedGpuFrameSizeDisagreement && primary != null
            && (apiWidth != width || apiHeight != height))
        {
            reportedGpuFrameSizeDisagreement = true;
            capi.Logger.Notification(
                "[Landscapes] GPU copy size comes from the concrete primary framebuffer "
                + "{0}x{1}; IRenderAPI reports {2}x{3} and is diagnostic only.",
                width, height, apiWidth, apiHeight);
        }
        return (width, height);
    }

    /// <summary>
    /// Regional arenas for the validated GPU path. They retain only its selected geometry
    /// representation and stay bounded by a draw-distance-derived ceiling.
    /// </summary>
    void AttachShadowArenas()
    {
        LodGpuArenaMode mode = LodGpuArenaPolicy.Parse(
            Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_ARENA"));
        if (mode is LodGpuArenaMode.Off or LodGpuArenaMode.Invalid)
        {
            capi.Logger.Error(
                "[Landscapes] REQUIRED PACKED TERRAIN PATH FAILED: GPU regional arenas "
                + "are off; cached opaque terrain will be suppressed.");
            return;
        }

        AttachShadowArenas(mode);
    }

    void AttachShadowArenas(LodGpuArenaMode mode)
    {
        LodGpuArenaPolicy.ConfigurePageBytes(
            Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_ARENA_PAGE_MB"));
        // Sized from the player's own cached-terrain distance rather than a fixed number, so
        // the pool can hold whatever their settings ask the mod to draw. A cap of zero means
        // unlimited and gets a generous horizon instead. Read here rather than per frame
        // because the arenas are real GL buffers; changing the distance mid-session does not
        // resize them, and the log line below says which distance this pool was built for.
        double sizedFor = FarViewDistanceCap > 0
            ? FarViewDistanceCap
            : LodGpuArenaPolicy.UnlimitedDrawDistanceBlocks;
        long ceiling = LodGpuArenaPolicy.CeilingBytesFor(
            sizedFor, Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_ARENA_MB"));
        try
        {
            if (gpuFailureInjection.Take(LodGpuFailureStage.ArenaSetup))
                throw new InvalidOperationException(
                    "Phase 9 injected a regional arena setup failure");

            var backend = new LodGpuOpenGlArenaBackend(
                message => capi.Logger.Warning("{0}", message));
            LodGpuGeometryRetention retention = LodGpuGeometryMirror.SelectRetention(
                PackedDrawEnabled, ClusterDrawEnabled);
            gpuShadow.AttachMirror(new LodGpuGeometryMirror(
                backend, ceiling, verify: mode == LodGpuArenaMode.Verify,
                retention: retention));
            shadowBuilder = new LodGpuIndirectBuilder();
            splitFarBuilder = new LodGpuIndirectBuilder();
            packedBuilder = new LodGpuIndirectBuilder(packed: true);
            packedFarBuilder = new LodGpuIndirectBuilder(packed: true);
            clusterBuilder = new LodGpuIndirectBuilder(packed: true, clustered: true);
            clusterFarBuilder = new LodGpuIndirectBuilder(packed: true, clustered: true);
            // .dlgpu can be run again while a drawer already exists - on to verify, for
            // instance. The mirror disposes its own predecessor; this one has to be told.
            indirectDrawer?.Dispose();
            packedDrawer?.Dispose();
            cullPass?.Dispose();
            cullPass = new LodGpuCullPass(
                message => capi.Logger.Warning(message), gpuTelemetry);
            indirectDrawer = new LodGpuIndirectDrawer(
                new LodGpuOpenGlDrawBackend(message => capi.Logger.Warning("{0}", message)),
                message => capi.Logger.Warning("{0}", message));
            packedDrawer = new LodGpuIndirectDrawer(
                new LodGpuOpenGlPackedDrawBackend(message => capi.Logger.Warning("{0}", message)),
                message => capi.Logger.Warning("{0}", message));
            long packedCeiling = LodGpuArenaPolicy.PackedLimits(ceiling).CeilingBytes;
            long clusterCeiling = LodGpuArenaPolicy.PackedClusterLimits(ceiling).CeilingBytes;
            capi.Logger.Notification(
                "[Landscapes] GPU arenas on: retaining {0} geometry only; ceilings are "
                + "{1} MiB expanded, {2} MiB packed and {3} MiB clustered, with {4} MiB "
                + "vertex pages and {5} page sets; content verification {6}. Sized for a "
                + "{7} draw distance ({8} sections); each ceiling "
                + "is a cap, not a reservation, so pages are only "
                + "committed as sections arrive.",
                retention,
                ceiling / (1024 * 1024),
                packedCeiling / (1024 * 1024),
                clusterCeiling / (1024 * 1024),
                LodGpuArenaPolicy.VertexPageBytes / (1024 * 1024),
                LodGpuArenaPolicy.PageSets(ceiling),
                mode == LodGpuArenaMode.Verify ? "on" : "off",
                FarViewDistanceCap > 0
                    ? FarViewDistanceCap + "-block"
                    : "an unlimited (modelled as "
                        + LodGpuArenaPolicy.UnlimitedDrawDistanceBlocks.ToString("0") + "-block)",
                LodGpuArenaPolicy.SectionsWithin(sizedFor).ToString("0"));
        }
        catch (Exception e)
        {
            capi.Logger.Error(
                "[Landscapes] REQUIRED PACKED TERRAIN PATH FAILED: GPU arenas could "
                + "not start; cached opaque terrain will be suppressed: {0}", e.Message);
        }
    }

    void ApplyGpuDrawFailureInjection()
    {
        if (indirectDrawer == null && packedDrawer == null) return;
        if (!gpuFailureInjection.Take(LodGpuFailureStage.IndirectDraw)) return;

        const string reason = "Phase 9 injected an indirect draw failure";
        indirectDrawer?.InjectFailure(reason);
        packedDrawer?.InjectFailure(reason);
    }

    /// <summary>Arena occupancy for the periodic report, or null when no arena is attached.</summary>
    public string? DescribeGpuArena() => gpuShadow.Mirror?.Describe();

    // ---- In-game GPU shadow switch ----

    LodGpuArenaMode? requestedShadowMode;

    /// <summary>
    /// The arena mode someone last ASKED for, which is what gets saved.
    ///
    /// The request rather than the effective state, matching the chunk mask: a path the
    /// capability probe refused this session reports itself off, and writing that back would
    /// turn it off permanently on a machine where a driver update might enable it.
    /// </summary>
    public bool GpuShadowRequested { get; private set; }

    /// <summary>
    /// Asks for the measurement shadow to be turned on or off. Called from a chat command,
    /// so it only records the request: capability probing, GL resource creation and teardown
    /// all belong to the render thread and happen at the start of the next frame.
    /// </summary>
    public bool RequestGpuShadow(string mode)
    {
        LodGpuArenaMode parsed = LodGpuArenaPolicy.Parse(mode);
        if (parsed == LodGpuArenaMode.Invalid) return false;

        requestedShadowMode = parsed;
        GpuShadowRequested = parsed != LodGpuArenaMode.Off;
        if (parsed != LodGpuArenaMode.Off) gpuTelemetry.RequestRuntimeValidation();
        return true;
    }

    void ApplyGpuShadowRequest()
    {
        if (requestedShadowMode is not LodGpuArenaMode mode) return;
        if (!gpuTelemetry.ProbeAttempted) return;
        requestedShadowMode = null;

        LodRenderPathSelection selection = LodRenderPathPolicy.Evaluate(
            mode == LodGpuArenaMode.Off ? "off" : "shadow", gpuTelemetry.Decision);

        if (mode == LodGpuArenaMode.Off)
        {
            requiredTerrainCapabilityFailure = null;
            renderPaths.SetShadowEnabled(false, selection);
            gpuShadow.DetachMirror();
            shadowBuilder = null;
            splitFarBuilder = null;
            packedBuilder = null;
            packedFarBuilder = null;
            clusterBuilder = null;
            clusterFarBuilder = null;
            // The drawer's buffers reference nothing the arenas own, but its whole reason
            // to exist goes away with them, and a stale vertex array would outlive the
            // pages its batches name.
            IndirectDrawEnabled = false;
            indirectDrawingThisFrame = false;
            indirectDrawer?.Dispose();
            indirectDrawer = null;
            packedDrawer?.Dispose();
            packedDrawer = null;
            cullPass?.Dispose();
            cullPass = null;
            GpuCullEnabled = false;
            ShadowIndirectCommands = 0;
            ShadowIndirectSections = 0;
            ShadowIndirectBatches = 0;
            ShadowIndirectDropped = 0;
            ShadowIndirectMissing = 0;
            ShadowIndirectCoverage = 0;
            capi.Logger.Notification(
                "[Landscapes] GPU measurement shadow off; every arena buffer released.");
            return;
        }

        if (!renderPaths.SetShadowEnabled(true, selection))
        {
            requiredTerrainCapabilityFailure = DescribeGpuValidationFailure();
            capi.Logger.Warning(
                "[Landscapes] GPU measurement shadow refused: {0}. Run .dlgpu on to "
                + "re-check after the frame or window changes.", selection.Reason);
            capi.ShowChatMessage(
                "[Landscapes] Distant terrain's GPU check still did not pass. Run "
                + ".dlgpu on to re-check; details are in the client log.");
            return;
        }

        requiredTerrainCapabilityFailure = null;
        AttachShadowArenas(mode);
        RemeshForShadowBackfill();
    }

    /// <summary>
    /// The arenas only ever see geometry as it is published, so the GPU path switched on
    /// mid-session would otherwise measure whatever happens to be remeshed afterwards.
    /// Marking the live sections dirty replays them through the ordinary publication path,
    /// at the ordinary budget, until the mirror matches what is actually on screen.
    /// </summary>
    void RemeshForShadowBackfill()
    {
        int queued = 0;
        foreach (long key in sectionMeshes.Keys)
        {
            world.RenderDirty.Add(key);
            queued++;
        }
        capi.Logger.Notification(
            "[Landscapes] GPU regional path on; {0} live sections queued for "
            + "re-mesh so the arenas match what is drawn. Give it a moment before reading "
            + "the numbers.", queued);
    }

    /// <summary>One line for the in-game switch, describing what is measured right now.</summary>
    public string DescribeGpuShadow()
    {
        LodGpuGeometryMirror? mirror = gpuShadow.Mirror;
        if (mirror == null)
        {
            return gpuTelemetry.ProbeAttempted && !gpuTelemetry.Decision.Tier1RuntimeValidated
                ? "off - this driver did not pass the GPU capability checks: "
                    + gpuTelemetry.Decision.Reason
                : "off";
        }

        return $"on{(mirror.Verifying ? " with content verification" : "")}. "
            + $"{mirror.Count} sections mirrored, {mirror.LiveBytes / (1024.0 * 1024.0):0.0} MiB live. "
            + $"Last frame {ShadowIndirectCommands} draw commands for {ShadowIndirectSections} "
            + $"sections would have been {ShadowIndirectBatches} multi-draw batches "
            + $"({ShadowIndirectCoverage:P0} of drawn terrain covered"
            + (ShadowIndirectDropped > 0 ? $", {ShadowIndirectDropped} spans stale" : "")
            + "). Nothing is drawn from the arenas.";
    }

    void PrepareLegacyFrame(LodRenderFrame frame)
    {
        renderCullDistanceSquared = frame.CullDistanceSquared;
    }

    void DrawPackedOpaque()
    {
        if (!indirectDrawingThisFrame)
        {
            LastIndirectBatches = 0;
            LastIndirectCommands = 0;
            LastIndirectLeftovers = drawList.Count;
            ReportRequiredTerrainFailure(DescribeRequiredTerrainFailure());
            return;
        }

        if (!depthSplitThisFrame) gpuTelemetry.BeginOpaque();
        ActiveNearBuilder?.Begin();
        if (depthSplitThisFrame) ActiveFarBuilder?.Begin();
        indirectPassActive = indirectDrawingThisFrame;
        indirectLeftovers.Clear();
        indirectFarLeftovers.Clear();
        indirectRecorded.Clear();
        indirectFarRecorded.Clear();
        try
        {
            if (OpaqueFrontToBack)
            {
                LodOpaqueDrawOrder.FillFrontToBack(
                    opaqueFrontToBack, drawList, camPos.X, camPos.Z);
                foreach (LodOpaqueDrawEntry entry in opaqueFrontToBack)
                {
                    long key = entry.Key;
                    if (!sectionMeshes.TryGetValue(key, out MeshRef? mesh)) continue;
                    if (SkipVanillaOwnedSection(key)) continue;
                    if (!SetupSectionTransform(key, renderCullDistanceSquared)) continue;
                    RenderOpaqueMesh(key, mesh);
                }
            }
            else
            {
                foreach (long key in drawList)
                {
                    if (!sectionMeshes.TryGetValue(key, out MeshRef? mesh)) continue;
                    if (SkipVanillaOwnedSection(key)) continue;
                    if (!SetupSectionTransform(key, renderCullDistanceSquared)) continue;
                    RenderOpaqueMesh(key, mesh);
                }
            }

            // The walk is over, so the command list is complete and the batches can be
            // formed. Closing it here rather than in the finally is what lets the pass
            // draw from it; a walk that threw simply leaves the builder to be reset by
            // the next frame's Begin.
            indirectPassActive = false;
            EndShadowCommands();
            // The real walk already recorded every box the classifier should see. Failure
            // accounting must not add the same section a second time.
            hzbCollecting = false;
            if (indirectDrawingThisFrame)
            {
                DrawIndirectOpaque();
                if (LastIndirectLeftovers > 0)
                    ReportRequiredTerrainFailure(
                        "one or more selected sections were absent from or rejected by the packed arenas");
                else
                    reportedRequiredTerrainFailure = null;
            }
        }
        finally
        {
            indirectPassActive = false;
            if (!depthSplitThisFrame) gpuTelemetry.EndOpaque();
        }
    }

    /// <summary>
    /// Issues this frame's multi-draws. Anything they cannot cover remains visibly absent.
    ///
    /// The established program keeps its own uniform state while it is not in use, so the
    /// switch costs one Use/Stop pair plus this frame's frame-uniforms for the second
    /// program - about thirty uploads against the eighty to one hundred and eighty
    /// per-section draws the batches replace.
    /// </summary>
    void DrawIndirectOpaque()
    {
        LastIndirectBatches = 0;
        LastIndirectCommands = 0;
        LastIndirectLeftovers = indirectLeftovers.Count + indirectFarLeftovers.Count;
        LodGpuIndirectBuilder? nearBuilder = ActiveNearBuilder;
        LodGpuIndirectDrawer? activeDrawer = ActiveDrawer;
        IShaderProgram? activeProgram = ActiveIndirectProgram;
        if (nearBuilder == null || activeDrawer == null || activeProgram == null
            || prog == null)
        {
            indirectLeftovers.AddRange(indirectRecorded);
            indirectFarLeftovers.AddRange(indirectFarRecorded);
            RejectIndirectLeftovers(indirectLeftovers);
            RejectIndirectLeftovers(indirectFarLeftovers);
            LastIndirectLeftovers = indirectLeftovers.Count + indirectFarLeftovers.Count;
            return;
        }

        if (depthSplitThisFrame && ActiveFarBuilder != null)
        {
            DrawSplitIndirectOpaque();
            return;
        }

        DrawIndirectBucket(nearBuilder, indirectRecorded, indirectLeftovers,
            applyFrameUniforms: true);
        RejectIndirectLeftovers(indirectLeftovers);
        LastIndirectLeftovers = indirectLeftovers.Count;
    }

    /// <summary>
    /// Draws near terrain, refreshes the pyramid while the terrain shader is released,
    /// then draws the far bucket against that same-frame picture. Each bucket has its own
    /// missing list, so a refused or partially issued multi-draw is reported before the
    /// frame leaves this method and never concealed by another renderer.
    /// </summary>
    void DrawSplitIndirectOpaque()
    {
        SplitDepthFrames++;
        LodGpuIndirectBuilder nearBuilder = ActiveNearBuilder!;
        LodGpuIndirectBuilder farBuilder = ActiveFarBuilder!;
        SplitDepthNearCommands += nearBuilder.CommandCount;
        SplitDepthFarCommands += farBuilder.CommandCount;

        bool nearDrewAnything = nearBuilder.CommandCount > 0 || indirectLeftovers.Count > 0;
        bool farHasAnything = farBuilder.CommandCount > 0 || indirectFarLeftovers.Count > 0;

        if (nearDrewAnything)
        {
            gpuTelemetry.BeginSplitNear();
            try
            {
                DrawIndirectBucket(nearBuilder, indirectRecorded, indirectLeftovers,
                    applyFrameUniforms: true);
                RejectIndirectLeftovers(indirectLeftovers);
            }
            finally
            {
                gpuTelemetry.EndSplitNear();
            }
        }

        if (nearDrewAnything && farHasAnything)
        {
            // Between opaque buckets, never after water. Water writes depth even though it is
            // blended, so including it would allow transparent surfaces to hide solid ground.
            // G78: the engine refuses the reduction shader while lodterrain is still active.
            prog!.Stop();
            try
            {
                BuildDepthPyramid(splitMid: true);
                if (LastHzbBuild.Built) SplitDepthMidBuilds++;
            }
            finally
            {
                prog.Use();
            }
        }

        if (farHasAnything)
        {
            gpuTelemetry.BeginSplitFar();
            try
            {
                DrawIndirectBucket(farBuilder, indirectFarRecorded, indirectFarLeftovers,
                    applyFrameUniforms: true);
                RejectIndirectLeftovers(indirectFarLeftovers);
            }
            finally
            {
                gpuTelemetry.EndSplitFar();
            }
        }

        LastIndirectLeftovers = indirectLeftovers.Count + indirectFarLeftovers.Count;
    }

    /// <summary>Issues one complete command bucket and reports any uncovered geometry.</summary>
    void DrawIndirectBucket(
        LodGpuIndirectBuilder builder,
        List<long> recorded,
        List<long> leftovers,
        bool applyFrameUniforms)
    {

        bool drew = false;
        LodGpuIndirectDrawer drawer = builder.Packed ? packedDrawer! : indirectDrawer!;
        IShaderProgram drawProgram = builder.Packed ? packedProg! : indirectProg!;
        prog!.Stop();
        try
        {
            drawProgram.Use();
            if (applyFrameUniforms)
            {
                ApplyFrameUniforms(drawProgram, ApprovedViewDistance());
            }
            LodGpuCullBucket cullBucket = !depthSplitThisFrame
                ? LodGpuCullBucket.General
                : ReferenceEquals(builder, ActiveFarBuilder)
                    ? LodGpuCullBucket.SplitFar
                    : LodGpuCullBucket.SplitNear;
            drew = drawer.Draw(builder, BuildCullRequest(cullBucket));
        }
        catch (Exception e)
        {
            capi.Logger.Error(
                "[Landscapes] REQUIRED PACKED TERRAIN PATH FAILED during draw; "
                + "cached opaque terrain is suppressed: {0}", e.Message);
        }
        finally
        {
            drawProgram.Stop();
            prog.Use();

            // Use() is not free of side effects, and this is the bug that cost four builds.
            //
            // ShaderProgramBase.Use() re-binds every sampler the shared includes declare -
            // shadowMapFar, shadowMapNear, sky, glow, liquidDepth - to whichever texture unit
            // the LINKER gave that name in THIS program. Those numbers are per program, and
            // the three variants here declare different sampler sets, so a unit that is free
            // in the packed program is not necessarily free in the legacy one. Texture unit
            // bindings are global GL state, so one of those binds can land on unit 7 and take
            // the seam field away.
            //
            // The opaque pass never saw it: it re-applies the frame uniforms immediately after
            // its own Use(), which rebinds everything. The water pass did, because nothing
            // re-applied anything between this Use() and its draws - and water and thin cover
            // are the two things drawn there. The result was a seam field that sampled a
            // shadow map and read ~0, which the shader reads as "distance zero from vanilla",
            // which is full band strength, over the whole field window and nowhere else. A
            // flat square, on water and on flowers, with the ground beside it correct.
            //
            // Exactly G42's shape: not an error, not a log line, and a wrong value that looks
            // like a legitimate answer.
            RebindOwnSamplers(prog);
        }

        if (drew)
        {
            LastIndirectBatches += drawer.LastBatches;
            LastIndirectCommands += drawer.LastCommands;
            // A multi-draw is a draw call; counting it keeps the periodic report's
            // submission figure meaning the same thing on both paths.
            OpaqueDrawCalls += drawer.LastBatches;
        }
        else
        {
            // Keep the failed candidates only so the missing coverage and its count are
            // reported. The retired per-section renderer must never conceal this failure.
            leftovers.AddRange(recorded);
        }
    }

    /// <summary>
    /// What this frame can offer the cull, or nothing at all.
    ///
    /// Every condition here is a reason to draw everything rather than a reason to guess. The
    /// switch is off, the pyramid was not built this frame, the shader never compiled - each
    /// returns a request the drawer will decline, and declining means every command the CPU
    /// approved is drawn. There is deliberately no branch that culls on partial information.
    ///
    /// The matrix is copied from the frustum rather than kept from earlier in the frame,
    /// because it must be the one the boxes were built against. The frustum was updated at
    /// the top of this frame from the engine's own matrices and the boxes are camera-relative
    /// against that same camera, so the two agree by construction.
    /// </summary>
    LodGpuCullRequest BuildCullRequest(LodGpuCullBucket bucket)
    {
        if (!GpuCullEnabled || !DepthPyramidEnabled || cullPass == null) return default;

        // Ahead of the picture checks, not behind them. Creation used to sit last, so a
        // session where no picture had been built yet never attempted it - and the status line
        // then said "the cull shader has not been created", which reads like a driver refusing
        // it. In the 2026-08-23 log that line was printed one second before the first picture
        // existed and was never asked again, leaving no evidence either way for the whole run.
        // Attempting it here means one rendered frame settles the question for good.
        if (!cullPass.Available && !cullPass.TryCreate()) return default;

        if (depthPyramid == null || !depthPyramid.Allocated) return default;
        if (!LastHzbBuild.Built) return default;

        frustum.CopyViewProjection(cullViewProjection);

        return new LodGpuCullRequest(
            cullPass,
            cullViewProjection,
            depthPyramid.TextureName,
            depthPyramid.Width,
            depthPyramid.Height,
            depthPyramid.Levels,
            LodHzbProjection.OcclusionDepthBias,
            bucket);
    }

    /// <summary>
    /// Rejects sections the required packed draw did not cover. Missing terrain is the
    /// deliberate development signal; substituting MeshRef draws here hid broken paths.
    /// </summary>
    void RejectIndirectLeftovers(List<long> _)
    {
        // Deliberately empty. The caller retains the count for status and reports the
        // aggregate failure after both near and far buckets have finished.
    }

    void ReportRequiredTerrainFailure(string reason)
    {
        if (reportedRequiredTerrainFailure == reason) return;
        reportedRequiredTerrainFailure = reason;
        capi.Logger.Error(
            "[Landscapes] REQUIRED PACKED TERRAIN PATH FAILED; cached opaque terrain "
            + "is suppressed and no legacy/expanded fallback will run: {0}.", reason);
    }

    string DescribeRequiredTerrainFailure()
    {
        if (!renderPaths.ShadowEnabled)
            return requiredTerrainCapabilityFailure ?? "regional GPU publication is disabled";
        if (!IndirectDrawEnabled) return "GPU drawing was explicitly disabled";
        if (!PackedDrawEnabled) return "packed drawing was explicitly disabled";
        if (!packedShaderOk || packedProg == null || packedProg.LoadError)
            return "the packed terrain shader is unavailable";
        if (packedDrawer is not { Ready: true }) return "the packed indirect drawer is unavailable";
        if (gpuShadow.Mirror == null || packedBuilder == null)
            return "the packed regional arenas are unavailable";
        if (ClusterDrawEnabled && clusterBuilder == null)
            return "clustered packed drawing was requested but its builder is unavailable";
        return "the required packed renderer did not become ready";
    }

    string DescribeGpuValidationFailure()
    {
        LodGpuDepthCopyValidation depthCopy = gpuTelemetry.DepthCopyValidation;
        string depthReason = string.IsNullOrEmpty(depthCopy.FailureReason)
            ? gpuTelemetry.Decision.Reason
            : depthCopy.FailureReason;
        return "GPU runtime validation did not pass: " + depthReason + "; "
            + depthCopy.Dimensions.Describe() + ". Run .dlgpu on to re-check";
    }

    /// <summary>
    /// Closes the shadow command list for this frame. It is only ever read as telemetry:
    /// no buffer is uploaded and no draw is issued from it in this phase.
    /// </summary>
    void EndShadowCommands()
    {
        LodGpuGeometryMirror? mirror = gpuShadow.Mirror;
        LodGpuIndirectBuilder? nearBuilder = ActiveNearBuilder;
        LodGpuIndirectBuilder? farBuilder = ActiveFarBuilder;
        if (nearBuilder == null || mirror == null) return;
        try
        {
            nearBuilder.End(mirror);
            if (depthSplitThisFrame && farBuilder != null)
                farBuilder.End(mirror);

            ShadowIndirectCommands = nearBuilder.CommandCount
                + (depthSplitThisFrame ? farBuilder?.CommandCount ?? 0 : 0);
            ShadowIndirectBatches = nearBuilder.Batches.Count
                + (depthSplitThisFrame ? farBuilder?.Batches.Count ?? 0 : 0);
            ShadowIndirectDropped = nearBuilder.CandidatesDropped
                + (depthSplitThisFrame ? farBuilder?.CandidatesDropped ?? 0 : 0);
            ShadowIndirectMissing = nearBuilder.MissingSections
                + (depthSplitThisFrame ? farBuilder?.MissingSections ?? 0 : 0);
            int addedSections = nearBuilder.AddedSections
                + (depthSplitThisFrame ? farBuilder?.AddedSections ?? 0 : 0);
            ShadowIndirectSections = addedSections;
            int total = addedSections + ShadowIndirectMissing;
            ShadowIndirectCoverage = total == 0 ? 1 : addedSections / (double)total;
        }
        catch (Exception e)
        {
            shadowBuilder = null;
            splitFarBuilder = null;
            packedBuilder = null;
            packedFarBuilder = null;
            clusterBuilder = null;
            clusterFarBuilder = null;
            capi.Logger.Error(
                "[Landscapes] REQUIRED PACKED TERRAIN PATH FAILED: indirect command "
                + "building stopped; uncovered cached terrain will remain absent: {0}", e.Message);
        }
    }

    public int ShadowIndirectCommands { get; private set; }
    public int ShadowIndirectSections { get; private set; }
    public int ShadowIndirectBatches { get; private set; }
    public int ShadowIndirectDropped { get; private set; }
    public int ShadowIndirectMissing { get; private set; }
    public double ShadowIndirectCoverage { get; private set; }

    /// <summary>
    /// Re-binds this mod's own samplers to the units its shader expects.
    ///
    /// Sampler UNIFORMS are per-program and survive anything. The texture unit BINDINGS they
    /// point at are global GL state, and the engine rewrites several of them on every
    /// <c>Use()</c> - so any code path that activates a program without re-applying the frame
    /// uniforms is drawing with whatever the engine left on those units.
    ///
    /// Cheap enough to be unconditional: four binds, once per pass, against a defect that is
    /// invisible until someone paints a debug colour and notices the shape is wrong.
    /// </summary>
    void RebindOwnSamplers(IShaderProgram target)
    {
        if (readinessMaskActive && readinessMask != null && readinessMaskTexture != null)
            target.BindTexture2D("readinessMask", readinessMaskTexture.TextureId,
                ReadinessMaskUnit);
        if (seamFieldActive && seamField != null && seamFieldTexture != null)
            target.BindTexture2D("seamField", seamFieldTexture.TextureId, SeamFieldUnit);
        if (blockTexturesLive && blockTableTexture != 0)
            target.BindTexture2D("blockTable", blockTableTexture, BlockTableUnit);
        if (((blendPipelineEnabled && BlockTexturesEnabled) || WaterSurfaceTextureReady)
            && blockTexturesLive
            && blockTextures != null
            && blockTextures.AtlasTextureId > 0)
        {
            target.BindTexture2D("blockAtlas", blockTextures.AtlasTextureId, BlockAtlasUnit);
        }
        if (slotTableTexture != null)
            target.BindTexture2D("slotTable", slotTableTexture.TextureId, SlotTableUnit);
        if (climateFieldTexture != null)
            target.BindTexture2D("climateField", climateFieldTexture.TextureId, ClimateFieldUnit);
        if (climateMapTexture != null)
            target.BindTexture2D("climateMaps", climateMapTexture.TextureId, ClimateMapsUnit);
        if (seasonColumnTexture != null)
            target.BindTexture2D("seasonColumn", seasonColumnTexture.TextureId,
                SeasonColumnUnit);
    }

    void DrawWaterMeshes()
    {
        var rapi = capi.Render;

        // Belt and braces with the rebind in DrawIndirectBucket: this pass is the one that was
        // measured drawing with a clobbered seam field, and it must be correct whether or not
        // an indirect bucket ran this frame to restore it.
        RebindOwnSamplers(prog!);
        rapi.GlDisableCullFace();
        rapi.GlToggleBlend(true);

        // Cached water must not write depth, and this is the root fix for the whole water
        // problem rather than a tuning choice.
        //
        // This mod draws in EnumRenderStage.Opaque. The engine draws its own water in the
        // LATER OIT stage, from SystemRenderTerrain's OIT renderer, into an accumulation
        // buffer that is composited over the opaque scene - and it enters that stage with
        // GlDepthMask(false) and the depth test still on, reading the depth this stage left
        // behind. So every cached water fragment that wrote depth here depth-REJECTED the
        // real animated surface behind it, everywhere the two overlapped, and the cache's
        // flat sheet silently stood in for vanilla's own water on the lake in front of the
        // player. Nothing errored and nothing was missing, which is why it was read as a
        // colour problem for three builds. G126.
        //
        // With the write off, the depth TEST still runs: cached water is still occluded by
        // anything nearer in the depth buffer, and it is still drawn after the opaque pass,
        // so cached and vanilla ground both remain in front of it where they should be. What
        // changes is only that vanilla's water is now allowed to draw on top - which makes
        // every suppression rule for cached water a soft one, because the failure mode of a
        // cell the ownership mask has not yet proven becomes a faint tint under real water
        // instead of a replacement of it.
        //
        // Restored explicitly rather than left to the engine: the opaque stage owns
        // GlDepthMask(true) and anything registered after this renderer inherits whatever we
        // leave behind.
        GL.DepthMask(false);
        gpuTelemetry.BeginWater();
        try
        {
            foreach (long key in drawList)
            {
                if (!waterMeshes.TryGetValue(key, out MeshRef? mesh)) continue;
                // Never skipped whole for ownership. A section is far wider than the near
                // cutoff, and the per-fragment test expresses both that cutoff and the
                // outward bound on ownership (G124) exactly, where a whole-section decision
                // could express neither.
                if (!SetupSectionTransform(key, renderCullDistanceSquared, waterPass: true)) continue;
                SubmitWaterMesh(key, mesh);
            }
        }
        finally
        {
            gpuTelemetry.EndWater();
            GL.DepthMask(true);
            rapi.GlToggleBlend(false);
        }
    }

    /// <summary>
    /// True when every ownership cell this section covers is committed vanilla-ready, so
    /// the mask would discard all of its fragments anyway. Skipping here removes the
    /// uniform uploads, the transform and the draw call itself, which is the only part of
    /// this design that saves CPU rather than shading.
    ///
    /// It requires the mask to be live: the aggregate and the texture are committed by the
    /// same publication, so a skip can never run ahead of what the GPU would have drawn.
    /// Residency is deliberately untouched - the mesh stays resident and warm so vanilla
    /// unloading restores cached coverage without a reload or a remesh.
    /// </summary>
    bool SkipVanillaOwnedSection(long key)
    {
        // A skipped section never reaches the shader, so it could never be painted.
        if (MaskDebugPaint) return false;
        if (!WholeMeshSkip) return false;
        if (!readinessMaskActive || readiness == null) return false;
        if (readiness.Classify(key) != VanillaSectionOwnership.VanillaOnly) return false;
        VanillaOwnedDrawsSkipped++;
        skippedLastFrame.Add(key);
        return true;
    }

    readonly HashSet<long> skippedLastFrame = new();

    public string DescribeOcclusionCulling() => OcclusionCullingEnabled
        ? "on: cached terrain renders after vanilla for ordinary depth rejection"
        : "off: cached terrain renders before vanilla";

    public string DescribeTemporalOcclusion()
    {
        if (temporalOcclusionFailed) return "failed; drawing all terrain";
        if (!TemporalOcclusionEnabled) return "off";
        if (!OcclusionCullingEnabled) return "suspended until post-vanilla order is on";
        return $"on ({TemporalOcclusionProfileName}): {LastTemporalOcclusionDrawsSkipped} hidden opaque draws skipped last frame, "
            + $"{LastTemporalOcclusionSeamDraws} seam draws protected, "
            + $"{LastTemporalOcclusionEdgeDraws} turning-edge draws protected, "
            + $"{TemporalOcclusionHiddenResults}/{TemporalOcclusionResults} hidden results, "
            + $"{TemporalOcclusionStaleResults} stale results/{TemporalOcclusionGlobalInvalidations} global invalidations, "
            + $"{temporalOcclusionPending.Count} queries pending";
    }

    bool TemporalOcclusionActive => TemporalOcclusionEnabled
        && OcclusionCullingEnabled && !temporalOcclusionFailed
        && !indirectDrawingThisFrame;

    /// <summary>
    /// Draw opaque cached terrain from the regional arenas instead of one call per
    /// section. Session-only and off by default: this is the first thing in the mod that
    /// puts a pixel on screen from a buffer the established renderer never touched.
    /// </summary>
    /// <remarks>
    /// Default-off, and settable from the environment so the benchmark harness can pin one
    /// side of the comparison for a whole run. Phase 3's gate is a controlled A/B, and
    /// session 40 lost a run to a comparison whose switches were set by hand and did not
    /// match; a chat command cannot be part of a scripted route.
    /// </remarks>
    public bool IndirectDrawEnabled { get; set; } =
        Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_INDIRECT") != "0";

    /// <summary>
    /// Pull exact twelve-byte greedy quads instead of expanded vertices/indices. It is an
    /// opt-in Phase 7 comparison layered on the same CPU candidates, section records,
    /// culling and fallback as the accepted indirect path.
    /// </summary>
    public bool PackedDrawEnabled { get; set; } =
        Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_PACKED") != "0";

    /// <summary>
    /// Phase 8 comparison: draw the packed stream as moderate 4x4 spatial clusters, each
    /// with its own conservative geometry bounds and cull command. Session-only and
    /// default-off until the extra commands and split quads beat their measured cost.
    /// </summary>
    public bool ClusterDrawEnabled { get; set; } =
        Environment.GetEnvironmentVariable("VINTAGEHORIZONS_GPU_CLUSTERS") != "0";

    /// <summary>
    /// Everything that has to hold before a frame may take the fast path. Read once per
    /// frame into <see cref="indirectDrawingThisFrame"/>.
    /// </summary>
    bool IndirectDrawAvailable => IndirectDrawEnabled
        && indirectShaderOk && indirectProg != null
        && indirectDrawer is { Ready: true }
        && shadowBuilder != null
        && gpuShadow.Mirror != null;

    bool PackedDrawAvailable => IndirectDrawEnabled && PackedDrawEnabled
        && packedShaderOk && packedProg != null
        && packedDrawer is { Ready: true }
        && packedBuilder != null
        && gpuShadow.Mirror != null;

    bool ClusterDrawAvailable => ClusterDrawEnabled
        && clusterBuilder != null
        && gpuShadow.Mirror != null;

    bool packedDrawingThisFrame;
    bool clusterDrawingThisFrame;

    public string DescribePackedDraw()
    {
        LodGpuGeometryMirror? mirror = gpuShadow.Mirror;
        if (!PackedDrawEnabled)
            return mirror == null
                ? "off"
                : $"off: expanded batching uses {mirror.LiveBytes / (1024.0 * 1024.0):0.0} MiB; "
                    + "unselected packed copies are not retained";
        if (!IndirectDrawEnabled)
            return "on, but idle: turn batched terrain drawing on with .vhindirect on";
        if (!packedShaderOk) return "unavailable: the packed shader variant did not compile";
        if (mirror == null || packedDrawer == null)
            return "unavailable: the regional arenas are not attached. Turn them on with .dlgpu on";
        if (packedDrawer.Failed)
            return "FAILED; cached opaque terrain is suppressed: "
                + packedDrawer.FailureReason;
        if (!mirror.RetainsPacked)
            return "on through clustered packed geometry; whole-section packed and expanded "
                + "regional copies are not retained";
        return $"on: 12-byte quads, {mirror.PackedLiveBytes / (1024.0 * 1024.0):0.0} MiB; "
            + "the expanded regional copy is not retained. "
            + $"Last frame {LastIndirectBatches} multi-draws covered {LastIndirectCommands} sections, "
            + $"{LastIndirectLeftovers} missing with no fallback";
    }

    public string DescribeClusterDraw()
    {
        LodGpuGeometryMirror? mirror = gpuShadow.Mirror;
        if (!ClusterDrawEnabled)
            return mirror == null
                ? "off"
                : "off: unselected clustered geometry is not retained";
        if (!PackedDrawEnabled || !IndirectDrawEnabled)
            return "on, but idle: the indirect and packed draw paths are off";
        if (mirror == null || clusterBuilder == null)
            return "unavailable: the regional arenas are not attached. Turn them on with .dlgpu on";
        return $"on: each section is up to {LodPackedClusterBuilder.CellCount} exact draw clusters. "
            + $"Last frame {LastIndirectCommands} cluster commands in {LastIndirectBatches} "
            + $"multi-draws, {LastIndirectLeftovers} whole sections missing with no fallback. "
            + $"Cluster geometry uses {mirror.ClusteredPackedLiveBytes / (1024.0 * 1024.0):0.0} MiB; "
            + "other regional geometry copies are not retained";
    }

    /// <summary>
    /// What depth culling is doing, in the terms someone standing in the world can act on.
    ///
    /// It reports whether the cull actually RAN last frame, not merely whether it is switched
    /// on. Those come apart constantly - no pyramid yet, the shader refused, the indirect path
    /// is off - and a switch that says "on" while nothing happens is how a person concludes a
    /// feature does nothing when it was never running.
    /// </summary>
    public string DescribeGpuCull()
    {
        if (!GpuCullEnabled) return "off: every section the CPU approves is drawn";
        if (cullPass == null)
            return "on, but idle: the regional arenas are not attached. Turn them on with .dlgpu on";
        if (!IndirectDrawEnabled)
            return "on, but idle: culling zeroes a batched draw command, and batching is off. "
                + "Turn it on with .vhindirect on";
        if (!DepthPyramidEnabled)
            return "on, but idle: there is no depth pyramid to test against. "
                + "Turn it on with .dlhzb on";
        if (!cullPass.Available)
        {
            // The two are completely different answers and used to print the same words. One
            // is a driver saying no; the other is "you asked before a single frame had been
            // drawn", which is the normal state for the instant a chat command replies.
            return cullPass.LastFailure.Length > 0
                ? "unavailable: " + cullPass.LastFailure
                : "on, but not started yet: no frame has been drawn since you switched it on. "
                    + "Play for a moment and run .vhcull again";
        }

        string ran = ActiveDrawer is { Culled: true }
            ? "last frame's commands were culled on the card"
            : "nothing was culled last frame";
        return "on: " + cullPass.Describe() + ". " + ran + ".";
    }

    public string DescribeIndirectDraw()
    {
        if (!indirectShaderOk) return "unavailable: the indirect shader variant did not compile";
        if (indirectDrawer == null || gpuShadow.Mirror == null)
            return "unavailable: the regional arenas are not attached. Turn them on with .dlgpu on";
        if (indirectDrawer.Failed) return indirectDrawer.Describe();
        if (!IndirectDrawEnabled)
            return "off: cached opaque terrain is suppressed; no per-section renderer runs";
        return "on: " + indirectDrawer.Describe()
            + $". Last frame {LastIndirectBatches} multi-draws covered {LastIndirectCommands} "
            + $"sections, {LastIndirectLeftovers} missing with no fallback. "
            + "Delayed occlusion is suspended while this is on"
            + (GpuCullEnabled
                ? "; depth culling has taken over suppression."
                : ", and nothing has replaced it. Turn on .vhcull to suppress hidden terrain, "
                    + "or leave it off to measure batching on its own.");
    }

    public int LastIndirectBatches { get; private set; }
    public int LastIndirectCommands { get; private set; }
    public int LastIndirectLeftovers { get; private set; }

    void RenderOpaqueMesh(long key, MeshRef mesh)
    {
        if (!TemporalOcclusionActive)
        {
            SubmitOpaqueMesh(key, mesh);
            return;
        }

        if (!temporalOcclusionQueries.TryGetValue(key, out TemporalOcclusionQuery? query))
        {
            query = new TemporalOcclusionQuery();
            temporalOcclusionQueries.Add(key, query);
        }

        // A mixed section is the exact vanilla/cache ownership seam: some of its
        // fragments are deliberately discarded for ready vanilla chunks and the rest
        // complete cached coverage. Never let one zero-sample result hide that entire
        // section, because the small cache-owned remainder is precisely the seam the
        // player notices. Cache-only sections beyond it retain full temporal culling.
        if (readinessMaskActive && readiness != null
            && readiness.Classify(key) == VanillaSectionOwnership.Mixed)
        {
            query.State.Invalidate(temporalOcclusionEpoch);
            LastTemporalOcclusionSeamDraws++;
            SubmitOpaqueMesh(key, mesh);
            return;
        }

        // A previous-frame hidden answer is least reliable where a fast yaw is bringing
        // terrain onto the screen. Protect only a narrow side band, and only while the
        // view is actually turning. The centre retains temporal culling, stationary views
        // pay nothing, and the extreme profile deliberately disables this guard so the
        // visual limit remains available for comparison.
        if (temporalOcclusionTurningThisFrame && temporalOcclusionEdgeGuard > 0
            && TemporalOcclusionNearSideEdge(key))
        {
            query.State.Invalidate(temporalOcclusionEpoch);
            LastTemporalOcclusionEdgeDraws++;
            SubmitOpaqueMesh(key, mesh);
            return;
        }

        int hiddenProbeInterval = temporalOcclusionTurningThisFrame
            ? Math.Min(temporalOcclusionHiddenProbeIntervalFrames,
                temporalOcclusionTurningProbeIntervalFrames)
            : temporalOcclusionHiddenProbeIntervalFrames;
        if (!query.State.ShouldDraw(frameCounter, temporalOcclusionEpoch,
            hiddenProbeInterval))
        {
            TemporalOcclusionDrawsSkipped++;
            LastTemporalOcclusionDrawsSkipped++;
            return;
        }

        if (!query.State.ShouldIssueQuery(frameCounter, temporalOcclusionEpoch,
            temporalOcclusionVisibleQueryIntervalFrames,
            hiddenProbeInterval))
        {
            SubmitOpaqueMesh(key, mesh);
            return;
        }

        // GL queries belong to the render context and cannot move to a worker. A hard
        // per-frame issue budget prevents many sections whose intervals align from all
        // adding driver commands to the same frame.
        if (temporalOcclusionQueriesIssuedThisFrame >= TemporalOcclusionQueryIssuesPerFrame)
        {
            SubmitOpaqueMesh(key, mesh);
            return;
        }

        // A renderer earlier in this stage may deliberately span a query across us.
        // Beginning the same target would fail, and blindly ending afterwards could close
        // the other renderer's query. Check once per frame and simply draw when occupied.
        if (!temporalOcclusionQueryTargetAvailable)
        {
            SubmitOpaqueMesh(key, mesh);
            return;
        }

        try
        {
            if (query.QueryId == 0) query.QueryId = GL.GenQuery();
            GL.BeginQuery(QueryTarget.AnySamplesPassed, query.QueryId);
        }
        catch (Exception e)
        {
            DisableTemporalOcclusion(e);
            // Query failure is an optimization failure, never a reason to lose terrain.
            SubmitOpaqueMesh(key, mesh);
            return;
        }

        query.State.BeginQuery(frameCounter, temporalOcclusionEpoch);
        temporalOcclusionPending.Add(key);
        temporalOcclusionQueriesIssuedThisFrame++;
        TemporalOcclusionQueries++;
        try
        {
            // This is both the ordinary visible draw and the exact visibility probe.
            // A hidden section therefore adds only two query commands, not proxy
            // geometry plus a conditional submission of the real mesh.
            SubmitOpaqueMesh(key, mesh);
        }
        finally
        {
            try { GL.EndQuery(QueryTarget.AnySamplesPassed); }
            catch (Exception e) { DisableTemporalOcclusion(e); }
        }
    }

    void SubmitOpaqueMesh(long key, MeshRef _)
    {
        if (liveMeshStats.TryGetValue(key, out LodLiveMeshStats stats))
        {
            OpaqueDrawVertices += stats.OpaqueVertices;
            OpaqueDrawIndices += stats.OpaqueIndices;
        }

        // In the required packed pass a section is submitted by writing its command; the
        // draws happen once, after the walk. A section the arenas do not hold is recorded
        // as missing and stays missing instead of silently changing renderer.
        bool recorded = RecordShadowCommand(key, out bool farBucket);
        if (indirectPassActive)
        {
            List<long> target = farBucket ? indirectFarRecorded : indirectRecorded;
            List<long> missing = farBucket ? indirectFarLeftovers : indirectLeftovers;
            if (recorded) target.Add(key);
            else missing.Add(key);
            return;
        }

        ReportRequiredTerrainFailure("a per-section opaque draw was requested outside the packed pass");
    }

    void SubmitWaterMesh(long key, MeshRef mesh)
    {
        WaterDrawCalls++;
        if (liveMeshStats.TryGetValue(key, out LodLiveMeshStats stats))
        {
            WaterDrawVertices += stats.WaterVertices;
            WaterDrawIndices += stats.WaterIndices;
        }
        capi.Render.RenderMesh(mesh);
    }

    void ResolveTemporalOcclusionQueries()
    {
        if (!TemporalOcclusionActive || temporalOcclusionPending.Count == 0) return;

        try
        {
            int checkedThisFrame = 0;
            for (int i = temporalOcclusionPending.Count - 1;
                i >= 0 && checkedThisFrame < TemporalOcclusionResultChecksPerFrame;
                i--, checkedThisFrame++)
            {
                long key = temporalOcclusionPending[i];
                if (!temporalOcclusionQueries.TryGetValue(key, out TemporalOcclusionQuery? query)
                    || query.QueryId == 0)
                {
                    temporalOcclusionPending.RemoveAt(i);
                    continue;
                }

                GL.GetQueryObject(query.QueryId, GetQueryObjectParam.QueryResultAvailable,
                    out int available);
                if (available == 0) continue;

                GL.GetQueryObject(query.QueryId, GetQueryObjectParam.QueryResult, out int samples);
                if (query.State.CompleteQuery(samples != 0, temporalOcclusionEpoch))
                {
                    TemporalOcclusionResults++;
                    if (samples == 0) TemporalOcclusionHiddenResults++;
                }
                else
                {
                    TemporalOcclusionStaleResults++;
                }
                temporalOcclusionPending.RemoveAt(i);
            }
        }
        catch (Exception e)
        {
            DisableTemporalOcclusion(e);
        }
    }

    bool TemporalOcclusionNearSideEdge(long key)
    {
        int footprint = LodWorld.KeyFootprintBlocks(key);
        double relX = LodWorld.KeySx(key) * (double)footprint - camPos.X;
        double relZ = LodWorld.KeySz(key) * (double)footprint - camPos.Z;
        return frustum.BoxNearSideEdge(
            relX, -camPos.Y, relZ,
            relX + footprint, worldHeight - camPos.Y, relZ + footprint,
            temporalOcclusionEdgeGuard);
    }

    void PrepareTemporalOcclusionQueries()
    {
        temporalOcclusionQueriesIssuedThisFrame = 0;
        temporalOcclusionQueryTargetAvailable = false;
        if (!TemporalOcclusionActive) return;

        try
        {
            GL.GetQuery(QueryTarget.AnySamplesPassed, GetQueryParam.CurrentQuery,
                out int activeQuery);
            temporalOcclusionQueryTargetAvailable = activeQuery == 0;
        }
        catch (Exception e)
        {
            DisableTemporalOcclusion(e);
        }
    }

    void UpdateTemporalOcclusionView(float[] view, float[] projection,
        double cameraX, double cameraY, double cameraZ)
    {
        temporalOcclusionTurningThisFrame = temporalOcclusionHasPreviousFrameView
            && ViewRotationChanged(view, temporalOcclusionPreviousFrameView);
        Array.Copy(view, temporalOcclusionPreviousFrameView, 16);
        temporalOcclusionHasPreviousFrameView = true;

        bool changed = !temporalOcclusionHasView
            || TranslationExceeded(cameraX, cameraY, cameraZ,
                temporalOcclusionCameraX, temporalOcclusionCameraY, temporalOcclusionCameraZ,
                temporalOcclusionTranslationLimitBlocks)
            || ViewRotationExceeded(view, temporalOcclusionView,
                temporalOcclusionRotationMatrixLimit)
            || ProjectionChanged(projection, temporalOcclusionProjection);
        if (!changed) return;

        Array.Copy(view, temporalOcclusionView, 16);
        Array.Copy(projection, temporalOcclusionProjection, 16);
        temporalOcclusionCameraX = cameraX;
        temporalOcclusionCameraY = cameraY;
        temporalOcclusionCameraZ = cameraZ;
        temporalOcclusionHasView = true;
        InvalidateTemporalOcclusionScene();
    }

    internal static bool TranslationExceeded(double x, double y, double z,
        double anchorX, double anchorY, double anchorZ, double limit)
    {
        double dx = x - anchorX;
        double dy = y - anchorY;
        double dz = z - anchorZ;
        return dx * dx + dy * dy + dz * dz >= limit * limit;
    }

    internal static bool ViewRotationExceeded(float[] current, float[] previous, float limit)
    {
        if (current.Length < 16) return true;
        // CameraMatrixOriginf has no useful world translation for cached geometry. Only
        // compare the 3x3 orientation basis; translation has its own block-space bound.
        ReadOnlySpan<int> rotation = [0, 1, 2, 4, 5, 6, 8, 9, 10];
        foreach (int i in rotation)
        {
            if (Math.Abs(current[i] - previous[i])
                >= limit) return true;
        }
        return false;
    }

    internal static bool ViewRotationChanged(float[] current, float[] previous)
    {
        if (current.Length < 16) return true;
        ReadOnlySpan<int> rotation = [0, 1, 2, 4, 5, 6, 8, 9, 10];
        foreach (int i in rotation)
        {
            if (BitConverter.SingleToInt32Bits(current[i])
                != BitConverter.SingleToInt32Bits(previous[i])) return true;
        }
        return false;
    }

    internal static bool ProjectionChanged(float[] current, float[] previous)
    {
        if (current.Length < 16) return true;
        for (int i = 0; i < 16; i++)
        {
            if (BitConverter.SingleToInt32Bits(current[i])
                != BitConverter.SingleToInt32Bits(previous[i])) return true;
        }
        return false;
    }

    void InvalidateTemporalOcclusionScene()
    {
        TemporalOcclusionGlobalInvalidations++;
        temporalOcclusionEpoch++;
        if (temporalOcclusionEpoch == long.MaxValue)
        {
            temporalOcclusionEpoch = 1;
            foreach (TemporalOcclusionQuery query in temporalOcclusionQueries.Values)
                query.State.Invalidate(temporalOcclusionEpoch);
        }
    }

    void DisableTemporalOcclusion(Exception error)
    {
        temporalOcclusionFailed = true;
        if (temporalOcclusionFailureReported) return;
        temporalOcclusionFailureReported = true;
        capi.Logger.Warning(
            "[Landscapes] delayed occlusion disabled; all cached terrain remains visible: {0}",
            error.Message);
    }

    void DisposeTemporalOcclusionQueries()
    {
        foreach (TemporalOcclusionQuery query in temporalOcclusionQueries.Values)
        {
            if (query.QueryId == 0) continue;
            try { GL.DeleteQuery(query.QueryId); }
            catch { /* teardown must not throw over a query the driver already released */ }
        }
        temporalOcclusionQueries.Clear();
        temporalOcclusionPending.Clear();
        temporalOcclusionHasView = false;
        temporalOcclusionHasPreviousFrameView = false;
        temporalOcclusionTurningThisFrame = false;
    }

    void RemoveTemporalOcclusionQuery(long key)
    {
        if (!temporalOcclusionQueries.Remove(key, out TemporalOcclusionQuery? query)) return;
        temporalOcclusionPending.Remove(key);
        if (query.QueryId == 0) return;
        try { GL.DeleteQuery(query.QueryId); }
        catch { /* eviction must not fail over an optional driver object */ }
    }

    readonly Dictionary<long, float> oceanityBySection = new();

    /// <summary>
    /// The game's own oceanity where this section sits, 0 inland to 1 open sea.
    ///
    /// It sets the wave amplitude and cannot be guessed: chunkliquid.vsh divides the warp by
    /// <c>5 * max(0.2, 1 - oceanity)</c>, so a pond moves 0.10 blocks where the open sea moves
    /// up to 0.96. Read from `IMapRegion.OceanMap` through the engine's own lerped accessor, so
    /// it is the same field the engine's tesselator bilerps into its vertex flags.
    ///
    /// Cached per section: it is a worldgen property and never changes, and an uncached miss
    /// answers 0, which is the calm end and the safe direction to be wrong in.
    /// </summary>
    float SectionOceanity(long key, double originX, double originZ, int footprint)
    {
        if (oceanityBySection.TryGetValue(key, out float known)) return known;

        float oceanity = 0f;
        try
        {
            int blockX = (int)originX + footprint / 2;
            int blockZ = (int)originZ + footprint / 2;
            int regionSize = capi.World.BlockAccessor.RegionSize;
            // The two-int overload deliberately. The Vec2i one is banned repo-wide because
            // the same-named uniform overload reaches glUniform2f and silently breaks integer
            // uniforms (G42), and the check that enforces it matches the token, not the call.
            IMapChunk? mapChunk = capi.World.BlockAccessor.GetMapChunk(
                blockX / VanillaReadinessMask.ChunkBlocks,
                blockZ / VanillaReadinessMask.ChunkBlocks);
            IntDataMap2D? ocean = mapChunk?.MapRegion?.OceanMap;
            if (ocean != null && ocean.Data != null && ocean.Data.Length > 0)
            {
                oceanity = Math.Clamp(
                    ocean.GetUnpaddedIntLerpedForBlockPos(blockX, blockZ, regionSize) / 255f,
                    0f, 1f);
                oceanityBySection[key] = oceanity;
            }
        }
        catch
        {
            // A map region that is not loaded answers nothing; calm water is the safe answer
            // and the next frame asks again rather than caching the miss.
        }
        return oceanity;
    }

    bool SetupSectionTransform(long key, float cullDistSq, bool waterPass = false)
    {
        int footprint = LodWorld.KeyFootprintBlocks(key);
        double originX = LodWorld.KeySx(key) * (double)footprint;
        double originZ = LodWorld.KeySz(key) * (double)footprint;
        if (waterPass)
        {
            prog!.Uniform("vhOceanity", SectionOceanity(key, originX, originZ, footprint));
        }

        double dx = originX + footprint / 2.0 - camPos.X;
        double dz = originZ + footprint / 2.0 - camPos.Z;
        if (dx * dx + dz * dz > cullDistSq) return false;
        float horizontalDistance = (float)Math.Sqrt(dx * dx + dz * dz);

        // Camera-relative box, matching the model matrix below. The Y range is the mesh's
        // own, from the pass being drawn: opaque and water are submitted separately and a
        // shoreline's water surface sits nowhere near the seabed under it. A section whose
        // bounds are unknown - anything published without them - keeps the old
        // bedrock-to-sky box, so a missing measurement can only ever draw too much.
        double relX = originX - camPos.X;
        double relZ = originZ - camPos.Z;
        double boxMinY = -camPos.Y;
        double boxMaxY = worldHeight - camPos.Y;
        LodHeightSpan span = SectionSpan(key, waterPass);
        if (span.HasGeometry)
        {
            boxMinY = span.MinY - camPos.Y;
            boxMaxY = span.MaxY - camPos.Y;
        }

        if (!frustum.BoxInView(relX, boxMinY, relZ, relX + footprint, boxMaxY, relZ + footprint))
        {
            culledThisFrame++;

            // Only for sections the real box rejected: how many of them the old
            // full-height box would have kept. That figure is the whole justification for
            // this work, and inferring it from a frame rate is not available.
            if (span.HasGeometry
                && frustum.BoxInView(relX, -camPos.Y, relZ,
                    relX + footprint, worldHeight - camPos.Y, relZ + footprint))
            {
                verticalCulledThisFrame++;
            }
            return false;
        }

        // Recorded once the section has survived every cheaper rejection - distance and
        // frustum - so the population is exactly the sections this renderer would draw.
        // Collected before the ownership skip and the occlusion skip further down, because
        // the question being measured is what DEPTH alone would remove, and mixing in
        // sections another rule already dropped would flatter the answer.
        //
        // Opaque only: the water pass walks a subset of the same sections and would count
        // them twice, and water is not an occluder.
        if (hzbCollecting && !waterPass
            && (!depthSplitThisFrame || !LodGpuDepthSplitPolicy.IsNear(horizontalDistance)))
        {
            hzbBoxes.Add(new LodHzbSectionBox(
                key,
                (float)relX, (float)boxMinY, (float)relZ,
                (float)(relX + footprint), (float)boxMaxY, (float)(relZ + footprint),
                horizontalDistance));
        }

        // Sides that border on never-captured area, so the shader can dissolve them
        // into the horizon instead of leaving a cliff at the edge of what we've seen.
        byte openEdges = (byte)(
            (HasNeighbourData(key, -1, 0) ? 0 : LodGpuSectionFacts.OpenMinusX)
            | (HasNeighbourData(key, 1, 0) ? 0 : LodGpuSectionFacts.OpenPlusX)
            | (HasNeighbourData(key, 0, -1) ? 0 : LodGpuSectionFacts.OpenMinusZ)
            | (HasNeighbourData(key, 0, 1) ? 0 : LodGpuSectionFacts.OpenPlusZ));

        // Six uniform uploads per section, and they are most of what the indirect path
        // exists to remove. During an indirect walk they are skipped: the same values go
        // into the section record instead. A section that ends up in the leftover pass
        // comes back through here with the flag clear and is set up normally.
        if (!indirectPassActive)
        {
            modelMat.Identity().Translate(relX, -camPos.Y, relZ);
            prog!.UniformMatrix("modelMatrix", modelMat.Values);
            prog.Uniform("columnBlocks", (float)LodWorld.ColumnStepBlocks(LodWorld.KeyLevel(key)));

            // The quarters a stand-in covers, for the passes drawn section by section - water,
            // thin cover and their depth and OIT companions. Without it a stand-in's water would
            // lie over the finer water its ready quarters already draw. Set for every section,
            // zero for ordinary ones, because a uniform carries its last value into the next draw.
            // Opaque ground takes no uniform: its clusters are selected before they are drawn.
            prog.Uniform("vhStandInQuarters", StandInQuartersFor(key));

            // Projection/fog use camera-relative coordinates, but colour noise needs the
            // section's stable world origin or a fixed patch changes colour while flying.
            // This uniform does not affect geometry, so float precision at extreme world
            // coordinates can only soften the cosmetic variation, never move terrain.
            prog.Uniform("noiseOrigin", (float)originX, (float)originZ, 0f, 0f);

            // Ownership addressing uses this integer origin plus the section-local
            // position, never a summed world coordinate, so a fragment at a chunk edge
            // cannot round onto its neighbour's ownership at large world coordinates.
            //
            // Set as two integer uniforms, never as one Vec2i: that overload reaches
            // glUniform2f, which an integer uniform rejects with GL_INVALID_OPERATION, so
            // the origin silently stayed at zero and the per-fragment mask addressed the
            // wrong cells for every section in the world. See G42.
            if (readinessMaskActive)
            {
                prog.Uniform("maskSectionOriginX",
                    (int)(originX / VanillaReadinessMask.ChunkBlocks));
                prog.Uniform("maskSectionOriginZ",
                    (int)(originZ / VanillaReadinessMask.ChunkBlocks));
            }

            prog.Uniform("sectionSize", (float)footprint);
            prog.Uniform("openEdges",
                (openEdges & LodGpuSectionFacts.OpenMinusX) != 0 ? 1f : 0f,
                (openEdges & LodGpuSectionFacts.OpenPlusX) != 0 ? 1f : 0f,
                (openEdges & LodGpuSectionFacts.OpenMinusZ) != 0 ? 1f : 0f,
                (openEdges & LodGpuSectionFacts.OpenPlusZ) != 0 ? 1f : 0f);
        }

        // The same values a multi-draw reads from the record buffer instead. Captured
        // here so the command list is built from the traversal's real decisions, not from
        // a second, possibly disagreeing, walk of the draw list.
        if (shadowBuilder != null)
        {
            lastSectionHorizontalDistance = horizontalDistance;
            lastSectionFacts = new LodGpuSectionFacts(
                key,
                (float)relX, (float)-camPos.Y, (float)relZ,
                footprint,
                (float)originX, (float)originZ,
                LodWorld.ColumnStepBlocks(LodWorld.KeyLevel(key)),
                (int)(originX / VanillaReadinessMask.ChunkBlocks),
                (int)(originZ / VanillaReadinessMask.ChunkBlocks),
                openEdges,
                // The cull box carried alongside, taken from the same locals the frustum
                // test above just used rather than recomputed. A second derivation could
                // disagree with the first by a mesh span the pass no longer has, and a box
                // that disagrees with the one the CPU judged is how the two paths end up
                // drawing different terrain.
                (float)boxMinY,
                (float)boxMaxY);
        }
        return true;
    }

    LodGpuIndirectBuilder? shadowBuilder;
    LodGpuIndirectBuilder? splitFarBuilder;
    LodGpuIndirectBuilder? packedBuilder;
    LodGpuIndirectBuilder? packedFarBuilder;
    LodGpuIndirectBuilder? clusterBuilder;
    LodGpuIndirectBuilder? clusterFarBuilder;

    LodGpuIndirectBuilder? ActiveNearBuilder =>
        clusterDrawingThisFrame ? clusterBuilder
        : packedDrawingThisFrame ? packedBuilder
        : shadowBuilder;

    LodGpuIndirectBuilder? ActiveFarBuilder =>
        clusterDrawingThisFrame ? clusterFarBuilder
        : packedDrawingThisFrame ? packedFarBuilder
        : splitFarBuilder;

    LodGpuIndirectDrawer? ActiveDrawer =>
        packedDrawingThisFrame ? packedDrawer : indirectDrawer;

    IShaderProgram? ActiveIndirectProgram =>
        packedDrawingThisFrame ? packedProg : indirectProg;
    LodGpuSectionFacts lastSectionFacts;
    float lastSectionHorizontalDistance;

    /// <summary>
    /// Records what a regional multi-draw would have submitted for a section the legacy
    /// path is drawing right now. It runs after every CPU decision the visible path makes -
    /// ownership skip, distance cap, frustum, temporal occlusion - so the command list is
    /// the one Phase 3 would actually issue, and the batch count it reports is the real
    /// answer to how far draw calls would fall.
    /// </summary>
    bool RecordShadowCommand(long key, out bool farBucket)
    {
        farBucket = false;

        // Ordered so the visible path pays one null field read per drawn section when the
        // shadow is off, which is the ordinary case.
        LodGpuIndirectBuilder? nearBuilder = ActiveNearBuilder;
        if (nearBuilder == null) return false;

        // Reject before even choosing a bucket. A stale distance is no safer than stale
        // box facts, and could otherwise put the missing command on the wrong side of the rebuild.
        if (lastSectionFacts.SectionKey != key)
        {
            nearBuilder.AddMissing();
            return false;
        }

        farBucket = depthSplitThisFrame
            && !LodGpuDepthSplitPolicy.IsNear(lastSectionHorizontalDistance);
        LodGpuIndirectBuilder? farBuilder = ActiveFarBuilder;
        LodGpuIndirectBuilder builder = farBucket && farBuilder != null
            ? farBuilder
            : nearBuilder;
        LodGpuGeometryMirror? mirror = gpuShadow.Mirror;
        if (mirror == null) return false;
        if (!mirror.TryGet(key, out LodGpuGeometryMirror.MirroredSection section))
        {
            // Drawn, but the arenas never managed to hold it. Counted rather than ignored:
            // batches over a partial mirror are not a draw-call reduction, and silence here
            // is exactly what made the first measured run look better than it was.
            builder.AddMissing();
            return false;
        }
        // The facts were written by the SetupSectionTransform call for this same section, one
        // step earlier in the walk. That coupling is ordering alone, and it used to be cheap
        // to get wrong: a mismatch skewed a measurement nobody was acting on. It is not cheap
        // now. The cull tests THIS record's box and zeroes THIS command, so stale facts would
        // hide a section on the strength of a different section's position, and the symptom
        // would be scattered terrain missing in some views - the exact fault this renderer
        // has already spent two sessions chasing once.
        //
        // A refused packed span must enter this frame's missing list. It must not disappear
        // from accounting merely because an unused expanded mirror record exists.
        //
        // A stand-in covering only some quarters submits only their clusters. Clusters are cut
        // exactly at quarter edges, so this is that quarter's geometry and nothing else.
        return builder.Add(section, lastSectionFacts, quarters: StandInQuartersFor(key));
    }

    int culledThisFrame;
    int traversalCulledThisFrame;
    int subtreeCulledThisFrame;
    int subtreeCulledMeshesThisFrame;
    int verticalCulledThisFrame;

    /// <summary>
    /// Sections rejected this interval by the mesh's own height that the old
    /// bedrock-to-sky box would have kept, summed over frames, and the worst single frame.
    /// This is the established renderer's share of Phase 3b, and it is a saving whether or
    /// not the GPU fast path is ever adopted.
    /// </summary>
    public long VerticalCulledSections { get; private set; }
    public int VerticalCulledMax { get; private set; }
    public int LastVerticalCulledCount { get; private set; }

    /// <summary>Quadtree nodes rejected this interval that a full-height box would have traversed.</summary>
    public long SubtreeCulledNodes { get; private set; }

    /// <summary>Resident meshes those rejected subtrees held, which is what the rejection saved.</summary>
    public long SubtreeCulledMeshes { get; private set; }
    public int SubtreeCulledMax { get; private set; }
    public int LastSubtreeCulledCount { get; private set; }
    public int SubtreeBoundsNodes => subtreeHeights.TrackedNodes;

    public LodSectionHeightStats SectionHeights => sectionHeightStats;
    readonly LodSectionHeightStats sectionHeightStats = new();

    void NoteSectionHeights(LodSectionHeights heights) =>
        sectionHeightStats.Add(heights.Either);

    /// <summary>
    /// The vertical extent of the geometry this pass would draw for the section, or an
    /// empty span when nothing recorded it. Read from the live mesh record the publication
    /// already maintains, so no second dictionary has to be kept in step with residency.
    /// </summary>
    /// <summary>
    /// Copies this frame's depth and reduces it into the pyramid. Nothing reads the result
    /// yet: this phase exists to establish what the pyramid COSTS, because Phase 4's gate is
    /// that copy and build time stay under the drawing they could remove, and no amount of
    /// source reading answers that.
    ///
    /// Failure is not an error state. A frame that could not build one simply has no
    /// pyramid, and a later phase with no pyramid must draw everything - the same direction
    /// every other fallback in this renderer fails in.
    /// </summary>
    void BuildDepthPyramid(bool splitMid = false)
    {
        if (!DepthPyramidEnabled || !hzbShaderOk) return;

        LodPhaseStart phase = LodPhaseCost.Start(TrackPhaseAllocations);
        bool timing = false;
        try
        {
            if (gpuFailureInjection.Take(LodGpuFailureStage.DepthCopy))
                throw new InvalidOperationException("Phase 9 injected a depth-copy failure");

            depthPyramid ??= new LodGpuDepthPyramid(capi, () => hzbShaderOk ? hzbProg : null);

            // Probed per frame rather than cached: the engine is free to change which
            // framebuffer it draws into, and a pyramid copied out of last frame's
            // attachment would be wrong in exactly the way nobody would notice.
            LodGpuDepthFacts depth = LodGpuCapabilities.ProbeActiveDepth();
            (int frameWidth, int frameHeight) = GpuFrameSize();

            timing = splitMid
                ? gpuTelemetry.BeginSplitHzb()
                : gpuTelemetry.BeginHzb();
            LastHzbBuild = depthPyramid.Build(depth, frameWidth, frameHeight);

            if (!LastHzbBuild.Built && !reportedHzbFailure)
            {
                reportedHzbFailure = true;
                capi.Logger.Warning(
                    "[Landscapes] the depth pyramid could not be built and nothing "
                    + "depends on it yet: {0}", LastHzbBuild.FailureReason);
            }
            else if (LastHzbBuild.Built && !reportedHzbBuilt)
            {
                reportedHzbBuilt = true;
                capi.Logger.Notification(
                    "[Landscapes] depth pyramid built: {0}x{1}, {2} levels.",
                    LastHzbBuild.Width, LastHzbBuild.Height, LastHzbBuild.Levels);
            }
        }
        catch (Exception e)
        {
            LastHzbBuild = default;
            DepthPyramidEnabled = false;
            capi.Logger.Warning(
                "[Landscapes] depth pyramid disabled for this session; rendering is "
                + "unchanged: {0}", e.Message);
        }
        finally
        {
            // A build that did not happen - a minimised window, a refused depth attachment -
            // must not leave a near-zero sample in an average of real builds.
            if (timing)
            {
                if (splitMid)
                {
                    if (LastHzbBuild.Built) gpuTelemetry.EndSplitHzb();
                    else gpuTelemetry.DiscardSplitHzb();
                }
                else
                {
                    if (LastHzbBuild.Built) gpuTelemetry.EndHzb();
                    else gpuTelemetry.DiscardHzb();
                }
            }
            HzbCost.Add(phase);
        }
    }

    bool reportedHzbFailure;
    bool reportedHzbBuilt;
    bool reportedHzbClassifier;

    LodHzbClassifier? hzbClassifier;
    Action<long, uint>? hzbVerdictObserver;
    long hzbDispatchEpoch = -1;
    long hzbVerdictEpoch = -1;
    bool hzbCollecting;

    /// <summary>
    /// Sections the pyramid called hidden that a completed occlusion query had positively
    /// SEEN. This is the number Phase 4's correctness gate is about, and it must be zero:
    /// every one of these is terrain a player could see that a depth test would have
    /// deleted. Counted rather than trusted, because the shader cannot be reasoned into
    /// being right about this and the symptom - terrain missing in some views - is exactly
    /// what this renderer has already lost sessions to.
    /// </summary>
    public long HzbHiddenButQuerySawIt { get; private set; }

    /// <summary>
    /// The safe disagreement: the query proved it hidden and the pyramid did not. Nothing
    /// is wrong with these, and their count is the honest measure of how much the pyramid
    /// is leaving on the table against a test that observes real pixels.
    /// </summary>
    public long HzbMissedWhatQueryHid { get; private set; }

    /// <summary>Verdicts that could be checked against a current query result at all.</summary>
    public long HzbVerdictsChecked { get; private set; }

    /// <summary>
    /// Sections the pyramid hid that the existing delayed occlusion had no opinion on.
    /// THIS is the phase's real prize: the renderer already skips much of what the pyramid
    /// finds, so an overall hidden percentage flatters it badly. What is genuinely new is
    /// only what nothing else had measured.
    /// </summary>
    public long HzbHiddenBeyondQueries { get; private set; }

    /// <summary>Both agreed it was hidden - correct, but not a new saving.</summary>
    public long HzbAgreedHidden { get; private set; }

    /// <summary>Comparisons refused because the two answers came from different views.</summary>
    public long HzbVerdictsAcrossViewChange { get; private set; }

    /// <summary>
    /// How stale an occlusion-query answer may be and still be treated as describing the
    /// same view as a pyramid verdict. Four frames: the verdict is already one frame old by
    /// the time it is read, and at 400 fps four frames is 10 ms, which is short enough that
    /// the camera cannot have moved far and long enough that ordinary query cadence still
    /// supplies comparisons. This bounds the comparison, not the renderer - a refused
    /// comparison costs a sample, never a draw.
    /// </summary>
    const int HzbComparisonMaxQueryAgeFrames = 4;

    /// <summary>
    /// Checks one verdict against the delayed occlusion query for the same section.
    ///
    /// Only current, completed query results count as evidence. A query that is still
    /// pending, or whose answer was thrown away by a view change or a mesh replacement,
    /// says nothing - and treating its silence as "visible" would invent disagreements
    /// that are really just missing data.
    /// </summary>
    // Last verdict per section, kept only so a person can ask about one. Cleared with the
    // interval, so it can never grow past the sections a single reporting window touched.
    readonly Dictionary<long, uint> hzbLastVerdict = new();

    void ObserveHzbVerdict(long key, uint verdict)
    {
        hzbLastVerdict[key] = verdict;

        if (!temporalOcclusionQueries.TryGetValue(key, out TemporalOcclusionQuery? query)) return;

        // Both answers have to describe the same view, and the occlusion epoch is too coarse
        // a test for that: it advances on a global invalidation, not on ordinary mouse
        // movement. Measured - a run with ZERO epoch changes still produced seven sections
        // the pyramid called hidden that a query had seen. So the age of the query's own
        // answer is checked as well: a result taken many frames ago describes a camera that
        // has since moved, whatever the epoch says.
        if (hzbVerdictEpoch != temporalOcclusionEpoch)
        {
            HzbVerdictsAcrossViewChange++;
            return;
        }

        long queryAge = frameCounter - query.State.LastQueryFrame;
        if (queryAge < 0 || queryAge > HzbComparisonMaxQueryAgeFrames)
        {
            HzbVerdictsAcrossViewChange++;
            return;
        }

        bool queryKnowsVisible = query.State.KnownVisible;
        bool queryKnowsHidden = query.State.Occluded;
        if (!queryKnowsVisible && !queryKnowsHidden)
        {
            // No query opinion at all. This is where the pyramid earns its keep over the
            // existing delayed occlusion: a section it can hide that nothing has measured
            // yet is a saving the current renderer does not already have.
            if (verdict == LodHzbClassifier.VerdictOccluded) HzbHiddenBeyondQueries++;
            return;
        }

        HzbVerdictsChecked++;

        if (verdict == LodHzbClassifier.VerdictOccluded && queryKnowsVisible)
        {
            HzbHiddenButQuerySawIt++;
            return;
        }
        if (verdict == LodHzbClassifier.VerdictOccluded && queryKnowsHidden)
        {
            HzbAgreedHidden++;
            return;
        }
        if (verdict != LodHzbClassifier.VerdictOccluded && queryKnowsHidden)
        {
            HzbMissedWhatQueryHid++;
        }
    }
    readonly List<LodHzbSectionBox> hzbBoxes = new(1024);
    readonly float[] hzbViewProjection = new float[16];

    /// <summary>Shadow-mode verdicts for the interval. Nothing drawn depends on them.</summary>
    internal LodHzbVerdicts HzbVerdicts => hzbClassifier?.Interval ?? default;

    /// <summary>
    /// Hands this frame's section boxes to the card and collects last frame's verdicts.
    ///
    /// Runs after the opaque walk, so the boxes are the ones the renderer actually decided
    /// on rather than a second walk's reconstruction of them, and after the pyramid exists.
    /// The verdicts are counted and nothing else: the established path has already drawn
    /// every one of these sections by the time the answer arrives.
    /// </summary>

    /// <summary>
    /// Creates the classifier and its compute program, once, outside any timed phase.
    /// Returns false while it is unavailable, which costs the measurement and nothing else.
    /// </summary>
    bool PrepareClassifier()
    {
        if (hzbClassifier == null)
        {
            hzbClassifier = new LodHzbClassifier(
                message => capi.Logger.Warning("{0}", message));
            hzbVerdictObserver ??= ObserveHzbVerdict;
            hzbClassifier.VerdictObserver = hzbVerdictObserver;
        }

        if (!hzbClassifier.Available && !hzbClassifier.TryCreate())
        {
            if (!reportedHzbClassifier)
            {
                reportedHzbClassifier = true;
                capi.Logger.Warning(
                    "[Landscapes] depth-pyramid classification is unavailable and "
                    + "nothing that draws depends on it: {0}", hzbClassifier.LastFailure);
            }
            return false;
        }

        if (!reportedHzbClassifier)
        {
            reportedHzbClassifier = true;
            capi.Logger.Notification(
                "[Landscapes] depth-pyramid classification running in shadow: it "
                + "counts what a depth test would hide and hides nothing.");
        }

        return true;
    }

    void ClassifyAgainstDepthPyramid()
    {
        if (!DepthPyramidEnabled || depthPyramid == null || !depthPyramid.Allocated) return;
        if (!LastHzbBuild.Built) return;

        // Built before the clock starts. Compiling and linking a compute program is a
        // one-time cost of about 160 ms on this driver, and timing it as part of the phase
        // reported a 160,690 us maximum for work that happens once and never again - which
        // is worse than useless in a figure whose whole job is to say what the pyramid
        // costs every frame.
        if (!PrepareClassifier()) return;

        LodPhaseStart phase = LodPhaseCost.Start(TrackPhaseAllocations);
        bool timing = false;
        try
        {
            // Timed apart from the pyramid build, because the two scale with different
            // things and only this one scales with the sampling width. Started after
            // PrepareClassifier for the same reason the build is: a one-time shader compile
            // inside the clock reports a per-frame cost that never happens again.
            timing = gpuTelemetry.BeginClassify();

            // The epoch these boxes were projected under. Verdicts arrive a frame later and
            // are compared against occlusion queries whose answers carry their own epoch;
            // without this the two can be from different views, and a camera turn between
            // them manufactures disagreements that are really just a mismatch. Measured:
            // zero unsafe disagreements over 2.9 million stationary checks, then 46 the
            // moment the camera turned.
            // The batch about to be READ was dispatched under the previous epoch; the one
            // about to be dispatched belongs to this one. Swapped in that order because
            // Classify reads before it dispatches.
            // Ahead of the epoch swap, because the swap is a hand-off: it says the batch about
            // to be dispatched belongs to this epoch. Advancing it and then not dispatching
            // would pair the next frame's verdicts with the wrong epoch entirely.
            frustum.CopyViewProjection(hzbViewProjection);

            hzbVerdictEpoch = hzbDispatchEpoch;
            hzbDispatchEpoch = temporalOcclusionEpoch;
            hzbClassifier!.Classify(
                hzbBoxes,
                hzbViewProjection,
                depthPyramid.TextureName,
                depthPyramid.Width,
                depthPyramid.Height,
                depthPyramid.Levels);
        }
        catch (Exception e)
        {
            capi.Logger.Warning(
                "[Landscapes] depth-pyramid classification stopped; rendering is "
                + "unchanged: {0}", e.Message);
            hzbClassifier?.Dispose();
            hzbClassifier = null;
        }
        finally
        {
            if (timing) gpuTelemetry.EndClassify();
            HzbCost.Add(phase);
        }
    }


    /// <summary>
    /// Look at a piece of distant terrain and ask why the depth test did or did not hide
    /// it.
    ///
    /// The counters say how often something happens; this says why it happened HERE, which
    /// is the only form a person can check against what is on their screen. It reports the
    /// card's verdict beside the CPU's own projection of the same box - and because the two
    /// implement the same rules, the CPU half can name the specific reason the card could
    /// only encode as a number.
    /// </summary>
    public string ExplainDepthPyramid(double startX, double startY, double startZ,
        float lookX, float lookY, float lookZ, int maxBlocks)
    {
        if (!hzbShaderOk) return "the hzbreduce shader is not available, so no pyramid exists";
        if (!DepthPyramidEnabled) return "the depth pyramid is off. Turn it on with .dlhzb on";
        if (depthPyramid == null || !depthPyramid.Allocated) return "no pyramid has been built yet";

        // Its own array, and not the one the cull writes: this runs on the chat thread while
        // the render thread is using that one. Filled once, and every projection below reads
        // it, so the whole explanation describes a single picture rather than drifting between
        // frames as it walks.
        frustum.CopyViewProjection(explainViewProjection);

        long key = 0;
        bool found = false;
        double hitDistance = 0;

        // A section adjacent to the camera has corners behind the camera plane, so the only
        // answer it can ever give is "declined". The first two attempts at this command both
        // landed on one and told nobody anything. Keep the first such section as a fallback,
        // then keep walking for one the test can actually judge.
        long fallbackKey = 0;
        bool haveFallback = false;
        double fallbackDistance = 0;

        for (int distance = 32; distance <= maxBlocks && !found; distance += 32)
        {
            double x = startX + lookX * distance;
            double y = startY + lookY * distance;
            double z = startZ + lookZ * distance;
            if (y < 0 || y >= worldHeight) break;

            // Finest first. A point sits inside sections at every level at once, and the
            // one being drawn there is the finest that holds a mesh - coarser parents exist
            // over the same ground and stop being drawn once their children are ready.
            // Searching coarsest-first reported a 2048-block L5 section 32 blocks away,
            // which was simply the largest box containing the camera and told nobody
            // anything about what they were looking at.
            for (int level = 0; level <= LodWorld.MaxLevel && !found; level++)
            {
                int footprint = LodSection.SectionBlocks << level;
                long candidate = LodWorld.SectionKey(
                    level, (int)Math.Floor(x / footprint), (int)Math.Floor(z / footprint));
                if (!liveMeshStats.ContainsKey(candidate)) continue;

                // Never the section the camera is standing in: its box encloses the near
                // plane, so the only thing it can ever report is that the test declined.
                double sectionOriginX = LodWorld.KeySx(candidate) * (double)footprint;
                double sectionOriginZ = LodWorld.KeySz(candidate) * (double)footprint;
                if (camPos.X >= sectionOriginX && camPos.X < sectionOriginX + footprint
                    && camPos.Z >= sectionOriginZ && camPos.Z < sectionOriginZ + footprint)
                {
                    continue;
                }

                // Judgeable? Ask before settling on it.
                int candidateSize = LodWorld.KeyFootprintBlocks(candidate);
                double candidateOriginX2 = LodWorld.KeySx(candidate) * (double)candidateSize;
                double candidateOriginZ2 = LodWorld.KeySz(candidate) * (double)candidateSize;
                LodHeightSpan candidateSpan = SectionSpan(candidate, waterPass: false);
                double candidateMinY = candidateSpan.HasGeometry
                    ? candidateSpan.MinY - camPos.Y : -camPos.Y;
                double candidateMaxY = candidateSpan.HasGeometry
                    ? candidateSpan.MaxY - camPos.Y : worldHeight - camPos.Y;

                LodHzbScreenBounds probe = LodHzbProjection.Project(
                    explainViewProjection,
                    candidateOriginX2 - camPos.X, candidateMinY, candidateOriginZ2 - camPos.Z,
                    candidateOriginX2 - camPos.X + candidateSize, candidateMaxY,
                    candidateOriginZ2 - camPos.Z + candidateSize);

                if (!probe.Usable)
                {
                    if (!haveFallback)
                    {
                        fallbackKey = candidate;
                        fallbackDistance = distance;
                        haveFallback = true;
                    }
                    continue;
                }

                key = candidate;
                hitDistance = distance;
                found = true;
            }
        }

        if (!found && haveFallback)
        {
            key = fallbackKey;
            hitDistance = fallbackDistance;
            found = true;
        }

        if (!found)
            return $"nothing cached is drawn along that line within {maxBlocks} blocks";

        int size = LodWorld.KeyFootprintBlocks(key);
        double originX = LodWorld.KeySx(key) * (double)size;
        double originZ = LodWorld.KeySz(key) * (double)size;
        double relX = originX - camPos.X;
        double relZ = originZ - camPos.Z;

        LodHeightSpan span = SectionSpan(key, waterPass: false);
        double boxMinY = span.HasGeometry ? span.MinY - camPos.Y : -camPos.Y;
        double boxMaxY = span.HasGeometry ? span.MaxY - camPos.Y : worldHeight - camPos.Y;

        LodHzbScreenBounds bounds = LodHzbProjection.Project(
            explainViewProjection, relX, boxMinY, relZ, relX + size, boxMaxY, relZ + size);

        var report = new System.Text.StringBuilder();
        report.Append($"L{LodWorld.KeyLevel(key)} section at {originX:0},{originZ:0}, ");
        report.Append($"{size} blocks across, about {hitDistance:0} blocks away. ");
        report.Append(span.HasGeometry
            ? $"Its terrain runs from y={span.MinY:0} to y={span.MaxY:0} ({span.Height:0} tall). "
            : "It has no recorded height, so it is bounded bedrock to sky. ");

        if (!bounds.Usable)
        {
            report.Append($"The test declined to judge it: {bounds.Reason}. ");
            report.Append("That always means draw it.");
            return report.ToString();
        }

        int levelUsed = LodHzbProjection.LevelFor(
            bounds.WidthPixels(depthPyramid.Width),
            bounds.HeightPixels(depthPyramid.Height),
            depthPyramid.Levels);
        report.Append($"On screen it covers about {bounds.WidthPixels(depthPyramid.Width):0} by ");
        report.Append($"{bounds.HeightPixels(depthPyramid.Height):0} pixels, which is read at ");
        report.Append($"pyramid level {levelUsed} of {depthPyramid.Levels - 1}. ");

        if (hzbLastVerdict.TryGetValue(key, out uint verdict))
        {
            report.Append(verdict switch
            {
                LodHzbClassifier.VerdictOccluded =>
                    "The card found everything in front of it nearer, so a depth test would hide it. ",
                LodHzbClassifier.VerdictBackground =>
                    "The card refused it because its rectangle includes sky, where nothing was "
                    + "drawn. No box overlapping open sky can ever be hidden, however buried its "
                    + "terrain is - which is a sign the box is too big rather than that the "
                    + "terrain is in view. ",
                LodHzbClassifier.VerdictFailedOpen =>
                    "The card declined to judge it and it is drawn. ",
                LodHzbClassifier.VerdictNearPlane =>
                    "The card declined: part of its box sits at or behind the camera plane, "
                    + "which happens to any section close enough to stand beside. It is drawn. ",
                LodHzbClassifier.VerdictOffScreen =>
                    "The card declined: it projects entirely off screen, which is the frustum "
                    + "test's business rather than the depth test's. It is drawn. ",
                LodHzbClassifier.VerdictDegenerate =>
                    "The card declined: its projection produced a value nothing can be judged "
                    + "from. It is drawn. ",
                _ => "The card found something behind the scene here, so it is genuinely in view. ",
            });
        }
        else
        {
            report.Append("The card has not returned a verdict for it in this reporting window. ");
        }

        if (temporalOcclusionQueries.TryGetValue(key, out TemporalOcclusionQuery? query))
        {
            report.Append(query.State.KnownVisible
                ? "An occlusion query has actually seen pixels of it, so it is really visible."
                : query.State.Occluded
                    ? "An occlusion query found it hidden too."
                    : "No current occlusion query result exists to check that against.");
        }
        else
        {
            report.Append("No occlusion query is tracking it.");
        }

        return report.ToString();
    }


    /// <summary>
    /// The whole depth-pyramid measurement, on demand.
    ///
    /// It exists because the periodic report fires once, thirty seconds after joining -
    /// which is before anyone has had a chance to turn this on, so in an ordinary session
    /// every figure below was collected and never shown. A measurement that can only be
    /// read by restarting with an environment variable set is not one a person will use.
    ///
    /// No angle brackets anywhere in here: this reaches the chat window, which parses its
    /// text as VTML, and a leading one silently swallows the whole reply (G67).
    /// </summary>
    /// <summary>
    /// What the aggregate subtree bound has actually rejected, on demand.
    ///
    /// The periodic report carries the same figures, but it fires once thirty seconds
    /// after a world loads unless continuous telemetry is switched on - which is before
    /// anybody has pointed the camera anywhere on purpose, and while terrain is still
    /// streaming. The number this work is judged on comes from looking up, looking down,
    /// or flying, so it has to be readable at the moment somebody is doing that.
    /// </summary>
    public string ReportSubtreeCulling()
    {
        if (!SubtreeHeightCulling)
        {
            return "subtree height culling off. Turn it on with .dlsubtree on, look up or "
                + "down or fly for a few seconds, then run .dlsubtree again";
        }

        var report = new System.Text.StringBuilder();
        report.Append($"subtree cull: {SubtreeCulledNodes} quadtree nodes rejected since the "
            + $"last reset that a full-height box would have traversed, holding "
            + $"{SubtreeCulledMeshes} resident meshes; worst single frame {SubtreeCulledMax} "
            + $"nodes, last frame {LastSubtreeCulledCount}");

        report.AppendLine();
        report.Append($"coverage: {SubtreeBoundsNodes} nodes carry an aggregate over "
            + $"{MeshCount} resident meshes | this frame the walk rejected "
            + $"{LastTraversalCulledCount} subtrees in total, drew {LastDrawCount}");

        report.AppendLine();
        report.Append(SubtreeCulledNodes == 0
            ? "nothing has been rejected vertically yet. That is a real answer, not a "
                + "failure: it would mean the bounds are correct and never tight enough to "
                + "reject. Run .dlsubtree reset, then look straight up or straight down for "
                + "a few seconds and read it again"
            : "run .dlsubtree reset before a specific view to measure that view alone");
        return report.ToString();
    }

    /// <summary>
    /// The floor and ceiling of every mesh RESIDENT RIGHT NOW, on demand.
    ///
    /// The periodic `section heights:` line measures a different population: the meshes
    /// PUBLISHED during an interval. Those are two different questions, and only one of
    /// them can be asked after a world settles - once terrain stops arriving nothing is
    /// published, so the periodic line reports an empty interval exactly when the answer
    /// matters. Worse, the only interval it has ever reported is the first thirty seconds
    /// after joining, which is the noisiest possible sample: a section meshed before its
    /// neighbour arrives walls its open side down to bedrock, and that wall is in the
    /// bounds until the seam repair rebuilds it.
    ///
    /// So this walks the live set instead. The floor bands are the point of it: a mean
    /// cannot tell "the terrain here really does run deep" apart from "most floors are
    /// honest and a few are pinned to bedrock", and those want opposite responses. The
    /// count of meshes still carrying a guessed edge is beside them, because that is the
    /// population the phantom walls can come from - if the floors are low while that count
    /// is near zero, missing neighbours are not the cause and the next look goes elsewhere.
    /// </summary>
    public string ReportLiveSectionHeights()
    {
        if (liveMeshStats.Count == 0) return "no meshes are resident yet";

        var stats = new LodSectionHeightStats();
        Span<int> floorBands = stackalloc int[5];
        int unknown = 0;
        float lowestFloor = float.PositiveInfinity;

        foreach (LodLiveMeshStats mesh in liveMeshStats.Values)
        {
            LodHeightSpan span = mesh.Heights.Either;
            if (!span.HasGeometry) { unknown++; continue; }
            stats.Add(span);
            if (span.MinY < lowestFloor) lowestFloor = span.MinY;
            floorBands[
                span.MinY <= 8f ? 0
                : span.MinY <= 32f ? 1
                : span.MinY <= 64f ? 2
                : span.MinY <= 128f ? 3
                : 4]++;
        }

        if (stats.Samples == 0)
            return $"{liveMeshStats.Count} meshes are resident and none of them reported bounds";

        var report = new System.Text.StringBuilder();
        report.Append("live section heights: ").Append(stats.Describe(worldHeight));
        if (unknown > 0) report.Append($" | {unknown} resident meshes reported no bounds");

        report.AppendLine();
        report.Append($"floors of the {stats.Samples} resident meshes: ");
        report.Append($"y0-8: {floorBands[0] * 100.0 / stats.Samples:0}%, ");
        report.Append($"y8-32: {floorBands[1] * 100.0 / stats.Samples:0}%, ");
        report.Append($"y32-64: {floorBands[2] * 100.0 / stats.Samples:0}%, ");
        report.Append($"y64-128: {floorBands[3] * 100.0 / stats.Samples:0}%, ");
        report.Append($"y128+: {floorBands[4] * 100.0 / stats.Samples:0}% ");
        report.Append($"(lowest y={lowestFloor:0})");

        report.AppendLine();
        report.Append($"edges: {meshedWithoutNeighbor.Count} resident meshes still carry a side "
            + $"guessed at because the neighbour was not in memory; {SeamRepairsQueued} seam "
            + "repairs have been queued since joining");

        report.AppendLine();
        report.Append(floorBands[0] * 2 > stats.Samples
            ? "most floors sit at bedrock. If this reading was taken after terrain settled, "
                + "the walls are being left behind rather than repaired, and every box-shaped "
                + "test is being told the terrain is far deeper than it is"
            : "most floors sit above bedrock, so the bounds describe terrain rather than the "
                + "whole world column");
        return report.ToString();
    }

    /// <summary>Zero the subtree-cull counters so one view can be measured on its own.</summary>
    public void ResetSubtreeCullingInterval()
    {
        SubtreeCulledNodes = 0;
        SubtreeCulledMeshes = 0;
        SubtreeCulledMax = 0;
    }

    public string ReportDepthPyramid()
    {
        if (!hzbShaderOk) return "the hzbreduce shader is not available, so no pyramid exists";
        if (!DepthPyramidEnabled)
            return "depth pyramid off. Turn it on with .dlhzb on, play for a few seconds, then run .dlhzb again";

        var report = new System.Text.StringBuilder();
        report.Append("depth pyramid: ").Append(DescribeDepthPyramid());

        report.AppendLine();
        report.Append($"cost: gpu {GpuHzbCost.AvgUs:0.0}us avg / {GpuHzbCost.MaxUs:0.0}us max");
        report.Append($" over {GpuHzbCost.Calls} timed builds");
        if (LateDepthPyramid)
        {
            report.Append($" | mid-frame gpu {GpuSplitHzbCost.AvgUs:0.0}us avg / "
                + $"{GpuSplitHzbCost.MaxUs:0.0}us max over {GpuSplitHzbCost.Calls} builds");
        }
        report.Append($" | classify gpu {GpuClassifyCost.AvgUs:0.0}us avg / "
            + $"{GpuClassifyCost.MaxUs:0.0}us max over {GpuClassifyCost.Calls} dispatches");
        report.Append($" | cpu {HzbCost.AvgUs:0.0}us avg / {HzbCost.MaxUs:0.0}us max");
        report.Append(" | gpu timing ").Append(DescribeGpuTiming());

        report.AppendLine();
        report.Append("picture: ").Append(DescribeLateDepthPyramid());

        report.AppendLine();
        report.Append("actual live commands: ")
            .Append(cullPass?.DescribeLiveTelemetry()
                ?? "the live cull pass has not been created");

        string byDistance = DescribeDepthPyramidByDistance();
        if (byDistance.Length > 0)
        {
            report.AppendLine();
            report.Append("hidden by distance: ").Append(byDistance);
        }

        string subdivision = DescribeDepthPyramidSubdivision();
        if (subdivision.Length > 0)
        {
            report.AppendLine();
            report.Append("headroom: ").Append(subdivision);
        }

        string undecided = DescribeDepthPyramidUndecided();
        if (undecided.Length > 0)
        {
            report.AppendLine();
            report.Append("undecided because: ").Append(undecided);
        }

        if (HzbVerdictsChecked > 0)
        {
            report.AppendLine();
            report.Append($"against occlusion queries: {HzbVerdictsChecked} checked, ");
            report.Append($"{HzbHiddenButQuerySawIt} called hidden that a query SAW ");
            report.Append(HzbHiddenButQuerySawIt == 0 ? "(good, must be zero)" : "(BAD, must be zero)");
            report.Append($", {HzbAgreedHidden} both hid");
            report.Append($", {HzbMissedWhatQueryHid} the query hid and the pyramid did not");
            report.AppendLine();
            report.Append($"new saving: {HzbHiddenBeyondQueries} hidden that no query had measured");
            report.Append(" - this is what the pyramid adds over the occlusion already running");
            report.Append($" | {HzbVerdictsAcrossViewChange} comparisons refused across a view change");
        }
        else
        {
            report.AppendLine();
            report.Append("against occlusion queries: nothing checked yet - no section has both ");
            report.Append("a pyramid verdict and a current query result");
        }

        return report.ToString();
    }

    /// <summary>
    /// Whether the second, same-frame depth opportunity is active and whether its mid-frame
    /// builds are actually succeeding.
    /// </summary>
    public string DescribeLateDepthPyramid()
    {
        if (!LateDepthPyramid)
            return "taken after the game's own terrain and before cached terrain, so only the "
                + "game's nearby hills can hide anything - cached hills cannot. "
                + "Switch with .vhlate on";

        if (SplitDepthFrames == 0)
            return "same-frame near/far split requested, but no split frame has drawn yet";

        return $"same-frame split at {LodGpuDepthSplitPolicy.NearRadiusBlocks:0} blocks "
            + $"with the established {LodHzbProjection.OcclusionDepthBiasSteps}-step safety "
            + $"margin on both buckets: "
            + $"{SplitDepthFrames} frames, {SplitDepthNearCommands} near commands and "
            + $"{SplitDepthFarCommands} far commands, {SplitDepthMidBuilds} mid-frame "
            + "pictures completed. A failed picture draws the far bucket without culling.";
    }

    /// <summary>Hidden share per distance band; empty until something has been classified.</summary>
    public string DescribeDepthPyramidByDistance() => hzbClassifier?.DescribeByDistance() ?? "";

    /// <summary>Why the test declined, when it did; empty when it never declined.</summary>
    public string DescribeDepthPyramidUndecided() => hzbClassifier?.DescribeUndecided() ?? "";

    /// <summary>
    /// What a finer draw unit would add. Measured, not argued: the question of whether
    /// cluster subdivision should move ahead of the rest of the plan is the owner's, and
    /// this is the evidence it should be decided on.
    /// </summary>
    public string DescribeDepthPyramidSubdivision() => hzbClassifier?.DescribeSubdivision() ?? "";

    /// <summary>One line for the periodic report and for `.dlhzb`.</summary>
    public string DescribeDepthPyramid()
    {
        if (!hzbShaderOk) return "the hzbreduce shader is not available";
        if (!DepthPyramidEnabled) return "off (.dlhzb on)";
        if (depthPyramid == null) return "on, not yet built";
        string classifier = hzbClassifier == null
            ? "classification not started"
            : hzbClassifier.Describe();
        return depthPyramid.Describe() + " || " + classifier;
    }

    LodHeightSpan SectionSpan(long key, bool waterPass) =>
        SectionHeightCulling && liveMeshStats.TryGetValue(key, out LodLiveMeshStats stats)
            ? (waterPass ? stats.Heights.Water : stats.Heights.Opaque)
            : LodHeightSpan.Empty;

    /// <summary>
    /// The box a quadtree node may be traversed with, or the empty span when it must keep
    /// the full-height one. Unknown aggregates and the switch being off both arrive here as
    /// empty, so there is one place where a subtree can lose its real bounds.
    /// </summary>
    LodHeightSpan SubtreeSpan(long key) =>
        SubtreeHeightCulling ? subtreeHeights.Of(key).CullingSpan : LodHeightSpan.Empty;

    /// <summary>
    /// Whether the neighbouring section holds (or covers) data. Checked at the drawn
    /// section's own level: a coarse section's neighbour is coarse too, and its
    /// presence in HasDataSet means something in that subtree was captured.
    /// </summary>
    bool HasNeighbourData(long key, int dx, int dz) =>
        world.HasDataSet.Contains(LodWorld.NeighborKey(key, dx, dz));

    public void ClearMeshes()
    {
        renderPaths.Clear(currentWorldEpoch());
    }

    /// <summary>
    /// Retire block-texture GPU state at a world boundary. The registry is cleared by the
    /// owning mod system; this resets render-side failure/version state so the next world
    /// receives a fresh upload attempt.
    /// </summary>
    public void ResetBlockTextureStateForWorld()
    {
        WaterSurfaceTextureSlot = 0;
        WaterSurfaceTintSlot = -1;
        textureSlotSourceReady = false;
        SetTextureSlotEmission(false);
        DisposeBlockTableTexture();
        blockTableVersion = -1;
        blockTableFailed = false;
    }

    void ClearMeshesLegacy(long worldEpoch)
    {
        foreach (MeshRef meshRef in sectionMeshes.Values) meshRef.Dispose();
        foreach (MeshRef meshRef in waterMeshes.Values) meshRef.Dispose();
        sectionMeshes.Clear();
        waterMeshes.Clear();
        liveMeshStats.Clear();
        liveTexturedMeshes = 0;
        MaybeReleaseBlockTextureTable();
        subtreeHeights.Clear();
        LiveOpaqueVertices = LiveOpaqueIndices = 0;
        LiveWaterVertices = LiveWaterIndices = 0;
        LiveGpuMeshBytes = 0;
        meshedWithoutNeighbor.Clear();
        meshBounds.Clear();
        farPlaneState.Reset();
        appliedZFar = 0;
        EffectiveFarDistance = LodFarDistance.MinimumProjectionDistance;
        meshJobInFlight.Clear();
        lastResidencyMs.Clear();
        meshEvictionOrder.Clear();
        meshEvictionQueued.Clear();
        seasonalRefreshActive = false;
        seasonalStateInitialized = false;
        seasonalRefreshSlot = 0;
        seasonalRefreshSnowObservationRevision = 0;
        publishedSnowObservationRevision = 0;
        tints.InvalidateReadiness();

        // Remembered climate is per world and nothing in it is true in the next one. The rows
        // stay on disk; only the copy in memory goes.
        //
        // No WRITE from here. This runs from the render path's mesh clear, whose ordering
        // relative to the storage worker's disposal is not ours to choose - and on the world
        // leave path the worker is already gone by the time this is reached. The owner of both
        // calls SaveClimateMemory first; see the mod system's leave path.
        climateField.Clear();
        ClimateMemory.Clear();
        snowLineY = pendingSnowLineY = 99999;
        DisposeTemporalOcclusionQueries();
        ClearReadiness();
    }

    void ClearReadiness()
    {
        readiness?.Clear();
        readiness = null;
        readinessSeedCursor = readinessBoundaryCursor = readinessInteriorCursor = 0;
        readinessDisabled = false;
        readinessFailureReported = false;
        gapTracker.Clear();
    }

    public void Dispose()
    {
        // Our own resources first. UnregisterRenderer refuses to run off the main thread,
        // and the game's shutdown crash path disposes mods from another one, so putting
        // the engine call first meant a crashing client freed none of its GPU meshes.
        ClearMeshes();
        renderPaths.Dispose();
        indirectDrawer?.Dispose();
        indirectDrawer = null;
        packedDrawer?.Dispose();
        packedDrawer = null;
        cullPass?.Dispose();
        cullPass = null;
        depthPyramid?.Dispose();
        depthPyramid = null;
        hzbClassifier?.Dispose();
        hzbClassifier = null;
        gpuTelemetry.Dispose();
        // Same reason as the meshes: the mask texture is ours, and a shutdown that never
        // reaches the engine call must still not leak it.
        DisposeReadinessMaskTexture();
        DisposeSeamFieldTexture();
        DisposeBlockTableTexture();
        DisposeSkyWitnessResources();
        DisposeTaaResources();
        UnregisterSilhouetteFogStage();
        ReleaseSilhouetteMeshes();
        world.SectionBecameResident -= OnSectionBecameResident;
        capi.Event.ChunkDirty -= OnReadinessChunkDirty;
        capi.Event.ReloadShader -= LoadShader;
        UnregisterWaterDepthStages();
        if (rendererRegistered)
        {
            capi.Event.UnregisterRenderer(this, EnumRenderStage.Opaque);
            rendererRegistered = false;
        }
    }
}
