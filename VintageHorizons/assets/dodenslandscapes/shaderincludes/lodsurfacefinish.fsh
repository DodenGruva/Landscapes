    // Vanilla's light term, derived by reading applyLight in fogandlight.vsh for the
    // case a LOD section is in: sky light, no baked block light, no glow. Every scale
    // factor there cancels except the contrast constant and the sun light level, leaving
    // the pixel multiplied by 1.05 * lightColor.a * ambientColor. The ambient colour
    // already carries the sunset hue, the night blue, and the scene-brightness dimming,
    // so nothing here has to re-derive any of them; sunLightLevel is the sun level, and
    // the comment on it says why treating it as 1 was worth 24% brightness everywhere.
    vec3 light = lightAmbientColor == 1
        ? 1.05 * sunLightLevel * rgbaAmbientIn
        : clamp(sunColor * dayLight, 0.02, 1.0);
    // Vanilla ground receives this factor through applyFogAndShadowWithNormal. Liquid calls
    // getBrightnessFromShadowMap directly and never applies the daylight multiplier, so
    // sharing ground's boost made cached water up to 22.7% brighter at the handoff.
    if (lightDayBoost == 1 && band != 1)
        light *= 1.0 + max(0.0, shadowIntensity * 2.0 - 1.66) / 1.5;

    // The engine applies minbrightness to terrain lighting too. Raise each channel equally
    // when this term falls below that luminance, before multiplying by the block's albedo.
    float lightLuminance = dot(light, vec3(0.2126, 0.7152, 0.0722));
    if (lightLuminance < minBrightness)
    {
        light += vec3(minBrightness - lightLuminance);
    }

    vec4 terraColor = vec4(albedo, outAlpha);
    terraColor.rgb *= shade * light;

    // Vanilla's own two water surface terms, on the same band the ground's corrections use.
    // Both are transcribed from chunkliquid.fsh's cold-liquid branch for the case a distant
    // LOD lake is in: still water, so its flow vector is zero and the flow counter passes
    // through; open water, so its depth-difference term is zero and drops out of the foam
    // exactly. Applied AFTER lighting and before fog, where vanilla applies them.
    // Every cached water fragment, NOT the blend band.
    //
    // These were band-weighted for cost, and that was a mistake with a visible edge: the two
    // terms faded out over the band and then stopped, so distant water carried a line across
    // it exactly where the band ended. Vanilla's water is shiny and grainy everywhere it is
    // drawn, out to its own horizon, so cached water carrying the same terms everywhere is
    // MORE faithful than fading them, not less - and a term that ends nowhere cannot draw an
    // edge anywhere. waterGlint and waterFoam are the strengths; zero turns each off.
    float waterAmount = band == 1 ? 1.0 : 0.0;
    if (waterAmount > 0.0 && (waterFoam > 0.0 || waterGlint > 0.0))
    {
        float upness = clamp(abs(normal.y), 0.0, 1.0);
        float waterVn = max(0.0, 0.9 - upness);
        vec3 foamPos = worldPos.xyz + vhFoamOrigin;
        float a = foamPos.x + foamPos.y;
        float b = foamPos.z;
        float noise1 = gnoise(vec3(a * 15.0, -abs(b * 15.0), vhWaterFlow))
            + gnoise(vec3(a * 5.0, b * 5.0, vhWaterFlow));
        float noise2 = gnoise(vec3(a, b, vhWaterFlow));

        if (waterFoam > 0.0)
        {
            float foamAmount = waterAmount * waterFoam;
            // Zero means deep/open water. Vanilla's measured scene-depth term pins its
            // shallow rim to zero there; feeding zero into the later noisy expression did
            // not do that and still produced bright, opaque foam patches across the ocean.
            float diff = waterShallowness > 0.0
                ? upness * clamp(
                    waterShallowness - noise1 / 2.0 + noise2 / 2.0, 0.0, 1.0)
                : 0.0;
            float grain = upness
                * (gnoise(vec3(a * 0.4, b * 0.4, vhWaterFlow)) / 2.0 + noise2 / 2.0 + 0.5) / 2.0;
            float bright = (terraColor.r + terraColor.g + terraColor.b) / 3.0;
            terraColor.rgb += vec3(max(0.0, bright * (diff * 0.3 + grain / 10.0)) * foamAmount);
            terraColor.a += (
                max(0.0, diff / 16.0
                    + grain / (12.0 - 8.0 * min(1.0, vhWindSpeed)))
                + waterVn / 4.0) / clamp(waterFresnelValue, 0.5, 1.0) * foamAmount;
        }

        // Specular sun glint. Vanilla also scales this by a shadow-map sample; distant open
        // water is unshadowed, and this shader has no shadow lookup, so that factor is 1.
        if (waterGlint > 0.0 && vhSunSpecular > 0.0)
        {
            vec3 noisepos = vec3(foamPos.x, foamPos.z, vhWaterWave / 8.0 + vhWindWave / 6.0);
            float dy = noise2 / 20.0 + clamp(gnoise(noisepos) / 10.0, 0.0, 0.6);
            vec3 waveNormal = normalize(vec3(dy, 1.0, -dy));
            vec3 eye = normalize(vec3(worldPos.x, worldPos.y - 2.0, worldPos.z));
            float p = dot(reflect(vhSunPosRel, waveNormal), eye);
            if (p > 0.0)
            {
                float sunb = clamp(vhSunPosRel.y * 10.0, 0.0, 1.0)
                    * clamp(1.5 - vhSunPosRel.y, 0.0, 1.0) * vhSunSpecular;
                float specular = pow(p, 50.0) * sunb;
                float weight = upness
                    * clamp(specular * clamp(1.5 * shadowIntensity, 0.0, 1.0), 0.0, 1.0)
                    * waterAmount * waterGlint;
                // chunkliquid.fsh fogs the reflected sky here, then fogs the finished liquid
                // below. Keep that exact ordering at the handoff even though it looks like a
                // double application when read in isolation.
                vec3 reflected = applyFog(vec4(vhReflectColor, 1.0), fogAmount).rgb;
                terraColor.rgb = mix(terraColor.rgb, reflected + noise1 * 0.2, weight);
                terraColor.a = mix(terraColor.a, terraColor.a + specular / 2.0, weight);
            }
        }

        terraColor.a = clamp(terraColor.a, 0.0, 1.0);
    }

    // Vanilla removes atmospheric fog from submerged ground, then mixes it toward
    // waterMurkColor. Sunlight absorption alone cannot reproduce either operation.
    // The engine's separate liquid-depth target is complete before opaque rendering,
    // so this reads no attachment of the framebuffer we are currently writing.
    // Use the same screen-space test as vanilla: the visible bed can be in a different
    // ownership cell from the water in front of it. No extra overlap or radius.
    float groundMurk = 0.0;
    float groundFog = fogAmount;
    if (!translucent && waterGroundMurk == 1)
    {
        groundMurk = getUnderwaterMurkiness();
        groundFog = clamp(fogAmount - 50.0 * groundMurk, 0.0, 1.0);
    }
    terraColor = applyFog(terraColor, groundFog);
    // chunkliquid.fsh makes distant water itself opaque as atmospheric fog rises:
    // applyFog changes its colour, then the shader adds fogAmount to alpha before OIT.
    // Leaving that second step out made vanilla's last visible water surge toward opaque
    // while cached water started again at its Fresnel/texture alpha, drawing the broad
    // handoff band even though the ground colour and liquid depth were continuous.
    if (band == 1) terraColor.a = clamp(terraColor.a + fogAmount, 0.0, 1.0);
    terraColor = applySpheresFog(terraColor, groundFog, worldPos.xyz);
    if (!translucent && waterGroundMurk == 1)
        terraColor.rgb = applyUnderwaterEffects(terraColor.rgb, groundMurk);

    // Dissolve both the far edge of the cache and the edges of the explored area
    // into the sky, so neither ends in a visible wall.
    float fade = max(smoothstep(0.75, 1.0, dist), edgeFade);

    // Only work out the sky where it is actually mixed in. mix(x, y, 0.0) is x, so
    // skipping this where fade is zero cannot change a pixel, and fade is zero across
    // the inner part of the band, which is most of the terrain on screen.
    //
    // It is worth skipping. getSkyColorAt costs several texture fetches, three
    // normalize calls, a pow, and a noise chain of eight sin calls, per fragment,
    // and all of it was being multiplied by zero.
    //
    // The branch is coherent: fade is a function of distance, so the fragments that
    // take it are a ring near the horizon rather than a speckle across the screen.
    if (fade > 0.0) {
        vec4 skyColor = vec4(1.0);
        vec4 skyGlow = vec4(1.0);
        vec3 worldPosInSky = normalize(worldPos.xyz) * 250.0;
        // Only the sky colour takes the corrected value. The glow clamp below is this
        // mod's own night dimming, not a transcription of vanilla, and its 0.05 constant
        // is calibrated against DayLightStrength - SkyDaylight never falls below 0.0625,
        // so feeding it here would leave a faint glow burning all night.
        float skyLight = lightSkyDayLight == 1 ? skyDayLight : dayLight;
        getSkyColorAt(worldPosInSky, sunPosition, 0.25, clamp(skyLight, 0.0, 1.0), horizonFog, skyColor, skyGlow);
        float murkiness = max(0.0, getSkyMurkiness() - 14.0 * fogDensityIn);
        skyColor.rgb = applyUnderwaterEffects(skyColor.rgb, murkiness);
        skyGlow.y *= clamp((dayLight - 0.05) * 2.0 - 50.0 * murkiness, 0.0, 1.0);

        vec4 fadeColor = skyColor;
        vec4 fadeGlow = skyGlow;
        if (silhouetteEnabled == 1)
        {
            vhSilhouetteTerrainColor(worldPos.xyz, yLevel, sunPosition, sunColor, dayLight,
                horizonFog, fogAmount, silhouetteSkyTint, silhouetteColorTint,
                silhouetteLightLevelBias, 1.0, fadeColor, fadeGlow);
            fadeGlow *= dist;
        }

        outColor = mix(terraColor, fadeColor, fade);
        outGlow = mix(vec4(0.0), fadeGlow, fade);
    } else {
        outColor = terraColor;
        outGlow = vec4(0.0);
    }

#if SSAOLEVEL > 0
    outGPosition = vec4(0.0);
    outGNormal = vec4(0.0);
#endif
