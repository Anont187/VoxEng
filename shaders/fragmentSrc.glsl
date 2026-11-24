/*
 * Copyright 2026 Anont187
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/
 */

#version 430 core

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragNormal;
layout(std430, binding = 0) readonly buffer GiVolume {
	uint giVoxels[];
};

struct svoNodeGpu {
	uvec2 childMask;

	uint childPointer;
	uint padding;
};

layout(std430, binding = 2) readonly buffer svoVolume {
	svoNodeGpu svoNodes[];
};

in vec2 vUv;
in vec2 vSkyUv;
in vec3 vWorldPos;
in float vAo;

flat in uint vLayer;
flat in uint vFace;

uniform sampler2DArray texAtlas;
uniform vec3 sunDir;
uniform int uMode;
uniform vec4 uColor;
uniform mat4 invViewProj;
uniform vec3 camPos;
uniform float timeOfDay;
uniform usamplerBuffer voxelTbo;
uniform int playerChunkX;
uniform int playerChunkZ;
uniform int playerChunkY;
uniform int maxHeightTerrain;
uniform usamplerBuffer brickTbo;
uniform int brickDimX;
uniform int brickDimY;
uniform int brickDimZ;
uniform int lightingMode;
uniform int uFrameIndex;
uniform sampler2D uScene;
uniform sampler2D uAccum;
uniform float uBlend;
uniform sampler2D uBloomInput;
uniform float uBloomThreshold;
uniform float uBloomStrength;
uniform vec2 uBloomTexelSize;
uniform vec3 uSsaoKernel[12];
uniform float uBlurWeights[5];
uniform sampler2D uSceneDepth;
uniform sampler2D uSceneNormal;
uniform sampler2D uSsaoNoise;
uniform vec2 uSsaoTexelSize;
uniform float uSsaoRadius;
uniform float uSsaoBias;
uniform float uSsaoIntensity;
uniform sampler2D uSsaoInput;
uniform sampler2D uSsaoResult;
uniform mat4 viewProj;
uniform float uSsaoEnabled;
uniform vec2 uSsgiOffsets[8];
uniform float uShadowFactor;
uniform float uGiStrength;
uniform sampler2D uLumTex;
uniform sampler2D uExposureTex;
uniform float uAdaptFactor;
uniform float uExposureKey;
uniform float uExposureBias;
uniform int uGiDebug;
uniform ivec3 giOrigins[3];
uniform float giVoxelSizes[3];
uniform float uAutoExposureFloor;
uniform float uExposureMin;
uniform float uExposureMax;
uniform int uGiDim;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uHorizonColor;
uniform vec2 uSunScreenUV;
uniform float uSunVisible;
uniform mat4 prevViewProj;
uniform vec2 uInternalTexelSize;
uniform int uGgtuEnabled;
uniform float uMotionBlurScale;

const int brickSize = 8;
const int cascadeDims[3] = int[3](32, 16, 8);
const int cascadeRays[3] = int[3](4, 8, 16);
const int cascadeOffsets[3] = int[3](0, 32768, 36864);
const float PI = 3.14159265;

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float valueNoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);

	float a = hash(i);
	float b = hash(i + vec2(1.0, 0.0));
	float c = hash(i + vec2(0.0, 1.0));
	float d = hash(i + vec2(1.0, 1.0));

	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
	float v = 0.0;
	float amp = 0.5;

	for (int i = 0; i < 4; i++) {
		v += amp * valueNoise(p);
		p *= 2.03;
		amp *= 0.5;
	}

	return v;
}

vec3 renderSky(vec2 screenUV) {
    vec4 nearH = invViewProj * vec4(screenUV * 2.0 - 1.0, -1.0, 1.0);
    vec4 farH  = invViewProj * vec4(screenUV * 2.0 - 1.0,  1.0, 1.0);

    vec3 dir = normalize((farH.xyz / farH.w) - (nearH.xyz / nearH.w));

    float sunHeight   = sunDir.y;
    float dayAmount   = smoothstep(-0.15, 0.25, sunHeight);
    float nightAmount = 1.0 - dayAmount;
    float sunsetMask  = exp(-sunHeight * sunHeight * 18.0);

    vec3 nightZenith  = vec3(0.004, 0.008, 0.022);
    vec3 nightHorizon = vec3(0.012, 0.022, 0.048);

    vec3 dayZenith    = (lightingMode == 1) ? uSkyColor     : vec3(0.20, 0.52, 1.02);
    vec3 dayHorizon   = (lightingMode == 1) ? uHorizonColor : vec3(0.68, 0.86, 1.04);

    vec3 zenith  = mix(nightZenith,  dayZenith,  dayAmount);
    vec3 horizon = mix(nightHorizon, dayHorizon, dayAmount);

    float zenithFactor = clamp(pow(max(dir.y, 0.0), 0.55), 0.0, 1.0);

    vec3 skyColor = mix(horizon, zenith, zenithFactor);

    vec2 hDir = normalize(vec2(dir.x, dir.z) + vec2(1e-6));
    vec2 hSun = normalize(vec2(sunDir.x, sunDir.z) + vec2(1e-6));

    float sunSide = smoothstep(-0.4, 1.0, dot(hDir, hSun));

    skyColor = mix(skyColor, vec3(1.15, 0.52, 0.20), sunsetMask * sunSide * 0.55);

    float sunDot  = dot(dir, sunDir);
    float sunDisc = smoothstep(0.9993, 0.9996, sunDot);
    float sunGlow = pow(max(sunDot, 0.0), 96.0) * 0.65 + pow(max(sunDot, 0.0), 10.0) * 0.12;

    vec3 activeSunColor = (lightingMode == 1) ? uSunColor : mix(vec3(1.0, 0.55, 0.25), vec3(1.0, 0.96, 0.88), smoothstep(0.0, 0.35, sunHeight));

    skyColor += activeSunColor * sunGlow * max(dayAmount, 0.15);
    skyColor += activeSunColor * sunDisc * 2.5 * dayAmount;

    vec3 moonDir = -sunDir;

    float moonDot = dot(dir, moonDir);
    float moonDisc = smoothstep(0.9994, 0.9997, moonDot);
    float moonGlow = pow(max(moonDot, 0.0), 48.0) * 0.25;

    skyColor += vec3(0.55, 0.72, 1.0) * (moonDisc * 1.4 + moonGlow) * nightAmount;

    float totalCloudAlpha = 0.0;
    if (nightAmount > 0.05 && dir.y > 0.03) {
        vec3 starDir = normalize(dir + vec3(timeOfDay * 0.15, 0.0, 0.0));

        vec2 starGrid = floor(starDir.xz / (starDir.y + 0.35) * 140.0);
        float starSeed = hash(starGrid);
        if (starSeed > 0.992) {
            float twinkle = 0.65 + 0.35 * sin(timeOfDay * 450.0 + starSeed * 100.0);
            float starBright = (starSeed - 0.992) * 125.0 * twinkle * nightAmount * smoothstep(0.03, 0.25, dir.y);

            skyColor += mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.92, 0.80), fract(starSeed * 37.0)) * starBright;
        }
    }

    if (dir.y > 0.02) {
        float t = timeOfDay * 20.0;
        float horizonFade = smoothstep(0.02, 0.22, dir.y);

        vec2 cirrusUV = dir.xz / (dir.y + 0.25) * 0.9 + vec2(t * 0.02, t * 0.008);
        vec2 stretchedUV = vec2(cirrusUV.x * 0.45 + fbm(cirrusUV) * 0.6, cirrusUV.y * 2.2);

        float cirrus = smoothstep(0.56, 0.82, fbm(stretchedUV)) * 0.38;

        vec3 cirrusDay   = mix(vec3(1.05, 1.06, 1.10), vec3(1.30, 0.72, 0.48), sunsetMask);
        vec3 cirrusNight = vec3(0.05, 0.08, 0.15);
        vec3 cirrusCol   = mix(cirrusNight, cirrusDay, dayAmount);

        skyColor = mix(skyColor, cirrusCol, cirrus * horizonFade);

        vec2 cloudUV = dir.xz / (dir.y + 0.16) * 1.45 + vec2(t * 0.045, t * 0.015);

        float warp = fbm(cloudUV * 0.55 + vec2(2.3, 6.1));
        float baseNoise = fbm(cloudUV + warp * 0.65);
        float edgeDetail = valueNoise(cloudUV * 4.5 - vec2(t * 0.05, 0.0));
        float cumulus = smoothstep(0.48, 0.77, baseNoise * 0.85 + edgeDetail * 0.15);

        vec3 activeLightDir = mix(moonDir, sunDir, dayAmount);

        float lightSample = fbm(cloudUV + normalize(activeLightDir.xz + vec2(1e-4)) * 0.12 + warp * 0.65);
        float cloudLit = clamp((baseNoise - lightSample) * 3.5 + 0.65, 0.0, 1.0);

        vec3 dayShadow   = vec3(0.56, 0.68, 0.84);
        vec3 dayBright   = mix(vec3(1.25, 1.24, 1.20), vec3(1.35, 0.74, 0.42), sunsetMask);
        vec3 nightShadow = vec3(0.015, 0.025, 0.048);
        vec3 nightBright = vec3(0.08, 0.13, 0.24);

        vec3 cShadow = mix(nightShadow, dayShadow, dayAmount);
        vec3 cBright = mix(nightBright, dayBright, dayAmount);

        float silverLining = pow(max(sunDot, 0.0), 12.0) * (1.0 - cumulus * 0.6) * dayAmount;

        vec3 cumulusCol = mix(cShadow, cBright, cloudLit) + activeSunColor * silverLining * 0.55;

        skyColor = mix(skyColor, cumulusCol, cumulus * horizonFade * 0.92);
        totalCloudAlpha = clamp(cumulus + cirrus * 0.5, 0.0, 1.0);
    }

    if (nightAmount > 0.02 && dir.y > 0.0) {
        float band = exp(-pow((dir.y - 0.35) * 3.0, 2.0));

        vec2 auroraUV = dir.xz / (dir.y + 0.5) * 0.6;

        float t = timeOfDay * 40.0;
        float n1 = fbm(auroraUV * 1.2 + vec2(t * 0.15, 0.0));
        float n2 = fbm(auroraUV * 2.4 + vec2(-t * 0.08, t * 0.03));
        float curtain = smoothstep(0.35, 0.75, n1 * 0.6 + n2 * 0.4);

        vec3 green  = vec3(0.15, 0.95, 0.45);
        vec3 purple = vec3(0.55, 0.25, 0.85);
        vec3 auroraColor = mix(green, purple, smoothstep(0.3, 0.7, dir.y));

        float cloudOcc = 1.0 - totalCloudAlpha * 0.75;

        skyColor += auroraColor * band * curtain * nightAmount * cloudOcc * 0.65;
    }

    return skyColor;
}

bool rmGetVoxel(ivec3 wp) {
	if (wp.y < 0 || wp.y >= maxHeightTerrain) return false;

	int cx = wp.x >> 5;
	int cy = wp.y >> 5;
	int cz = wp.z >> 5;
	int rx = cx - playerChunkX + 2;
	int rz = cz - playerChunkZ + 2;
	if (rx < 0 || rx >= 5 || rz < 0 || rz >= 5 || cy < 0 || cy >= (maxHeightTerrain >> 5)) return false;

	int lx = wp.x & 31;
	int ly = wp.y & 31;
	int lz = wp.z & 31;
	int rowIdx = (cy * 25 + rz * 5 + rx) * 1024 + ly * 32 + lz;

	uint bits = texelFetch(voxelTbo, rowIdx).r;

	return (bits & (1u << uint(lx))) != 0u;
}

bool rmBrickOccupied(ivec3 brickIdx) {
	if (brickIdx.x < 0 || brickIdx.y < 0 || brickIdx.z < 0) return false;
	if (brickIdx.x >= brickDimX || brickIdx.y >= brickDimY || brickIdx.z >= brickDimZ) return false;

	int idx = (brickIdx.y * brickDimZ + brickIdx.z) * brickDimX + brickIdx.x;

	return texelFetch(brickTbo, idx).r != 0u;
}

bool rmSvoBitSet(uvec2 mask, int bitIndex) {
	if (bitIndex < 32) {
		return (mask.x & (1u << uint(bitIndex))) != 0u;
	}

	return (mask.y & (1u << uint(bitIndex - 32))) != 0u;
}

int rmSvoEmptyCellSize(ivec3 worldVoxel) {
	ivec3 svoOrigin = ivec3(
		(playerChunkX - 1) * 32,
		(playerChunkY - 1) * 32,
		(playerChunkZ - 1) * 32
	);

	ivec3 local = worldVoxel - svoOrigin;
	if (any(lessThan(local, ivec3(0))) || any(greaterThanEqual(local, ivec3(64)))) {
		return -1;
	}

	int rootBit = (local.x >> 4) + ((local.y >> 4) * 4) + ((local.z >> 4) * 16);
	if (!rmSvoBitSet(svoNodes[0].childMask, rootBit)) {
		return 16;
	}

	int macroIndx = int(svoNodes[0].childPointer) + rootBit;

	svoNodeGpu macroNode = svoNodes[macroIndx];

	int microX = (local.x >> 2) & 3;
	int microY = (local.y >> 2) & 3;
	int microZ = (local.z >> 2) & 3;
	int microBit = microX + microY * 4 + microZ * 16;
	if (!rmSvoBitSet(macroNode.childMask, microBit)) {
		return 4;
	}

	int leafIndex = int(macroNode.childPointer) + microBit;

	svoNodeGpu leafNode = svoNodes[leafIndex];

	int voxelX = local.x & 3;
	int voxelY = local.y & 3;
	int voxelZ = local.z & 3;
	int voxelBit = voxelX + voxelY * 4 + voxelZ * 16;
	if (rmSvoBitSet(leafNode.childMask, voxelBit)) {
		return 0;
	}

	return 1;
}

bool rmDda(vec3 ro, vec3 rd, float maxDist, out vec3 outNormal, out vec3 outHitPos) {
	ivec3 v = ivec3(floor(ro));
	ivec3 s = ivec3(sign(rd));

	vec3 invRd = 1.0 / (rd + vec3(equal(rd, vec3(0.0))) * 1e-30);
	vec3 tDelta = abs(invRd);
	vec3 fr = fract(ro);
	vec3 tMax;

	tMax.x = (rd.x > 0.0) ? (1.0 - fr.x) * tDelta.x : (rd.x < 0.0) ? fr.x * tDelta.x : 1e30;
    tMax.y = (rd.y > 0.0) ? (1.0 - fr.y) * tDelta.y : (rd.y < 0.0) ? fr.y * tDelta.y : 1e30;
    tMax.z = (rd.z > 0.0) ? (1.0 - fr.z) * tDelta.z : (rd.z < 0.0) ? fr.z * tDelta.z : 1e30;

    ivec3 bv = v >> 3;

    vec3 brickDelta = vec3(brickSize) * abs(invRd);
    vec3 brickTMax;

    brickTMax.x = (rd.x > 0.0) ? (float((bv.x + 1) * brickSize) - ro.x) * invRd.x : (rd.x < 0.0) ? (float(bv.x * brickSize) - ro.x) * invRd.x : 1e30;
    brickTMax.y = (rd.y > 0.0) ? (float((bv.y + 1) * brickSize) - ro.y) * invRd.y : (rd.y < 0.0) ? (float(bv.y * brickSize) - ro.y) * invRd.y : 1e30;
    brickTMax.z = (rd.z > 0.0) ? (float((bv.z + 1) * brickSize) - ro.z) * invRd.z : (rd.z < 0.0) ? (float(bv.z * brickSize) - ro.z) * invRd.z : 1e30;

    ivec3 brickOrigin = ivec3((playerChunkX - 2) * 32, 0, (playerChunkZ - 2) * 32) >> 3;
    ivec3 tboBrick = bv - brickOrigin;

    bool brickOccupied = rmBrickOccupied(tboBrick);

    float t = 0.0;

    outNormal = vec3(0.0);
    outHitPos = vec3(0.0);

    int maxSteps = int(ceil(maxDist * (abs(rd.x) + abs(rd.y) + abs(rd.z)))) + 8;

    for (int i = 0; i <= maxSteps; i++) {
    	if (t > maxDist) break;
        if (v.y < 0 || v.y >= maxHeightTerrain) break;

        int dxChunk = (v.x >> 5) - playerChunkX;
        int dzChunk = (v.z >> 5) - playerChunkZ;
        if (dxChunk < -2 || dxChunk > 2 || dzChunk < -2 || dzChunk > 2) break;

        int svoSize = rmSvoEmptyCellSize(v);
        if (svoSize == 0 && rmGetVoxel(v)) {
        	outHitPos = ro + rd * t;

        	return true;
        }

        if (svoSize > 1) {
        	ivec3 svoOrigin = ivec3((playerChunkX - 1) * 32, (playerChunkY - 1) * 32, (playerChunkZ - 1) * 32);

        	ivec3 local = v - svoOrigin;
        	ivec3 cellMin = svoOrigin + (local / svoSize) * svoSize;
        	ivec3 cellMax = cellMin + ivec3(svoSize);

        	float tx = 1e30;
        	float ty = 1e30;
        	float tz = 1e30;
        	if (rd.x > 0.0) {
        		tx = (float(cellMax.x) - ro.x) / rd.x;
        	} else if (rd.x < 0.0) {
        		tx = (float(cellMin.x) - ro.x) / rd.x;
        	}

        	if (rd.y > 0.0) {
        		ty = (float(cellMax.y) - ro.y) / rd.y;
        	} else if (rd.y < 0.0) {
        		ty = (float(cellMin.y) - ro.y) / rd.y;
        	}

        	if (rd.z > 0.0) {
        		tz = (float(cellMax.z) - ro.z) / rd.z;
        	} else if (rd.z < 0.0) {
        		tz = (float(cellMin.z) - ro.z) / rd.z;
        	}

        	float nextCellT = min(tx, min(ty, tz));
        	if (nextCellT > maxDist) break;
        	if (tx < ty && tx < tz) {
        		outNormal = vec3(-sign(rd.x), 0.0, 0.0);
        	} else if (ty < tz) {
        		outNormal = vec3(0.0, -sign(rd.y), 0.0);
        	} else {
        		outNormal = vec3(0.0, 0.0, -sign(rd.z));
        	}

        	t = max(nextCellT, t) + 1e-4;
        	if (t > maxDist) break;

        	vec3 pos = ro + rd * t;

        	v = ivec3(floor(pos));

        	fr = fract(pos);

        	tMax.x = (rd.x > 0.0) ? t + (1.0 - fr.x) * tDelta.x : (rd.x < 0.0) ? t + fr.x * tDelta.x : 1e30;
            tMax.y = (rd.y > 0.0) ? t + (1.0 - fr.y) * tDelta.y : (rd.y < 0.0) ? t + fr.y * tDelta.y : 1e30;
            tMax.z = (rd.z > 0.0) ? t + (1.0 - fr.z) * tDelta.z : (rd.z < 0.0) ? t + fr.z * tDelta.z : 1e30;

            bv = v >> 3;

            brickTMax.x = (rd.x > 0.0) ? (float((bv.x + 1) * brickSize) - ro.x) * invRd.x : (rd.x < 0.0) ? (float(bv.x * brickSize) - ro.x) * invRd.x : 1e30;
            brickTMax.y = (rd.y > 0.0) ? (float((bv.y + 1) * brickSize) - ro.y) * invRd.y : (rd.y < 0.0) ? (float(bv.y * brickSize) - ro.y) * invRd.y : 1e30;
            brickTMax.z = (rd.z > 0.0) ? (float((bv.z + 1) * brickSize) - ro.z) * invRd.z : (rd.z < 0.0) ? (float(bv.z * brickSize) - ro.z) * invRd.z : 1e30;
        
            tboBrick = bv - brickOrigin;

            brickOccupied = rmBrickOccupied(tboBrick);

            continue;
        }

        if (!brickOccupied) {
        	float nextBrickT = min(min(brickTMax.x, brickTMax.y), brickTMax.z);
        	if (nextBrickT > maxDist) break;

            t = nextBrickT + 1e-4;

            if (t >= brickTMax.x) {
                bv.x += s.x;
                brickTMax.x += brickDelta.x;
            }

            if (t >= brickTMax.y) {
                bv.y += s.y;
                brickTMax.y += brickDelta.y;
            }

            if (t >= brickTMax.z) {
                bv.z += s.z;
                brickTMax.z += brickDelta.z;
            }

            vec3 pos = ro + rd * t;

            v = ivec3(floor(pos));
            fr = fract(pos);

            tMax.x = (rd.x > 0.0) ? t + (1.0 - fr.x) * tDelta.x : (rd.x < 0.0) ? t + fr.x * tDelta.x : 1e30;
            tMax.y = (rd.y > 0.0) ? t + (1.0 - fr.y) * tDelta.y : (rd.y < 0.0) ? t + fr.y * tDelta.y : 1e30;
            tMax.z = (rd.z > 0.0) ? t + (1.0 - fr.z) * tDelta.z : (rd.z < 0.0) ? t + fr.z * tDelta.z : 1e30;
            
            bv = v >> 3;

            ivec3 brickOrigin = ivec3((playerChunkX - 2) * 32, 0, (playerChunkZ - 2) * 32) >> 3;
    		ivec3 tboBrick = bv - brickOrigin;

            brickOccupied = rmBrickOccupied(tboBrick);

            continue;
        }

        if (rmGetVoxel(v)) {
            outHitPos = ro + rd * t;
            return true;
        }

    	if (tMax.x < tMax.y && tMax.x < tMax.z) {
    		v.x += s.x; 
    		t = tMax.x; 
    		tMax.x += tDelta.x; 
    		outNormal = vec3(-float(s.x), 0.0, 0.0);

    		if ((v.x >> 3) != bv.x) {
                bv.x += s.x;
                brickTMax.x += brickDelta.x;

                ivec3 brickOrigin = ivec3((playerChunkX - 2) * 32, 0, (playerChunkZ - 2) * 32) >> 3;
    			ivec3 tboBrick = bv - brickOrigin;

                brickOccupied = rmBrickOccupied(tboBrick);
            }

        } else if (tMax.y < tMax.z) {
            v.y += s.y; 
            t = tMax.y; 
            tMax.y += tDelta.y; 
            outNormal = vec3(0.0, -float(s.y), 0.0);

            if ((v.y >> 3) != bv.y) {
                bv.y += s.y;
                brickTMax.y += brickDelta.y;

                ivec3 brickOrigin = ivec3((playerChunkX - 2) * 32, 0, (playerChunkZ - 2) * 32) >> 3;
    			ivec3 tboBrick = bv - brickOrigin;

                brickOccupied = rmBrickOccupied(tboBrick);
            }

        } else {
            v.z += s.z; 
            t = tMax.z; 
            tMax.z += tDelta.z; 
            outNormal = vec3(0.0, 0.0, -float(s.z));

            if ((v.z >> 3) != bv.z) {
                bv.z += s.z;
                brickTMax.z += brickDelta.z;

                ivec3 brickOrigin = ivec3((playerChunkX - 2) * 32, 0, (playerChunkZ - 2) * 32) >> 3;
    			ivec3 tboBrick = bv - brickOrigin;

                brickOccupied = rmBrickOccupied(tboBrick);
            }

        }
    }

    return false;
}

vec3 rmSky(vec3 dir) {
	float sunH = sunDir.y;
	float day = smoothstep(-0.1, 0.3, sunH);

	vec3 c = mix(vec3(0.02, 0.02, 0.06), vec3(0.45, 0.60, 0.85), day);

	float sd = max(dot(dir, sunDir), 0.0);

	c += vec3(1.0, 0.85, 0.6) * pow(sd, float(128.0)) * 2.0;

	return c;
}

vec3 rmCosHemisphere(vec3 n, vec2 seed) {
    float u1 = hash(seed);
    float u2 = hash(seed + vec2(1.7, 3.1));
    float r = sqrt(u1);
    float phi = 6.2831853 * u2;

    vec3 local = vec3(r * cos(phi), sqrt(max(0.0, 1.0 - u1)), r * sin(phi));
    vec3 up = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, n));
    vec3 b = cross(n, t);

    return normalize(t * local.x + n * local.y + b * local.z);
}

vec3 aces(vec3 x) {
	const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;

	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 agx(vec3 x) {
    x = max(x, vec3(0.0));

    mat3 inset = mat3(
        0.842479, 0.042328, 0.042375,
        0.078433, 0.878468, 0.078433,
        0.079223, 0.079166, 0.879142
    );

    x = inset * x;
    x = log2(clamp(x, 1e-5, 65504.0));
    x = (x + 12.47393) / (12.47393 + 4.026069);
    x = clamp(x, 0.0, 1.0);

    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;

    return clamp(15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x, 0.0, 1.0);
}

float ign(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

float hgPhase(float cosTheta, float g) {
    float g2 = g * g;

    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 0.001), 1.5));
}



vec3 bloomThreshold(vec3 c) {
	float brightness = max(c.r, max(c.g, c.b));
	float knee = 0.25;
	float soft = clamp((brightness - uBloomThreshold + knee) / (2.0 * knee), 0.0, 1.0);
	float contribution = max(brightness - uBloomThreshold, 0.0) + soft * soft * knee;

	contribution /= max(brightness, 1e-5);

	return c * contribution;
}

vec3 reconstructWorld(vec2 uv, float depth) {
    vec4 h = invViewProj * vec4(
        uv * 2.0 - 1.0,
        depth * 2.0 - 1.0,
        1.0
    );

    return h.xyz / h.w;
}

vec3 computeVolumetricSunRays(vec2 uv, vec2 fragCoord) {
    float sunH = sunDir.y;
    if (sunH < -0.12) return vec3(0.0);

    float depth = texture(uSceneDepth,uv).r;

    bool isSky = depth >= 0.9999;

    vec3 worldPos = reconstructWorld(uv, depth);
    vec3 rayVec = worldPos - camPos;

    float dist = length(rayVec);

    vec3 rd = rayVec / max(dist, 1e-4);

    float maxDist = isSky ? 48.0 : min(dist, 80.0);
    float jitter = fract(ign(fragCoord) + float(uFrameIndex) * 0.6180339);

    const int steps1 = 10;

    float stepSize = maxDist / float(steps1);

    ivec3 brickOrigin = ivec3((playerChunkX - 2) * 32, 0, (playerChunkZ - 2) * 32) >> 3;

    float transmittance = 0.0;

    for (int i = 0; i < steps1; i++) {
        float t = (float(i) + jitter) * stepSize;

        vec3 samplePos = camPos + rd * t;

        float heightDensity = exp(-max(samplePos.t - 28.0, 0.0) * 0.035);

        bool inShadow = false;

        for (int s = 1; s <= 4; s++) {
            vec3 shadowPos = samplePos + sunDir * (float(s) * 8.0);

            ivec3 bv = (ivec3(floor(shadowPos)) >> 3) - brickOrigin;
            if (rmBrickOccupied(bv)) {
                inShadow = true;

                break;
            }
        }

        if (!inShadow) {
            transmittance += heightDensity * stepSize;
        }
    }

    float cosTheta = dot(rd, normalize(sunDir));
    float phase = clamp(hgPhase(cosTheta, 0.82), 0.0, 1.35);

    vec3 normalizedSun = uSunColor / max(max(uSunColor.r, max(uSunColor.g, uSunColor.b)), 1.0);

    float elevationFade = smoothstep(-0.12, 0.08, sunH);
    float skyAtten = isSky? 0.35 : 1.0;

    return normalizedSun * (transmittance * 0.0045) * phase * elevationFade * skyAtten;
}

vec3 colorGrade(vec3 c, float sunH) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.722));
    float shadowMask = 1.0 - smoothstep(0.0, 0.45, luma);
    float highlightMask = smoothstep(0.55, 1.0, luma);

    vec3 shadowTint = vec3(0.94, 0.98, 1.05);
    vec3 highlightTint = mix(vec3(1.06, 0.96, 0.88), vec3(1.03, 1.01, 0.98), smoothstep(0.0, 0.4, sunH));

    c *= mix(vec3(1.0), shadowTint, shadowMask * 0.6);
    c *= mix(vec3(1.0), highlightTint, highlightMask * 0.5);

    float maxChannel = max(c.r, max(c.g, c.b));
    float minChannel = min(c.r, min(c.g, c.b));
    float sat = maxChannel - minChannel;
    float vibrance = 0.22 * (1.0 - sat);

    c = mix(vec3(luma), c, 1.0 + vibrance);
    c = clamp(c, 0.0, 1.0);
    c = mix(c, c * c * (3.0 - 2.0 * c), 0.18);

    return c;
}

vec3 bloomBlur(vec2 uv) {
	vec2 t = uBloomTexelSize;

	vec3 result = texture(uBloomInput, uv).rgb * 0.227027;

	result += texture(uBloomInput, uv + t * 1.384615).rgb * 0.316216;
	result += texture(uBloomInput, uv - t * 1.384615).rgb * 0.316216;
	result += texture(uBloomInput, uv + t * 3.230769).rgb * 0.070270;
    result += texture(uBloomInput, uv - t * 3.230769).rgb * 0.070270;

    return result;
}

float linearDepth(float d) {
	float z = d * 2.0 - 1.0;

	return (2.0 * 0.1 * 800.0) / (800.0 + 0.1 - z * (800.0 - 0.1));
}

float giCoverage(int c, vec3 wp) {
    vec3 p = (wp - vec3(giOrigins[c])) / max(giVoxelSizes[c], 1e-3);
    vec3 e = min(p, float(uGiDim - 1) - p);

    return smoothstep(0.0, 4.0, min(e.x, min(e.y, e.z)));
}

vec3 giSampleCascade(int c, vec3 wp) {
    vec3 p = clamp((wp - vec3(giOrigins[c])) / max(giVoxelSizes[c], 1e-3), vec3(0.0), vec3(float(uGiDim - 1)));
    vec3 pf = floor(p);

    ivec3 b = ivec3(pf);

    vec3 f = p - pf;

    ivec3 c1 = min(b + ivec3(1), ivec3(uGiDim - 1));

    int base = c * uGiDim * uGiDim * uGiDim;

    #define Lgt(x,y,z) unpackUnorm4x8(giVoxels[base + ((z) * uGiDim + (y)) * uGiDim + (x)]).rgb

    vec3 v000 = Lgt(b.x, b.y, b.z);
    vec3 v100 = Lgt(c1.x, b.y, b.z);
    vec3 v010 = Lgt(b.x, c1.y, b.z);
    vec3 v110 = Lgt(c1.x, c1.y, b.z);
    vec3 v001 = Lgt(b.x, b.y, c1.z);
    vec3 v101 = Lgt(c1.x, b.y, c1.z);
    vec3 v011 = Lgt(b.x, c1.y, c1.z);
    vec3 v111 = Lgt(c1.x, c1.y, c1.z);

    #undef Lgt

    return mix(mix(mix(v000, v100, f.x), mix(v010, v110, f.x), f.y), mix(mix(v001, v101, f.x), mix(v011, v111, f.x), f.y), f.z);
}

vec3 giSample(vec3 wp) {
    float d = length(wp - camPos);
    if (d < 24.0) {
        return giSampleCascade(0, wp);
    }

    if (d < 32.0) {
        vec3 l0 = giSampleCascade(0, wp);
        vec3 l1 = giSampleCascade(1, wp);

        return mix(l0, l1, smoothstep(24.0, 32.0, d));
    }
    
    if (d < 56.0) {
        return giSampleCascade(1, wp);
    }
    
    vec3 l1 = giSampleCascade(1, wp);
    vec3 l2 = giSampleCascade(2, wp) * giCoverage(2, wp);
    if (d < 72.0) {
        return mix(l1, l2, smoothstep(56.0, 72.0, d));
    }
    
    return l2;
}

vec3 rgbToYCoCg(vec3 c) {
    return vec3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.50 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
    );
}

vec3 yCoCgToRgb(vec3 ycc) {
    return vec3(
        ycc.x + ycc.y - ycc.z,
        ycc.x + ycc.z,
        ycc.x - ycc.y - ycc.z
    );
}

vec3 applyMotionBlur(sampler2D tex, vec2 uv, vec2 velocity) {
    float speed = length(velocity);
    if (speed < 0.0015 || uMotionBlurScale <= 0.01) {
        return texture(tex, uv).rgb;
    }

    vec2 blurVec = velocity * uMotionBlurScale;

    float maxLen = 0.035;
    if (length(blurVec) > maxLen) {
        blurVec = normalize(blurVec) * maxLen;
    }

    vec3 sum = texture(tex, uv).rgb;

    sum += texture(tex, clamp(uv - blurVec * 0.33, 0.001, 0.999)).rgb;
    sum += texture(tex, clamp(uv - blurVec * 0.66, 0.001, 0.999)).rgb;
    sum += texture(tex, clamp(uv - blurVec * 1.00, 0.001, 0.999)).rgb;

    return sum * 0.25;
}

vec3 measuUpscale(sampler2D tex, vec2 uv) {
    vec2 t = uInternalTexelSize;
    if (uGgtuEnabled == 0) {
        vec2 texSize = 1.0 / t;
        vec2 snappedUv = (floor(uv * texSize) + 0.5) * t;

        return texture(tex, snappedUv).rgb;
    }

    vec3 c = texture(tex, uv).rgb;
    vec3 n = texture(tex, uv + vec2(0.0, t.y)).rgb;
    vec3 s = texture(tex, uv + vec2(0.0, -t.y)).rgb;
    vec3 e = texture(tex, uv + vec2(t.x, 0.0)).rgb;
    vec3 w = texture(tex, uv + vec2(-t.x, 0.0)).rgb;
    vec3 nw = texture(tex, uv + vec2(-t.x, t.y)).rgb;
    vec3 ne = texture(tex, uv + vec2(t.x, t.y)).rgb;
    vec3 sw = texture(tex, uv + vec2(-t.x, -t.y)).rgb;
    vec3 se = texture(tex, uv + vec2(t.x, -t.y)).rgb;

    vec3 lumaW = vec3(0.2126, 0.7152, 0.0722);

    float lC = dot(c, lumaW);
    float lN = dot(n, lumaW); 
    float lS = dot(s, lumaW); 
    float lE = dot(e, lumaW);
    float lW = dot(w, lumaW);
    float lNW = dot(nw, lumaW);
    float lNE = dot(ne, lumaW);
    float lSW = dot(sw, lumaW);
    float lSE = dot(se, lumaW);
    float gradH = abs(lW - lE) + 0.5 * (abs(lNW - lNE) + abs(lSW - lSE));
    float gradV = abs(lN - lS) + 0.5 * (abs(lNW - lSW) + abs(lNE - lSE));
    float edgeDir = clamp((gradV - gradH) / max(gradV + gradH, 1e-4), -1.0, 1.0);

    vec3 alongH = 0.5 * (w + e);
    vec3 alongV = 0.5 * (n + s);
    vec3 edgeSmooth = mix(alongV, alongH, edgeDir * 0.5 + 0.5);

    float edgeStrength = clamp((gradH + gradV) * 1.2, 0.0, 0.35);

    vec3 bases = mix(c, edgeSmooth, edgeStrength);

    float minL = min(lC, min(min(lN, lS), min(lE, lW)));
    float maxL = max(lC, max(max(lN, lS), max(lE, lW)));
    float contrast = maxL - minL;
    float sharp = clamp(0.55 * (1.0 - contrast * 1.1), 0.12, 0.55);

    vec3 res = bases * (1.0 + 4.0 * sharp) - (n + s + e + w) * sharp;

    vec3 minRGB = min(c, min(min(n, s), min(e, w)));
    vec3 maxRGB = max(c, max(max(n, s), max(e, w)));

    return clamp(res, minRGB, maxRGB);
}


void main() {
	fragNormal = vec4(0.0);

    if (uMode == 13) {
        vec2 warpedUv = vSkyUv;

        if (uGgtuEnabled == 1) {
            float depth = texture(uSceneDepth, vSkyUv).r;
            
            vec3 worldPos = reconstructWorld(vSkyUv, depth);

            vec4 prevClip = prevViewProj * vec4(worldPos, 1.0);
            if (prevClip.w > 0.0) {
                warpedUv = clamp((prevClip.xy / prevClip.w) * 0.5 + 0.5, 0.001, 0.999);
            }
        }

        vec3 ac = measuUpscale(uAccum, warpedUv);
        vec3 bloom = texture(uBloomInput, warpedUv).rgb;

        ac += bloom * uBloomStrength;

        float vig = smoothstep(3.5, 0.8, distance(vSkyUv, vec2(0.5)));

        vig = mix(0.6, 1.0, vig);
        ac *= vig;

        float exposure = (lightingMode == 1) ? texelFetch(uExposureTex, ivec2(0), 0).r : 1.1;

        vec3 exposed = ac * exposure;
        vec3 c = (lightingMode == 1) ? mix(aces(exposed), agx(exposed), 0.5) : aces(exposed);

        c = pow(c, vec3(1.0 / 2.2));
        c = colorGrade(c, sunDir.y);

        fragColor = vec4(c, 1.0);
        return;
    }

	if (uMode == 12) {
		float topLod = float(textureQueryLevels(uLumTex)) - 1.0;
	    float logAvg = textureLod(uLumTex, vec2(0.5), topLod).r;
	    float avgLum = exp2(logAvg);
	    float exposureTarget = clamp(uExposureKey / max(avgLum, uAutoExposureFloor) * uExposureBias, uExposureMin, uExposureMax);
	    float prev = texelFetch(uExposureTex, ivec2(0), 0).r;

	    fragColor = vec4(mix(prev, exposureTarget, uAdaptFactor), 0.0, 0.0, 1.0);

	    return;
	}

	if (uMode == 11) {
		vec3 c = texture(uAccum, vSkyUv).rgb;

		float l = dot(c, vec3(0.2126, 0.7152, 0.0722));

		fragColor = vec4(log2(max(l, uAutoExposureFloor)), 0.0, 0.0, 1.0);

		return;
	}

	if (uMode == 10) {
		vec2 texel = 1.0 / vec2(textureSize(uSsaoInput, 0));

		vec4 result = vec4(0.0);

		float wsum = 0.0;
		float centerDepth = linearDepth(texture(uSceneDepth, vSkyUv).r);

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				vec2 uv = vSkyUv + vec2(x, y) * texel;

				float d = linearDepth(texture(uSceneDepth, uv).r);
				float depthWeight = exp(-abs(d - centerDepth) * 5.0);
				float spatialWeight = exp(-float(x * x + y * y) * 0.15);
				float w = depthWeight * spatialWeight;

				result += texture(uSsaoInput, uv) * w;

				wsum += w;
			}
		}

		fragColor = result / wsum;

		return;
	}

	if (uMode == 9) {
		vec2 px = vec2(textureSize(uScene, 0));
	    vec2 uv = gl_FragCoord.xy / px;

	    vec3 blur = vec3(0.0);

	    const int R = 6;

	    for (int y = -R; y <= R; ++y) {
	        for (int x = -R; x <= R; ++x) {
	            blur += texture(uScene, uv + vec2(x, y) / px).rgb;
	        }
	    }

	    blur /= float((2 * R + 1) * (2 * R + 1));

	    float l = dot(blur, vec3(0.299, 0.587, 0.114));

	    blur = mix(blur, vec3(l), 0.35);
	    blur *= vec3(0.75, 0.85, 1.05);

	    float t = uv.y;

	    blur += smoothstep(0.6, 0.0, t) * 0.15;

	    fragColor = vec4(blur, 0.80);
	    return;
	}

	if (uMode == 8) {
		vec2 uv = vSkyUv;

		float depth = texture(uSceneDepth, uv).r;

		if (depth >= 0.99999) {
			fragColor = vec4(1.0, 0.0, 0.0, 0.0);

			return;
		}

		vec3 fragPos = reconstructWorld(uv, depth);
		vec3 normal = texture(uSceneNormal, uv).xyz * 2.0 - 1.0;
		normal = normalize(normal);

		float fragZ = linearDepth(depth);

		vec3 randomVec = vec3(texture(uSsaoNoise, gl_FragCoord.xy / 4.0).xy * 2.0 - 1.0, 0.0);
		vec3 tangent = randomVec - normal * dot(randomVec, normal);
		tangent = normalize(tangent);

		vec3 bitTan = cross(normal, tangent);

		mat3 tbn = mat3(tangent, bitTan, normal);

		float occlusion = 0.0;

		for (int i = 0; i < 6; i++) {
			vec3 samplePos = fragPos + tbn * uSsaoKernel[i] * uSsaoRadius;

			vec4 clip = viewProj * vec4(samplePos, 1.0);
			if (clip.w <= 0.0) continue;

			vec2 sampleUv = clip.xy / clip.w * 0.5 + 0.5;
			if (sampleUv.x <= 0.0 || sampleUv.x >= 1.0 || sampleUv.y <= 0.0 || sampleUv.y >= 1.0) continue;

			float sceneZ = linearDepth(texture(uSceneDepth, sampleUv).r);
			float sampleZ = clip.w;
			float range = smoothstep(0.0, 1.0, uSsaoRadius / max(abs(fragZ - sceneZ), 1e-4));
			if (sceneZ < sampleZ - uSsaoBias) occlusion += range;
		}

		float aoRaw = 1.0 - (occlusion / 6.0) * uSsaoIntensity;
		aoRaw = clamp(aoRaw, 0.0, 1.0);

		float ao = mix(0.55, 1.0, aoRaw);

		fragColor = vec4(ao, 0.0, 0.0, 0.0);

		return;
	}

	if (uMode == 7) {
		vec2 uv = gl_FragCoord.xy / vec2(textureSize(uBloomInput, 0));

		fragColor = vec4(bloomBlur(uv), 1.0);

		return;
	}

	if (uMode == 6) {
		vec2 uv = vSkyUv;
		vec2 t = 1.0 / vec2(textureSize(uBloomInput, 0));

		vec3 c = (texture(uBloomInput, uv + t * vec2(-1.0, -1.0)).rgb + texture(uBloomInput, uv + t * vec2( 1.0, -1.0)).rgb + texture(uBloomInput, uv + t * vec2(-1.0,  1.0)).rgb + texture(uBloomInput, uv + t * vec2( 1.0,  1.0)).rgb) * 0.25;

		fragColor = vec4(bloomThreshold(c), 1.0);

		return;
	}

	if (uMode == 5) {
		vec2 size = vec2(textureSize(uAccum, 0));
		vec2 frag = gl_FragCoord.xy;

	    vec3 ac = measuUpscale(uAccum, vSkyUv);
	    vec3 bloom = texture(uBloomInput, vSkyUv).rgb;

	    ac += bloom * uBloomStrength;

	    vec2 uv = frag / size;

	    float vig = smoothstep(3.5, 0.8, distance(uv, vec2(0.5)));

	    vig = mix(0.6, 1.0, vig);
	    ac *= vig;

	    float exposure = (lightingMode == 1) ? texelFetch(uExposureTex, ivec2(0), 0).r : 1.1;

        vec3 exposed = ac * exposure;
	    vec3 c = (lightingMode == 1) ? mix(aces(exposed), agx(exposed), 0.5) : aces(exposed);

	    c = pow(c, vec3(1.0 / 2.2));
        c = colorGrade(c, sunDir.y);

	    if (uGiDebug == 1 && lightingMode == 1) {
	    	vec4 nearH = invViewProj * vec4(vSkyUv * 2.0 - 1.0, -1.0, 1.0);
		    vec4 farH  = invViewProj * vec4(vSkyUv * 2.0 - 1.0,  1.0, 1.0);

		    vec3 ro = camPos;
		    vec3 rd = normalize((farH.xyz / farH.w) - (nearH.xyz / nearH.w));
		    vec3 hn, hp;

		    if (rmDda(ro, rd, 256.0, hn, hp)) {
		        fragColor = vec4(pow(clamp(giSample(hp + hn * 0.75) * 2.0, 0.0, 1.0), vec3(0.5)), 1.0);
		    } else {
		        fragColor = vec4(0.0);
		    }

		    return;
	    }

	    fragColor = vec4(c, 1.0);
	    return;
	}

	if (uMode == 4) {
		ivec2 px = ivec2(gl_FragCoord.xy);

		vec3 s = texelFetch(uScene, px, 0).rgb;

		if (lightingMode == 1) {
			float ao = texture(uSsaoResult, vSkyUv).r;

			s *= ao;
            s += computeVolumetricSunRays(vSkyUv, gl_FragCoord.xy);
		}

		if (uGgtuEnabled == 1) {
            float depth = texelFetch(uSceneDepth, px, 0).r;
            
            vec3 worldPos = reconstructWorld(vSkyUv, depth);

            vec4 prevClip = prevViewProj * vec4(worldPos, 1.0);

            vec2 prevUv = (prevClip.xy / max(prevClip.w, 1e-4)) * 0.5 + 0.5;

            if (prevClip.w > 0.0 && prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0) {
                vec3 history = texture(uAccum, prevUv).rgb;

                float centerAo = (lightingMode == 1) ? texture(uSsaoResult, vSkyUv).r : 1.0;

                vec3 cN = rgbToYCoCg(texelFetch(uScene, px + ivec2( 0,  1), 0).rgb * centerAo);
                vec3 cS = rgbToYCoCg(texelFetch(uScene, px + ivec2( 0, -1), 0).rgb * centerAo);
                vec3 cE = rgbToYCoCg(texelFetch(uScene, px + ivec2( 1,  0), 0).rgb * centerAo);
                vec3 cW = rgbToYCoCg(texelFetch(uScene, px + ivec2(-1,  0), 0).rgb * centerAo);
                vec3 cC = rgbToYCoCg(s);

                vec3 minC = min(cC, min(min(cN, cS), min(cE, cW)));
                vec3 maxC = max(cC, max(max(cN, cS), max(cE, cW)));
                vec3 clampedHistory = yCoCgToRgb(clamp(rgbToYCoCg(history), minC, maxC));

                s = mix(clampedHistory, s, uBlend);
            }
        }

        fragColor = vec4(s, 1.0);

        return;
	}
	
	// if (uMode == 3) {
	// 	vec2 ndc = vSkyUv * 2.0 - 1.0;
	// 	vec4 nearH = invViewProj * vec4(ndc, -1.0, 1.0);
	// 	vec4 farH = invViewProj * vec4(ndc, 1.0, 1.0);
	// 	vec3 ro = camPos;
	// 	vec3 rd = normalize((farH.xyz / farH.w) - (nearH.xyz / nearH.w));
	// 	vec3 normal;
	// 	vec3 albedo;
	// 	vec3 color;
	// 	vec2 seed  = vSkyUv * 0.137 + vec2(timeOfDay * 137, timeOfDay * 91.0) + vec2(float(uFrameIndex) * 0.618034, float(uFrameIndex) * 0.381966);
	// 	vec3 hitPos;
	// 	if (!rmDda(ro, rd, 200.0, normal, hitPos)) {
	// 		color = rmSky(rd);
	// 	} else {
	// 		albedo = vec3(0.45, 0.40, 0.28);

	// 		vec3 sn, sp;

	// 		bool shadowed = rmDda(hitPos + normal * 0.01, normalize(sunDir), 64.0, sn, sp);

	// 		vec3 gi = vec3(0.0);
	// 		vec3 gn, gp;
	// 		vec3 giDir = rmCosHemisphere(normal, seed);
	// 		if (rmDda(hitPos + normal * 0.01, giDir, 32, gn, gp)) {
	// 			gi += vec3(0.15, 0.12, 0.10);
	// 		} else {
	// 			gi = rmSky(giDir) * 0.4;
	// 		}

	// 		float ndotl = max(dot(normal, normalize(sunDir)), 0.0);
	// 		float shadowFactor = shadowed ? 0.15 : 1.0;

	// 		vec3 direct = vec3(1.0, 0.92, 0.75) * ndotl * shadowFactor;
	// 		vec3 skyAmbient = mix(vec3(0.20, 0.15, 0.10), vec3(0.35, 0.45, 0.60), normal.y * 0.5 + 0.5);

	// 		color = albedo * (direct + gi * 0.7 + skyAmbient * 0.4);
	// 	}

	// 	fragColor = vec4(color, 1.0);

	// 	return;
	// }

	if (uMode == 2) {
		fragColor = vec4(renderSky(vSkyUv), 1.0);

		return;
	}

	if (uMode == 1) {
		fragColor = uColor;

		return;
	} 

	vec3 normal = vec3(0.0, 1.0, 0.0);

	if (vFace == 0u) normal = vec3(1, 0, 0);
	else if (vFace == 1u) normal = vec3(-1, 0, 0);
	else if (vFace == 2u) normal = vec3(0, 1, 0);
	else if (vFace == 3u) normal = vec3(0, -1, 0);
	else if (vFace == 4u) normal = vec3(0, 0, 1);
	else if (vFace == 5u) normal = vec3(0, 0, -1);

	vec3 texColor = texture(texAtlas, vec3(vUv, float(vLayer))).rgb;

	float sunHeight = sunDir.y;

	vec3 sunColor = mix(vec3(1.0, 0.50, 0.20), vec3(1.0, 0.95, 0.85), smoothstep(0.0, 0.4, sunHeight));

	float sunAmount = smoothstep(-0.1, 0.3, sunHeight);
	float ndotl  = max(dot(normal, sunDir), 0.0);

	vec3 color;

	fragNormal = vec4(normal * 0.5 + 0.5, 1.0);

	if (lightingMode == 0) {
		vec3 skyAmb = vec3(0.45, 0.55, 0.70);
		vec3 grdAmb = vec3(0.20, 0.15, 0.10);
		vec3 ambient = mix(grdAmb, skyAmb, normal.y * 0.5 + 0.5);
		vec3 lighting = ambient + sunColor * ndotl * 0.8 * sunAmount;

		color = texColor * lighting;
	} else {
        vec3 viewVec = vWorldPos - camPos;

		float viewDist = length(viewVec);
		
        vec3 viewDir = viewVec / max(viewDist, 0.0001);

        float vertexAo = mix(0.45, 1.0, vAo);
        float upFactor = normal.y;

        vec3 skyHemisphere = mix(uHorizonColor * 0.45, uSkyColor * 0.65, max(upFactor, 0.0));
        vec3 groundBounce = uHorizonColor * vec3(0.22, 0.18, 0.14);
        vec3 ambient = mix(groundBounce, skyHemisphere, smoothstep(-0.6, 0.6, upFactor)) * vertexAo * 0.65;
        vec3 indirect = (viewDist < 96.0) ? giSample(vWorldPos + normal * (giVoxelSizes[0] * 0.75)) * uGiStrength * vertexAo : skyHemisphere * 0.20 * uGiStrength * vertexAo;

        float wrapNdotL = max((dot(normal, sunDir) + 0.15) / 1.15, 0.0);
        float sharpNdotL = max(dot(normal, sunDir), 0.0);
        float diffuse = mix(wrapNdotL, sharpNdotL, 0.75);

        vec3 direct = uSunColor * diffuse * sunAmount;

        float sunShadow = 1.0;

        if (ndotl > 0.0 && viewDist < 36.0 && sunAmount > 0.01) {
            vec3 sn, sp;

            if (rmDda(vWorldPos + normal * 0.05, sunDir, 24.0, sn, sp)) {
                sunShadow = mix(uShadowFactor, 1.0, smoothstep(28.0, 38.0, viewDist));
            }
        }

        direct *= sunShadow;

		color = texColor * (ambient + direct + indirect);

        float fogDist = viewDist * 0.0045;
        float fogAmt = 1.0 - exp(-pow(fogDist, 1.6));
        float fogHeight = exp(-max(vWorldPos.y - 40.0, 0.0) * 0.02);

        fogAmt *= mix(0.55, 1.0, fogHeight);
        fogAmt = clamp(fogAmt, 0.0, 0.92);

        float sunForward = max(dot(viewDir, normalize(sunDir)), 0.0);

        vec3 aerialTint = mix(vec3(0.42, 0.56, 0.74), vec3(0.95, 0.78, 0.55), pow(sunForward, 4.0) * 0.8);
        vec3 fogColor = mix(aerialTint, uSkyColor * 0.9, 0.35) + uSunColor * pow(sunForward, 6.0) * 0.25;

        color = mix(color, fogColor, fogAmt);
	}

	if (vFace == 6u) {
		if (lightingMode == 0) {
			discard;
		}

		float dayAmount = smoothstep(-0.1, 0.3, sunDir.y);

		vec3 rayColor = mix(vec3(0.85, 0.40, 0.15), vec3(0.95, 0.90, 0.75), smoothstep(0.1, 0.5, sunDir.y));

		fragColor = vec4(rayColor * dayAmount * 0.35, 0.40);
		fragNormal = vec4(0.0, 1.0, 0.0, 1.0);

		return;
	}

	fragColor = vec4(color, 1.0);
}