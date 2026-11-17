#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

const float multiBounceFeedbackMultiplier = 0.45;
// WARNING: unserious comments
// i guess ill put comments in compute because this is so hard to understand
// this buffer contains final gi output yes
layout(std430, binding = 0) buffer giOutputBuffer {
	uint giOutput[];
};

// the voxel material id uploaded by the cpu
// here: 0 = air, 1 = stone, 2 = dirt, 3 = grass, 4 = snow

layout(std430, binding = 3) readonly buffer voxelInputBuffer {
	uint voxelType[];
};

// 1 pass rcgi woo

layout(std430, binding = 4) buffer giHistoryBuffer {
	vec4 giHistory[];
};

// some uniforms

uniform int uCascadeIndex;
uniform vec3 uSunDir;
uniform int uFrameIndex;
uniform ivec3 giOrigins[3];
uniform float giVoxelSizes[3];
uniform int uGiDim;
uniform int uRayCount;

int nGi;

uint voxelIndex(ivec3 p) {
	nGi = int(uGiDim);
	return uint((p.z * nGi + p.y) * nGi + p.x);
}

bool insideGrid(ivec3 p) {
	nGi = int(uGiDim);
	return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, ivec3(nGi)));
}

vec3 albedoFor(uint type) {
	if (type == 1u) return vec3(0.15, 0.15, 0.17);
	if (type == 2u) return vec3(0.32, 0.24, 0.18);
	if (type == 3u) return vec3(0.07, 0.23, 0.03);
	if (type == 4u) return vec3(0.83, 0.87, 0.93);

	return vec3(0.0);
}

vec3 getRayDir(int r, float jitter) {
	float a = float(r) * 1.04719755 + jitter;
	float z = (float(r) + 0.5) / float(uRayCount) - 1.0;
	float rad = sqrt(max(0.0, 1.0 - z * z));

	return vec3(cos(a) * rad, z, sin(a) * rad);
}

vec3 sampleUpperCascade(int upperC, vec3 wp) {
	uint giVoxels = uint(uGiDim * uGiDim * uGiDim);

	vec3 p = clamp((wp - vec3(giOrigins[upperC])) / giVoxelSizes[upperC], vec3(0.0), vec3(float(uGiDim - 1)));

	uint idx = uint(upperC) * giVoxels + voxelIndex(ivec3(floor(p)));

	return unpackUnorm4x8(giOutput[idx]).rgb;
}

void main() {
	uint giVoxels = uint(uGiDim * uGiDim * uGiDim);

	nGi = int(uGiDim);

	ivec3 p = ivec3(gl_GlobalInvocationID.xyz);
	if (!insideGrid(p)) return;

	// no more stages, 1 pass

	uint localIdx = voxelIndex(p);
	uint cOffset = uint(uCascadeIndex) * giVoxels;
	uint gIdx = cOffset + localIdx;

	bool hasNearbySolid = false;

	for (int z = -1; z <= 1 && !hasNearbySolid; z++) {
		for (int y = -1; y <= 1 && !hasNearbySolid; y++) {
			for (int x = -1; x <= 1; x++) {
				ivec3 np = p + ivec3(x, y, z);
				if (insideGrid(np) && voxelType[cOffset + voxelIndex(np)] != 0u) {
					hasNearbySolid = true;
					break;
				}
			}
		}
	}

	if (!hasNearbySolid) {
		giHistory[gIdx] = vec4(0.0);
		giOutput[gIdx] = 0u;

		return;
	}

	float tStart = (uCascadeIndex == 0) ? 0.6 : (uCascadeIndex == 1) ? 4.0 : 10.0;
	float tEnd = (uCascadeIndex == 0) ? 4.0 : (uCascadeIndex == 1) ? 10.0 : 20.0;
	float dt = (tEnd - tStart) * 0.25;
	float jitter = fract(float(uFrameIndex) * 0.6180339 + float(localIdx) * 0.073) * 6.2831853;

	vec3 probeLocal = vec3(p) + 0.5;
	vec3 probeWorld = vec3(giOrigins[uCascadeIndex]) + probeLocal * giVoxelSizes[uCascadeIndex];

	float day = smoothstep(-0.1, 0.3, uSunDir.y);

    vec3 sunCol = mix(vec3(1.0, 0.50, 0.20), vec3(1.0, 0.95, 0.85), smoothstep(0.0, 0.4, uSunDir.y));
    vec3 skyCol = mix(vec3(0.01, 0.01, 0.04), vec3(0.45, 0.55, 0.70), day);
    vec3 rad = vec3(0.0);

    for (int r = 0; r < uRayCount; ++r) {
    	vec3 rd = getRayDir(r, jitter);

    	bool hit = false;

    	uint hitType = 0u;

    	ivec3 hitVox = p;

    	for (int s = 0; s < 4; ++s) {
    		ivec3 q = ivec3(floor(probeLocal + rd * (tStart + (float(s) + 0.5) * dt)));
    		if (!insideGrid(q)) break;

    		uint vt = voxelType[cOffset + voxelIndex(q)];
    		if (vt != 0u) {
    			hit = true;
    			hitType = vt;
    			hitVox = q;

    			break;
    		}
    	}

    	if (hit) {
    		vec3 albedo = albedoFor(hitType) * 1.8;

    		ivec3 airVox = hitVox - ivec3(round(rd));
    		ivec3 sampleVox = insideGrid(airVox) ? airVox : hitVox;

    		bool sh = insideGrid(hitVox + ivec3(round(uSunDir * 3.0))) && voxelType[cOffset + voxelIndex(hitVox + ivec3(round(uSunDir * 3.0)))] != 0u;

    		vec3 direct = albedo * sunCol * day * (sh ? 0.10 : 1.0);
    		vec3 bounce = albedo * giHistory[cOffset + voxelIndex(sampleVox)].rgb * multiBounceFeedbackMultiplier;

    		rad += direct + bounce;
    	} else if (uCascadeIndex < 2) {
    		rad += sampleUpperCascade(uCascadeIndex + 1, probeWorld + rd * (tEnd * giVoxelSizes[uCascadeIndex]));
    	} else {
    		rad += skyCol * (rd.y * 0.4 + 0.6);
    	}
    }

    vec3 blended = mix(giHistory[gIdx].rgb, rad / float(uRayCount), 0.20);

    giHistory[gIdx] = vec4(blended, 1.0);

    giOutput[gIdx] = packUnorm4x8(vec4(clamp(blended, 0.0, 1.0), 1.0));
}