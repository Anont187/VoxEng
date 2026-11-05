#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

const int nGi = 64;
const uint giVoxels = 64u * 64u * 64u;

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

// original direct plus ambient lighting yes

layout(std430, binding = 4) buffer giBaseBuffer {
	vec4 giBase[];
};

// whats ping pong, its a buffer here, for biding 5 and 6

layout(std430, binding = 5) buffer giLightBufferA {
	vec4 giLightA[];
};

layout(std430, binding = 6) buffer giLightBufferB {
	vec4 giLightB[];
};

// hmm, 0 = direct lighting, 1 = a (singular) gi propagation iteration, and 2 is pack final lighting

uniform int uStage;

// propagation is 1 = read a and write b, 2 is write a and read b

uniform int uReadA;
uniform int uCascadeIndex;
uniform vec3 uSunDir;

uint voxelIndex(ivec3 p) {
	return uint((p.z * nGi + p.y) * nGi + p.x);
}

bool insideGrid(ivec3 p) {
	return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, ivec3(nGi)));
}

bool solidAt(ivec3 p) {
	if (!insideGrid(p)) return false;

	return voxelType[voxelIndex(p)] != 0u;
}

vec3 srgbToLinear(vec3 c) {
	bvec3 cutoff = lessThanEqual(c, vec3(0.04045));

	vec3 low = c / 12.92;
	vec3 high = pow((c + 0.055) / 1.055, vec3(2.4));

	return mix(high, low, cutoff);
}

vec3 albedoFor(uint type) {
	vec3 srgb = vec3(0.0);

	if (type == 1u) srgb = vec3(107.0, 107.0, 112.0) / 255.0;
	if (type == 2u) srgb = vec3(115.0, 82.0, 51.0) / 255.0;
	if (type == 3u) srgb = vec3(77.0, 133.0, 51.0) / 255.0;
	if (type == 4u) srgb = vec3(235.0, 240.0, 247.0) / 255.0;

	return srgbToLinear(srgb);
}

vec3 previousLight(uint i) {
	return (uReadA != 0) ? giLightA[i].rgb : giLightB[i].rgb;
}

void main() {
	ivec3 p = ivec3(gl_GlobalInvocationID.xyz);
	if (!insideGrid(p)) return;

	uint i = voxelIndex(p);
	uint type = voxelType[i];

	// this has 3 stages, 0, 1, 2, this is stage 0

	if (uStage == 0) {
		if (type == 0u) {
			giBase[i] = vec4(0.0);
			giLightA[i] = vec4(0.0);

			return;
		}

		vec3 albedo = albedoFor(type);
		vec3 n = vec3(0.0);

		// too many !solidAt, i dont know why dont ask me

		if (!solidAt(p + ivec3( 1, 0, 0))) n.x += 1.0;
        if (!solidAt(p + ivec3(-1, 0, 0))) n.x -= 1.0;

        if (!solidAt(p + ivec3(0,  1, 0))) n.y += 1.0;
        if (!solidAt(p + ivec3(0, -1, 0))) n.y -= 1.0;

        if (!solidAt(p + ivec3(0, 0,  1))) n.z += 1.0;
        if (!solidAt(p + ivec3(0, 0, -1))) n.z -= 1.0;

        vec3 directAndAmbient = vec3(0.0);

        if (dot(n, n) >= 1e-6) {
        	n = normalize(n);

        	float ndl = max(dot(n, normalize(uSunDir)), 0.0);
        	float sunHeight = uSunDir.y;
        	float day = smoothstep(-0.1, 0.3, sunHeight);

        	vec3 sunColor = mix(vec3(1.00, 0.50, 0.20), vec3(1.00, 0.95, 0.85), smoothstep(0.0, 0.4, sunHeight));

        	float dayAmount = smoothstep(-0.15, 0.25, uSunDir.y);

        	// here comes le shadows

        	bool shadowed = false;

        	if (ndl > 0.0) {
        		for (int s = 1; s <= 12; ++s) {
        			ivec3 q = p + ivec3(round(uSunDir * float(s)));
        			if (!insideGrid(q)) break;
        			if (solidAt(q)) {
        				shadowed = true;

        				break;
        			}
        		}
        	}

        	vec3 skyAmb = mix(vec3(0.01, 0.01, 0.05), vec3(0.45, 0.55, 0.70), dayAmount);
			vec3 grdAmb = mix(vec3(0.005, 0.005, 0.01), vec3(0.20, 0.15, 0.10), dayAmount);
			vec3 ambient = mix(grdAmb, skyAmb, n.y * 0.5 + 0.5);

			float shadowFactor = shadowed ? 0.10 : 1.0;

			vec3 directLight = sunColor * ndl * day * shadowFactor;

			directAndAmbient = albedo * directLight;
        }

        giBase[i] = vec4(directAndAmbient, 0.0);
        giLightA[i] = vec4(directAndAmbient, 0.0);

        return;
	}

	// stage 1 indirect light propagation

	if (uStage == 1) {
		const ivec3 dirs[6] = ivec3[6](ivec3(1, 0, 0), ivec3(-1, 0, 0), ivec3(0, 1, 0), ivec3(0, -1, 0), ivec3(0, 0, 1), ivec3(0, 0, -1));

		vec3 sum = vec3(0.0);

		int openCount = 0;

		for (int d = 0; d < 6; ++d) {
			ivec3 q = p + dirs[d];

			if (!insideGrid(q)) continue;
			if (type == 0u) {
				sum += previousLight(voxelIndex(q));
			} else if (!solidAt(q)) {
				sum += previousLight(voxelIndex(q));

				openCount++;
			}
		}

		vec3 result;
		if (type != 0u) {
			vec3 incoming = (openCount > 0) ? sum / float(openCount) : vec3(0.0);

			result = giBase[i].rgb + albedoFor(type) * incoming * 0.55;
		} else {
			result = sum * (1.0 / 6.0);
		}

		if (uReadA != 0) {
			giLightB[i] = vec4(result, 0.0);
		} else {
			giLightA[i] = vec4(result, 0.0);
		}

		return;
	}

	// stage 2..? pack final gi into exist ouput

	if (uStage == 2) {
		vec3 c = clamp(giLightB[i].rgb, vec3(0.0), vec3(1.0));

		uint r = uint(c.r * 255.0);
		uint g = uint(c.g * 255.0);
		uint b = uint(c.b * 255.0);
		uint packed1 = r | (g << 8u) | (b << 16u) | (255u << 24u);
		uint outputIndex  = uint(uCascadeIndex) * giVoxels + i;

		giOutput[outputIndex] = packed1;
	}
}