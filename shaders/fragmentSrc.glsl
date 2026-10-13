#version 430 core

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragNormal;
layout(std430, binding = 0) readonly buffer GiVolume {
	uint giVoxels[];
};

in vec2 vUv;
in vec2 vSkyUv;
in vec3 vWorldPos;

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
uniform vec3 giOrigins[3];
uniform vec3 giVoxelSizes[3];

const int brickSize = 8;

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
    vec3 dir   = normalize((farH.xyz / farH.w) - (nearH.xyz / nearH.w));

    float sunHeight = sunDir.y;
    float dayAmount = smoothstep(-0.15, 0.25, sunHeight);

    vec3 nightSky = vec3(0.0001, 0.0002, 0.0005);
    vec3 daySky   = vec3(0.10, 0.27, 0.69);
    vec3 dawnDusk = vec3(0.69, 0.10, 0.02);

    vec3 skyColor = mix(nightSky, daySky, dayAmount);

    float sunsetMask = exp(-sunHeight * sunHeight * 25.0);
    vec2 hDir = normalize(vec2(dir.x, dir.z) + vec2(1e-6));
    vec2 hSun = normalize(vec2(sunDir.x, sunDir.z) + vec2(1e-6));
    float sunSide = smoothstep(-0.5, 1.0, dot(hDir, hSun));
    skyColor = mix(skyColor, dawnDusk, sunsetMask * sunSide * 0.9);

    float horizon = exp(-abs(dir.y) * 2.5);
    skyColor += vec3(0.15, 0.12, 0.08) * horizon * dayAmount;

    float sunDot = dot(dir, sunDir);
    float sunDisc = smoothstep(0.9993, 0.9996, sunDot);
    float sunGlow = pow(max(sunDot, 0.0), 128.0);
    vec3 sunColor = mix(vec3(1.0, 0.55, 0.25), vec3(1.0, 0.95, 0.85),
                        smoothstep(0.05, 0.4, sunHeight));
    skyColor += sunColor * sunGlow * 1.5 * max(dayAmount, 0.2);
    skyColor += sunColor * sunDisc * 8.0;

    float nightAmount = 1.0 - dayAmount;

    if (nightAmount > 0.02 && dir.y > 0.0) {
        float band = exp(-pow((dir.y - 0.35) * 3.0, 2.0));

        vec2 uv = dir.xz / max(dir.y, 0.08);

        float t = timeOfDay * 60.0;
        uv.x += t * 0.02;

        float curtain = fbm(uv * 0.7 + vec2(t * 0.04, 0.0));
        curtain = smoothstep(0.35, 0.70, curtain);

        float rays = fbm(vec2(uv.x * 2.2, uv.y * 0.12) + vec2(t * 0.02, 0.0));
        rays = smoothstep(0.45, 0.80, rays);

        float shape = curtain * (0.35 + 0.65 * rays);
        float altitude = smoothstep(-0.25, 0.35, dir.y - 0.35);
        vec3 green = vec3(0.18, 1.0, 0.55);
        vec3 pink = vec3(1.0, 0.30, 0.55);
        vec3 auroraColor = mix(green, pink, altitude);

        float patchs = fbm(uv * 0.15 + vec2(t * 0.008, 0.0));
        patchs = 0.4 + 0.6 * patchs;

        float intensity = band * shape * patchs * nightAmount;

       	skyColor += auroraColor * intensity * 0.55;
    }

    if (nightAmount > 0.001) {
        float band = exp(-pow((dir.y - 0.35) * 3.0, 2.0));
        vec2 auroraUV = dir.xz / (dir.y + 0.5) * 0.6;
        float t = timeOfDay * 40.0;
        float n1 = fbm(auroraUV * 1.2 + vec2(t * 0.15, 0.0));
        float n2 = fbm(auroraUV * 2.4 + vec2(-t * 0.08, t * 0.03));
        float curtain = n1 * 0.6 + n2 * 0.4;
        curtain = smoothstep(0.35, 0.75, curtain);
        vec3 green  = vec3(0.15, 0.95, 0.45);
        vec3 purple = vec3(0.55, 0.25, 0.85);
        vec3 auroraColor = mix(green, purple, smoothstep(0.3, 0.7, dir.y));
        float intensity = band * curtain * nightAmount;
        skyColor += auroraColor * intensity * 0.7;
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

bool rmDda(vec3 ro, vec3 rd, float maxDist, out vec3 outNormal, out vec3 outHitPos) {
	ivec3 v = ivec3(floor(ro));
	ivec3 s = ivec3(sign(rd));

	vec3 invRd = 1.0 / (rd + vec3(equal(rd, vec3(0.0))) * 1e-30);
	vec3 tDelta = abs(invRd);
	vec3 fr = fract(ro);
	vec3 tMax;

	tMax.x = (rd.x > 0.0) ? (1.0 - fr.x) * tDelta.x : fr.x * tDelta.x;
    tMax.y = (rd.y > 0.0) ? (1.0 - fr.y) * tDelta.y : fr.y * tDelta.y;
    tMax.z = (rd.z > 0.0) ? (1.0 - fr.z) * tDelta.z : fr.z * tDelta.z;

    ivec3 bv = v >> 3;

    vec3 brickDelta = vec3(brickSize) * abs(invRd);
    vec3 brickTMax;

    brickTMax.x = (rd.x > 0.0) ? (float((bv.x + 1) * brickSize) - ro.x) * invRd.x : (float(bv.x * brickSize) - ro.x) * invRd.x;
    brickTMax.y = (rd.y > 0.0) ? (float((bv.y + 1) * brickSize) - ro.y) * invRd.y : (float(bv.y * brickSize) - ro.y) * invRd.y;
    brickTMax.z = (rd.z > 0.0) ? (float((bv.z + 1) * brickSize) - ro.z) * invRd.z : (rd.z < 0.0) ? (float(bv.z * brickSize) - ro.z) * invRd.z : 1e30;

    ivec3 brickOrigin = ivec3((playerChunkX - 2) * 32, 0, (playerChunkZ - 2) * 32) >> 3;
    ivec3 tboBrick = bv - brickOrigin;

    bool brickOccupied = rmBrickOccupied(tboBrick);

    float t = 0.0;

    outNormal = vec3(0.0);
    outHitPos = vec3(0.0);

    for (int i = 0; i <= int(maxHeightTerrain) / 2; i++) {
    	if (t > maxDist) break;
        if (v.y < 0 || v.y >= maxHeightTerrain) break;

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

            tMax.x = (rd.x > 0.0) ? (1.0 - fr.x) * tDelta.x : fr.x * tDelta.x;
            tMax.y = (rd.y > 0.0) ? (1.0 - fr.y) * tDelta.y : fr.y * tDelta.y;
            tMax.z = (rd.z > 0.0) ? (1.0 - fr.z) * tDelta.z : fr.z * tDelta.z;
            
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

float fragAo(vec3 worldPos, vec3 normal) {
	ivec3 vox = ivec3(floor(worldPos - normal * 0.5));
	ivec3 inorm = ivec3(round(normal));

	vec3 absN = abs(normal);

	ivec3 uDir, vDir;

	float fu, fv;

	vec3 lp = worldPos - vec3(vox);

	if (absN.x > 0.5) {
		uDir = ivec3(0, 1, 0);
		vDir = ivec3(0, 0, 1);

		fu = lp.y;
		fv = lp.z;
	} else if (absN.y > 0.5) {
		uDir = ivec3(1, 0, 0);
		vDir = ivec3(0, 0, 1);

		fu = lp.x;
		fv = lp.z;
	} else {
		uDir = ivec3(1, 0, 0);
		vDir = ivec3(0, 1, 0);

		fu = lp.x;
		fv = lp.y;
	}

	#define corner(ud, vd) (rmGetVoxel(vox + (ud) + inorm) && rmGetVoxel(vox +  (vd) + inorm)) ? 0.0 : float(3 - int(rmGetVoxel(vox + (ud) + inorm)) - int(rmGetVoxel(vox + (vd) + inorm)) - int(rmGetVoxel(vox + (ud) + (vd) + inorm)))

	float bl = corner(-uDir, -vDir);
	float br = corner( uDir, -vDir);
	float tr = corner( uDir,  vDir);
	float tl = corner(-uDir,  vDir);

	#undef corner

	return mix(mix(bl, br, fu), mix(tl, tr, fu), fv);
}

vec3 bloomThreshold(vec3 c) {
	float brightness = max(c.r, max(c.g, c.b));
	float knee = 0.25;
	float soft = clamp((brightness - uBloomThreshold + knee) / (2.0 * knee), 0.0, 1.0);
	float contribution = max(brightness - uBloomThreshold, 0.0) + soft * soft * knee;

	contribution /= max(brightness, 1e-5);

	return c * contribution;
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

vec3 reconstructWorld(vec2 uv, float depth) {
	vec4 h = invViewProj * vec4(
		uv * 2.0 - 1.0,
		depth * 2.0 - 1.0,
		1.0
	);

	return h.xyz / h.w;
}

float linearDepth(float d) {
	float z = d * 2.0 - 1.0;

	return (2.0 * 0.1 * 800.0) / (800.0 + 0.1 - z * (800.0 - 0.1));
}

vec3 giSampleCascade(int c, vec3 wp) {
	vec3 p = (wp - giOrigins[c]) / max(giVoxelSizes[c], 1e-3);
	if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(63.0)))) return vec3(0.0);

	vec3 pf = floor(p);

	ivec3 b = ivec3(pf);

	vec3 f = p - pf;

	ivec3 c1 = min(b + ivec3(1), ivec3(63));

	int base = c * 64 * 64 * 64;

	#define Lgt(x,y,z) unpackUnorm4x8(giVoxels[base + ((z) * 64 + (y)) * 64 + (x)]).rgb

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
	if (d < 28.0) return giSampleCascade(0, wp);
	if (d < 56.0) {
		float t = smoothstep(28.0, 40.0, d);

		return mix(giSampleCascade(0, wp), giSampleCascade(1, wp), t);
	}

	if (d < 112.0) {
		float t = smoothstep(56.0, 80.0, d);

		return mix(giSampleCascade(1, wp), giSampleCascade(2, wp), t);
	}

	return giSampleCascade(2, wp);
}

void main() {
	fragNormal = vec4(0.0);

	if (uMode == 12) {
		float topLod = float(textureQueryLevels(uLumTex)) - 1.0;
	    float logAvg = textureLod(uLumTex, vec2(0.5), topLod).r;
	    float avgLum = exp2(logAvg);
	    float exposureTarget = clamp(uExposureKey / max(avgLum, 1e-4) * uExposureBias, 0.1, 4.0);
	    float prev = texelFetch(uExposureTex, ivec2(0), 0).r;

	    fragColor = vec4(mix(prev, exposureTarget, uAdaptFactor), 0.0, 0.0, 1.0);

	    return;
	}

	if (uMode == 11) {
		vec3 c = texture(uAccum, vSkyUv).rgb;

		float l = dot(c, vec3(0.2126, 0.7152, 0.0722));

		fragColor = vec4(log2(max(l, 1e-4)), 0.0, 0.0, 1.0);

		return;
	}

	if (uMode == 10) {
		vec2 texel = 1.0 / vec2(textureSize(uSsaoInput, 0));

		vec4 result = vec4(0.0);

		float wsum = 0.0;

		for (int x = -2; x <= 2; x++) {
			for (int y = -2; y <= 2; y++) {
				float w = exp(-float(x * x + y * y) * 0.5);

				result += texture(uSsaoInput, vSkyUv + vec2(x, y) * texel) * w;

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

		for (int i = 0; i < 12; i++) {
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

		float aoRaw = 1.0 - (occlusion / 12.0) * uSsaoIntensity;
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

	    vec3 ac = texture(uAccum, vSkyUv).rgb;
	    vec3 bloom = texture(uBloomInput, vSkyUv).rgb;

	    ac += bloom * uBloomStrength;

	    vec2 uv = frag / size;

	    float vig = smoothstep(3.5, 0.8, distance(uv, vec2(0.5)));
	    vig = mix(0.6, 1.0, vig);
	    ac *= vig;

	    float exposure = (lightingMode == 1) ? texelFetch(uExposureTex, ivec2(0), 0).r : 1.1;

	    vec3 c = aces(ac * exposure);

	    c = pow(c, vec3(1.0 / 2.2));

	    if (uGiDebug == 1 && lightingMode == 1) {
	    	vec4 nearH = invViewProj * vec4(vSkyUv * 2.0 - 1.0, -1.0, 1.0);
		    vec4 farH  = invViewProj * vec4(vSkyUv * 2.0 - 1.0,  1.0, 1.0);

		    vec3 ro = camPos;
		    vec3 rd = normalize((farH.xyz / farH.w) - (nearH.xyz / nearH.w));
		    vec3 hn, hp;

		    if (rmDda(ro, rd, 256.0, hn, hp)) {
		        fragColor = vec4(giSample(hp + hn * 0.75) * 4.0, 1.0);
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
		vec3 a = texelFetch(uAccum, px, 0).rgb;

		if (lightingMode == 1) {
			float ao = texture(uSsaoResult, vSkyUv).r;

			s *= ao;
		}

		vec3 m = mix(a, s, uBlend);

		fragColor = vec4(m, 1.0);

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
		vec3 rayOrigin = vWorldPos + normal * 0.05 + sunDir * 0.01;
		vec3 sn, sp;

		bool shadowed = false;
		if (ndotl > 0.0 && sunAmount > 0.0) shadowed = rmDda(rayOrigin, sunDir, 64.0, sn, sp);

		vec3 skyAmb = vec3(0.45, 0.55, 0.70);
		vec3 grdAmb = vec3(0.20, 0.15, 0.10);
		vec3 ambient = mix(grdAmb, skyAmb, normal.y * 0.5 + 0.5);

		float shadowFactor = shadowed ? uShadowFactor : 1.0;

		vec3 direct = ambient + sunColor * ndotl * sunAmount * shadowFactor;
		vec3 indirect = giSample(vWorldPos + normal * 1.25) * uGiStrength;

		color = texColor * (direct + indirect);
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