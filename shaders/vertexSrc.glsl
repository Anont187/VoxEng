#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUv;
layout(location = 2) in float aLayer;
layout(location = 3) in float aFace;

uniform mat4 mvp;

out vec2 vUv;
out vec2 vSkyUv;
out vec3 vWorldPos;

flat out uint vLayer;
flat out uint vFace;

void main() {
	vUv = aUv;
	vSkyUv = aPos.xy * 0.5 + 0.5;
	vWorldPos = aPos;
	vLayer = uint(aLayer);
	vFace = uint(aFace);
	gl_Position = mvp * vec4(aPos, 1.0);
}