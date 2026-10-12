// dev is lazy so bear with it
// to change shader (via tapping tab), go to the bottom of the 
// fragment shader, find the else block after the lightingMode
// == 0, that else {...} block is the shader for realism itself

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <string>
#include <cstdlib>
#include "glm/glm/glm.hpp"
#include "glm/glm/gtc/matrix_transform.hpp"
#include "glm/glm/gtc/type_ptr.hpp"
#include <unordered_map>
#include <unordered_set>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static std::string readFile(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		std::cerr << "ERROR: Failed to open shader: " << path << "\n";

		return "";
	}

	std::stringstream ss;

	ss << f.rdbuf();

	return ss.str();
}

std::string vertexShaderSource = readFile("shaders/vertexSrc.glsl");
std::string fragmentShaderSource = readFile("shaders/fragmentSrc.glsl");

const char* vertexSrc = vertexShaderSource.c_str();
const char* fragmentSrc = fragmentShaderSource.c_str();

int pastWidth = 640;
int pastHeight = 480;

float timeOfDay = 0.15f;  // 0=dawn, 0.25=noon, 0.5=dusk, 0.75=midnight. Start mid-morning.
float cameraSpeed = 4.3f;
float sprintSpeed = 16.0f;
float baseSensitivity = 0.071f;
float bfov = 70.0f;
float zoomFov = 10.0f;
float zoomSensitivity = 0.015f;

float dayLengthSec = 360.0f;

const float PI = 3.14159265f;

constexpr int chunkSize = 32;
constexpr int worldHeight = 256;
constexpr int chunksPerColumn = worldHeight / chunkSize;
constexpr int renderRadius = 2;
constexpr int noiseStep = 2;
constexpr int noiseGridSize = chunkSize / noiseStep + 1;
constexpr int maxChunkSlots = 256;
constexpr int vboSlotBytes = 510 * 514;
constexpr int eboSlotBytes = 128 * 1024;
constexpr int brickSize = 8;
constexpr int brickDimX = (chunkSize * (renderRadius * 2 + 1)) / brickSize;
constexpr int brickDimY = worldHeight / brickSize;
constexpr int brickDimZ = brickDimX;
constexpr int giDim = 64;
constexpr int giCascades = 3;
constexpr int giVoxelsPerCascade = giDim * giDim * giDim;

bool giDirty = true;

float gravity = 28.0f;
float jumpVel = 8.0f;
double doubleTapTime = 0.25;
float eyeHeight = 1.6f;
float terminalVel = -50.0f;
float flyAcceleration = 30.0f;
float flyDrag = 6.0f;
float flyMaxSpeed = 10.0f;

constexpr bool kCurtainRays = false;
constexpr int baseSnowLine = worldHeight * 58 / 100;

const char* stringed = "basic window";
float renderScale = 0.76f;

float cfov = 70.0f;
float sensitivity = 0.071f;
float lastMouseX = 0.0f;
float lastMouseY = 0.0f;
float verticalVel = 0.0f;

double lastSpaceTap = 0.0;

bool isCameraRotateable = false;
bool firstMouseInput = true;
bool goWireFrame = false;
bool f1WasPressed = false;
bool flyMode = false;
bool spaceWasPressed = false;
bool isGrounded = false;
bool suppressJump = false;
bool chunksNeedUpdate = true;

glm::vec3 flyVelocity(0.0f);

struct drawElementsCommand {
	GLuint count;
	GLuint instanceCount;
	GLuint firstIndex;
	GLint baseVertex;
	GLuint baseInstance;

	glm::vec3 aabbMin;
	glm::vec3 aabbMax;
};

struct Chunk {
	uint32_t voxel[3][chunkSize][chunkSize] = {};
	uint8_t height[chunkSize][chunkSize] = {};
	uint8_t maxSolidY = 0;

	bool hasAnySolid = false;

	int chunkX;
	int chunkY;
	int chunkZ;
};

struct vertex {
	float x, y, z;

	uint8_t u, v;
	uint8_t layer;
	uint8_t face;
	uint8_t ao;
};

struct chunkMesh {
	std::vector<vertex> vertices;
	std::vector<uint32_t> indices;
};

struct frustum {
	glm::vec4 planes[6];
};

struct visCmd {
	const drawElementsCommand* cmd;
	float distSq;
};

enum : uint8_t {
	BlockTypeAir = 0,
	BlockTypeStone = 1,
	BlockTypeDirt = 2,
	BlockTypeGrass = 3,
	BlockTypeSnow = 4,
};

static_assert(vboSlotBytes % sizeof(vertex) == 0, "slot size must be a multiple of the vertex size");

static inline uint64_t chunkKey(int cx, int cy, int cz) {
	uint64_t ucx = static_cast<uint64_t>(cx + 1000000) & 0x1FFFFF;
    uint64_t ucy = static_cast<uint64_t>(cy + 1000000) & 0x1FFFFF;
    uint64_t ucz = static_cast<uint64_t>(cz + 1000000) & 0x1FFFFF;

    return (ucx << 42) | (ucy << 21) | ucz;
}

static inline int keyToCx(uint64_t k) {
	return static_cast<int>((k >> 42) & 0x1FFFFF) - 1000000;
}

static inline int keyToCy(uint64_t k) {
	return static_cast<int>((k >> 21) & 0x1FFFFF) - 1000000;
}

static inline int keyToCz(uint64_t k) {
	return static_cast<int>(k & 0x1FFFFF) - 1000000;
}

static inline float noiseFade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static inline float noiseLerp(float t, float a, float b) {
    return a + t * (b - a);
}

static inline float noiseGrad(int hash, float x, float y) {
    int h = hash & 7;
    float u = h < 4 ? x : y;
    float v = h < 4 ? y : x;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

static inline float perlin2d(float x, float y) {
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;

    x -= floorf(x);
    y -= floorf(y);

    float u = noiseFade(x);
    float v = noiseFade(y);

    int a = (X + Y * 57) & 255;
    int b = (X + 1 + Y * 57) & 255;
    int c = (X + (Y + 1) * 57) & 255;
    int d = (X + 1 + (Y + 1) * 57) & 255;

    float res = noiseLerp(v, noiseLerp(u, noiseGrad(a, x, y), noiseGrad(b, x - 1, y)), noiseLerp(u, noiseGrad(c, x, y - 1), noiseGrad(d, x - 1, y - 1)));

    return (res+ 1.0f) * 0.5f;
}

static inline int floorDiv(int a, int b) {
	int q = a / b, r = a % b;
	return (r != 0 && (r < 0) != (b < 0)) ? q - 1 : q;
}

static inline int sign(float v) {
	return (v > 0.0f) - (v < 0.0f);
}

static std::vector<visCmd> vis;

using chunkMap = std::unordered_map<uint64_t, Chunk>;

frustum extractFrustum(const glm::mat4& vp) {
	frustum f;
    f.planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]); // left
    f.planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]); // right
    f.planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]); // bottom
    f.planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]); // top
    f.planes[4] = glm::vec4(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]); // near
    f.planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]); // far

    for (auto& p : f.planes) {
    	float len = glm::length(glm::vec3(p));

    	if (len > 0.0f) p /= len;
    }

    return f;
}

GLuint loadTextureArray(const char** paths, int count, int texSize) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);

    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_SRGB8_ALPHA8, texSize, texSize, count, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    std::vector<unsigned char> fallback(texSize * texSize * 4);

    for (int i = 0; i < texSize * texSize; i++) {
    	fallback[i * 4 + 0] = 255;
    	fallback[i * 4 + 1] = 0;
    	fallback[i * 4 + 2] = 255;
    	fallback[i * 4 + 3] = 255;
    }

    for (int i = 0; i < count; i++) {
    	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, texSize, texSize, 1, GL_RGBA, GL_UNSIGNED_BYTE, fallback.data());
    }

    for (int i = 0; i < count; i++) {
        int w = 0, h = 0, ch = 0;

        stbi_set_flip_vertically_on_load(false);

        unsigned char* data = stbi_load(paths[i], &w, &h, &ch, 4);

        if (!data) {
            std::cerr << "Failed to load texture: " << paths[i] << '\n';
            continue;
        }

        if (w != texSize || h != texSize) {
        	std::cerr << "texture mismatch: " << paths[i] << " is " << w << "x" << h << ", expected " << texSize << "x" << texSize << '\n';
        	
        	stbi_image_free(data);
        	
        	continue;
        }

        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, w, h, 1, GL_RGBA, GL_UNSIGNED_BYTE, data);

        stbi_image_free(data);
    }

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

    return tex;
}

bool aabbInFrustum(const frustum& f, const glm::vec3& mn, const glm::vec3& mx) {
	for (const glm::vec4& p : f.planes) {
		glm::vec3 pv(p.x >= 0.0f ? mx.x : mn.x, p.y >= 0.0f ? mx.y : mn.y, p.z >= 0.0f ? mx.z : mn.z);

		if (p.x * pv.x + p.y * pv.y + p.z * pv.z + p.w < 0.0f) return false;
	}

	return true;
}

inline uint8_t getVoxel(const Chunk& chunk, int x, int y, int z) {
	return ((chunk.voxel[0][y][z] >> x) & 1) | (((chunk.voxel[1][y][z] >> x) & 1) << 1) | (((chunk.voxel[2][y][z] >> x) & 1) << 2);
}

inline void setVoxel(Chunk& chunk, int x, int y, int z, uint8_t t) {
	chunk.voxel[0][y][z] = (chunk.voxel[0][y][z] & ~(1u << x)) | (((t) & 1u) << x);
    chunk.voxel[1][y][z] = (chunk.voxel[1][y][z] & ~(1u << x)) | (((t >> 1) & 1u) << x);
    chunk.voxel[2][y][z] = (chunk.voxel[2][y][z] & ~(1u << x)) | (((t >> 2) & 1u) << x);
}

void addFace(chunkMesh& mesh, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d, uint8_t layer, uint8_t face, uint8_t uvW, uint8_t uvH, uint8_t aoA, uint8_t aoB, uint8_t aoC, uint8_t aoD) {
	uint32_t base = static_cast<uint32_t>(mesh.vertices.size());

	mesh.vertices.push_back({a.x, a.y, a.z, 0, 0, layer, face    , aoA});
	mesh.vertices.push_back({b.x, b.y, b.z, uvW, 0, layer, face  , aoB});
	mesh.vertices.push_back({c.x, c.y, c.z, uvW, uvH, layer, face, aoC});
	mesh.vertices.push_back({d.x, d.y, d.z, 0, uvH, layer, face  , aoD});

	mesh.indices.push_back(base + 0);
	mesh.indices.push_back(base + 1);
	mesh.indices.push_back(base + 2);

	mesh.indices.push_back(base + 2);
	mesh.indices.push_back(base + 3);
	mesh.indices.push_back(base + 0);
}

void addGreedyQuad(chunkMesh& mesh, int axis, int8_t value, int slice, int uStart, int vStart, int width, int height, const glm::vec3& chunkOffset) {
	glm::vec3 p0(0.0f);
	glm::vec3 pu(0.0f);
	glm::vec3 pv(0.0f);
	glm::vec3 puv(0.0f);

	int uAxis = (axis + 1) % 3;
	int vAxis = (axis + 2) % 3;

	p0[axis] = static_cast<float>(slice);
	p0[uAxis] = static_cast<float>(uStart);
	p0[vAxis] = static_cast<float>(vStart);

	pu = p0;
	pu[uAxis] += static_cast<float>(width);

	pv = p0;
	pv[vAxis] += static_cast<float>(height);

	puv = pu;
	puv[vAxis] += static_cast<float>(height);

	p0 += chunkOffset;
	pu += chunkOffset;
	pv += chunkOffset;
	puv += chunkOffset;

	int signDir = value > 0 ? 1 : -1;

	uint8_t type = static_cast<uint8_t>(std::abs(static_cast<int>(value)));

	uint16_t face;

	if (axis == 0) {
		face = (signDir > 0) ? 0 : 1;
	} else if (axis == 1) {
		face = (signDir > 0) ? 2 : 3;
	} else {
		face = (signDir > 0) ? 4 : 5;
	}

	uint8_t uvW = static_cast<uint8_t>(width);
	uint8_t uvH = static_cast<uint8_t>(height);
	uint8_t face8 = static_cast<uint8_t>(face);
	uint8_t layer = 0;

	if (type == BlockTypeGrass) {
		layer = (axis == 1 && signDir > 0) ? 1 : 0;
	} else if (type == BlockTypeStone) {
		layer = 2;
	} else if (type == BlockTypeSnow) {
		layer = (axis == 1 && signDir > 0) ? 3 : 2;
	} else if (type == BlockTypeDirt) {
		layer = 0;
	}

	if (signDir > 0) {
		addFace(mesh, p0, pu, puv, pv, layer, face8, uvW, uvH, 255, 255, 255, 255);
	} else {
		addFace(mesh, p0, pv, puv, pu, layer, face8, uvH, uvW, 255, 255, 255, 255);
	}
}

bool isSolid(const Chunk& chunks, int x, int y, int z) {
	if (x < 0 || x >= chunkSize || y < 0 || y >= chunkSize || z < 0 || z >= chunkSize) return false;
	return getVoxel(chunks, x, y, z) != 0;
}

const Chunk* getChunkAt(const chunkMap& chunks, int cx, int cy, int cz) {
	auto it = chunks.find(chunkKey(cx, cy, cz));

	return it != chunks.end() ? &it->second : nullptr;
}

bool isSolidAtWorld(const chunkMap& chunks, int x, int y, int z) {
	if (y < 0 || y >= worldHeight) return false;

	int cx = floorDiv(x, chunkSize);
	int cy = floorDiv(y, chunkSize);
	int cz = floorDiv(z, chunkSize);

	int lx = x - cx * chunkSize;
	int ly = y - cy * chunkSize;
	int lz = z - cz * chunkSize;

	const Chunk* c = getChunkAt(chunks, cx, cy, cz);

	if (!c) return false;
	if (ly > c->height[lx][lz]) return false;

	return isSolid(*c, lx, ly, lz);

}

chunkMesh buildChunkMesh(chunkMesh& out, const Chunk& chunk, int chunkX, int chunkY, int chunkZ, const Chunk* negX, const Chunk* posX, const Chunk* negY, const Chunk* posY, const Chunk* negZ, const Chunk* posZ) {
	out.vertices.clear();
	out.indices.clear();

	int8_t mask[chunkSize][chunkSize];

	glm::vec3 chunkOffset(chunkX * chunkSize, chunkY * chunkSize, chunkZ * chunkSize);

	for (int axis = 0; axis < 3; axis++) {

        int uAxis = (axis + 1) % 3;
        int vAxis = (axis + 2) % 3;

        int sliceMax = chunkSize;

        if (axis == 1) {
        	sliceMax = std::min(chunkSize, static_cast<int>(chunk.maxSolidY) + 2);
        }

        for (int slice = -1; slice < sliceMax; slice++) {

            for (int v = 0; v < chunkSize; v++) {
                for (int u = 0; u < chunkSize; u++) {

                    int x1 = 0, y1 = 0, z1 = 0;
                    int x2 = 0, y2 = 0, z2 = 0;

                    int a[3] = {0, 0, 0};
                    int b[3] = {0, 0, 0};

                    a[axis] = slice;
                    b[axis] = slice + 1;

                    a[uAxis] = u;
                    a[vAxis] = v;

                    b[uAxis] = u;
                    b[vAxis] = v;

                    x1 = a[0];
                    y1 = a[1];
                    z1 = a[2];

                    x2 = b[0];
                    y2 = b[1];
                    z2 = b[2];

                    uint8_t voxelA = false;
                    uint8_t voxelB = false;

                    auto sampleVoxel = [&](int x, int y, int z) -> uint8_t {
                    	if (x >= 0 && x < chunkSize && y >= 0 && y < chunkSize && z >= 0 && z < chunkSize) return getVoxel(chunk, x, y, z);

                    	if (x < 0)          return negX ? getVoxel(*negX, x + chunkSize, y, z) : 0;
						if (x >= chunkSize) return posX ? getVoxel(*posX, x - chunkSize, y, z) : 0;
						if (z < 0)          return negZ ? getVoxel(*negZ, x, y, z + chunkSize) : 0;
						if (z >= chunkSize) return posZ ? getVoxel(*posZ, x, y, z - chunkSize) : 0;
						if (y < 0)          return negY ? getVoxel(*negY, x, y + chunkSize, z) : 0;
						if (y >= chunkSize) return posY ? getVoxel(*posY, x, y - chunkSize, z) : 0;
						return false;
                    };

                    voxelA = sampleVoxel(x1, y1, z1);
                    voxelB = sampleVoxel(x2, y2, z2);

                    bool solidA = voxelA != 0;
                    bool solidB = voxelB != 0;

                    if (solidA != solidB) {
                        mask[u][v] = solidA ? int8_t(voxelA) : -int8_t(voxelB);
                    } else {
                        mask[u][v] = 0;
                    }
                }
            }

            for (int v = 0; v < chunkSize; v++) {
            	bool rowEmpty = true;

            	for (int u = 0; u < chunkSize; u++) {
            		if (mask[u][v] != 0) {
            			rowEmpty = false;

            			break;
            		}
            	}

            	if (rowEmpty) continue;

                for (int u = 0; u < chunkSize; ) {

                    int8_t value = mask[u][v];

                    if (value == 0) {
                        u++;
                        continue;
                    }

                    int width = 1;

                    while (u + width < chunkSize && mask[u + width][v] == value) {
                        width++;
                    }

                    int height = 1;

                    bool done = false;

                    while (v + height < chunkSize && !done) {

                        for (int x = 0; x < width; x++) {
                            if (mask[u + x][v + height] != value) {
                                done = true;
                                break;
                            }
                        }

                        if (!done)
                            height++;
                    }

                    int plane = slice + 1;

                    addGreedyQuad(out, axis, value, plane, u, v, width, height, chunkOffset);

                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            mask[u + x][v + y] = 0;
                        }
                    }

                    u += width;
                }
            }
        }
    }

    if (kCurtainRays && chunk.hasAnySolid) {
    	uint8_t faceFlag = 6u;
    	uint8_t layerFlag = 0u;

    	for (int z = 0; z < chunkSize - 1; z++) {
    		for (int x = 0; x < chunkSize - 1; x++) {
    			int currentSurfaceY = chunk.height[x][z];
    			int neighbourXSurfaceY = chunk.height[x + 1][z];
    			int neighbourZSurfaceY = chunk.height[x][z + 1];

    			if (currentSurfaceY != neighbourXSurfaceY) {
    				int minY = std::min(currentSurfaceY, neighbourXSurfaceY) + 1;
    				int maxY = std::max(currentSurfaceY, neighbourXSurfaceY);

    				if (maxY < chunkSize) {
    					float worldX = chunkOffset.x + static_cast<float>(x) + 1.0f;
    					float worldZStart = chunkOffset.z + static_cast<float>(z);
    					float worldZEnd = chunkOffset.z + static_cast<float>(z) + 1.0f;

    					uint32_t base = static_cast<uint32_t>(out.vertices.size());

    					out.vertices.push_back({worldX, chunkOffset.y + static_cast<float>(minY), worldZStart, 0, 0, layerFlag, faceFlag});
						out.vertices.push_back({worldX, chunkOffset.y + static_cast<float>(minY), worldZEnd, 1, 0, layerFlag, faceFlag});
						out.vertices.push_back({worldX, chunkOffset.y + static_cast<float>(maxY), worldZEnd, 1, 1, layerFlag, faceFlag});
						out.vertices.push_back({worldX, chunkOffset.y + static_cast<float>(maxY), worldZStart, 0, 1, layerFlag, faceFlag});

						out.indices.push_back(base + 0); 
						out.indices.push_back(base + 1); 
						out.indices.push_back(base + 2); 
						out.indices.push_back(base + 0);
						out.indices.push_back(base + 2); 
						out.indices.push_back(base + 3);
					}
				}

				if (currentSurfaceY != neighbourZSurfaceY) {
					int minY = std::min(currentSurfaceY, neighbourZSurfaceY) + 1;
					int maxY = std::max(currentSurfaceY, neighbourZSurfaceY);

					if (maxY < chunkSize) {
						float worldZ = chunkOffset.z + static_cast<float>(z) + 1.0f;
						float worldXStart = chunkOffset.x + static_cast<float>(x);
						float worldXEnd = chunkOffset.x + static_cast<float>(x) + 1.0f;

						uint32_t base = static_cast<uint32_t>(out.vertices.size());

						out.vertices.push_back({worldXStart, chunkOffset.y + static_cast<float>(minY), worldZ, 0, 0, layerFlag, faceFlag});
						out.vertices.push_back({worldXEnd,   chunkOffset.y + static_cast<float>(minY), worldZ, 1, 0, layerFlag, faceFlag});
						out.vertices.push_back({worldXEnd,   chunkOffset.y + static_cast<float>(maxY), worldZ, 1, 1, layerFlag, faceFlag});
						out.vertices.push_back({worldXStart, chunkOffset.y + static_cast<float>(maxY), worldZ, 0, 1, layerFlag, faceFlag});

						out.indices.push_back(base + 0); 
						out.indices.push_back(base + 1); 
						out.indices.push_back(base + 2); 
						out.indices.push_back(base + 0);
						out.indices.push_back(base + 2); 
						out.indices.push_back(base + 3);
    				}
    			}
    		}
    	}
    }

	return out;
}

static inline float sstep(float a, float b, float x) {
	float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);

	return t * t * (3.0f - 2.0f * t);
}

static float ridgedFbm(float x, float z) {
	float sum = 0.0f;
	float amp = 1.0f;
	float freq = 1.0f;
	float norm = 0.0f;
	float weight = 1.0f;

	for (int i = 0; i < 4; i++) {
		float n = 1.0f - std::fabs(perlin2d(x * freq + i * 19.3f, z * freq + i * 7.1f) * 2.0f - 1.0f);

		n *= n;
		n *= weight;

		weight = std::clamp(n * 1.6f, 0.0f, 1.0f);

		sum += n * amp;

		norm += amp;

		amp *= 0.5f;
		freq *= 2.1f;
	}

	return sum / norm;
}

int terrainHeight(int x, int z) {
	float wx = x + (perlin2d(x * 0.002f + 11.0f, z * 0.002f + 7.0f)  - 0.5f) * 220.0f;
	float wz = z + (perlin2d(x * 0.002f + 53.0f, z * 0.002f + 29.0f) - 0.5f) * 220.0f;
	float continental = perlin2d(x * 0.0012f, z * 0.0012f);
	float mountain = perlin2d(wx * 0.0022f, wz * 0.0022f);
	float hills = perlin2d(x * 0.010f, z * 0.010f);
	float detail = perlin2d(x * 0.04f, z * 0.04f);
	float mask = sstep(0.58f, 0.76f, mountain);
	float rid = ridgedFbm(wx * 0.0045f, wz * 0.0045f);

	const float h = static_cast<float>(worldHeight);

	float base = h * (0.10f + 0.14f * continental);
	float rolling = (hills - 0.5f) * h * 0.12f * (1.0f - 0.6f * mask);
	float rough = (detail - 0.5f) * h * (0.02f + 0.055f * mask);
	float peaks = mask * h * (0.08f + 0.58f * rid);

	return std::clamp(static_cast<int>(base + rolling + rough + peaks), 1, worldHeight - 1);
}

static void buildCoarseHeightGrid(int chunkX, int chunkZ, float grid[noiseGridSize][noiseGridSize]) {
	for (int gz = 0; gz < noiseGridSize; gz++) {
		for (int gx = 0; gx < noiseGridSize; gx++) {
			int wx = chunkX * chunkSize + gx * noiseStep;
			int wz = chunkZ * chunkSize + gz * noiseStep;

			grid[gx][gz] = static_cast<float>(terrainHeight(wx, wz));
		}
	}
}

static float sampleHeight(const float grid[noiseGridSize][noiseGridSize], int localX, int localZ) {
	int gx0 = localX / noiseStep;
    int gz0 = localZ / noiseStep;
    int gx1 = std::min(gx0 + 1, noiseGridSize - 1);
    int gz1 = std::min(gz0 + 1, noiseGridSize - 1);

    float fx = static_cast<float>(localX % noiseStep) / noiseStep;
    float fz = static_cast<float>(localZ % noiseStep) / noiseStep;
    float h00 = grid[gx0][gz0];
    float h10 = grid[gx1][gz0];
    float h01 = grid[gx0][gz1];
    float h11 = grid[gx1][gz1];
    float hx0 = h00 + (h10 - h00) * fx;
    float hx1 = h01 + (h11 - h01) * fx;

    return hx0 + (hx1 - hx0) * fz;
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void updFpsCounter(GLFWwindow* window, double time, GLuint visible, GLuint total, GLuint triangles) {
	static double prevT = 0.0;
	static int fc = 0;

	double ct = time;
	double et = ct - prevT;
	double mspf;

	if (et >= 0.5) {
		double fps = (double)fc / et;
		mspf = (fps > 0.0) ? (1000.0 / fps) : 0.0;

		char ts[256];

		std::snprintf(ts, sizeof(ts), "%s | FPS: %.1f | Frame Time: %.2fms | Chunk: %u/%u | Triangles: %u", stringed, fps, mspf, visible, total, triangles);

		glfwSetWindowTitle(window, ts);

		prevT = ct;
		fc = 0;
	}

	fc++;
}

bool playerCollides(const chunkMap& chunks, const glm::vec3& pos) {
    const float halfWidth = 0.3f;
    const float feetY = pos.y - eyeHeight;
    const float playerHeight = 1.8f;

    int minX = static_cast<int>(std::floor(pos.x - halfWidth));
    int maxX = static_cast<int>(std::floor(pos.x + halfWidth));

    int minY = static_cast<int>(std::floor(feetY));
    int maxY = static_cast<int>(std::floor(feetY + playerHeight));

    int minZ = static_cast<int>(std::floor(pos.z - halfWidth));
    int maxZ = static_cast<int>(std::floor(pos.z + halfWidth));

    for (int y = minY; y <= maxY; y++) {
        for (int z = minZ; z <= maxZ; z++) {
            for (int x = minX; x <= maxX; x++) {
                if (isSolidAtWorld(chunks, x, y, z)) {
                    return true;
                }
            }
        }
    }

    return false;
}

void movePlayer(const chunkMap& chunks, glm::vec3& pos, float dx, float dy, float dz) {
	if (dx != 0.0f) {
		float oldX = pos.x;

		pos.x += dx;

		if (playerCollides(chunks, pos)) pos.x = oldX;
	}

	if (dz != 0.0f) {
		float oldZ = pos.z;

		pos.z += dz;

		if (playerCollides(chunks, pos)) pos.z = oldZ;
	}

	if (dy != 0.0f) {
		float oldY = pos.y;

		pos.y += dy;

		if (playerCollides(chunks, pos)) {
			pos.y = oldY;

			if (dy < 0.0f) isGrounded = true;

			verticalVel = 0.0f;
		} else if (dy < 0.0f) {
			isGrounded = false;
		}
	}
}


bool resolveInsideBlock(const chunkMap& chunks, glm::vec3& pos) {
	if (!playerCollides(chunks, pos)) {
		return false;
	}

	int tries = 0;

	while (playerCollides(chunks, pos) && tries < 200) {
		pos.y += 1.0f;
		tries++;
	}

	return true;
}

bool blockOverlapsPlayer(const glm::vec3& pos, const glm::ivec3& block) {
    const float halfWidth = 0.3f;
    const float feetY = pos.y - eyeHeight;
    const float playerHeight = 1.8f;

    int minX = static_cast<int>(std::floor(pos.x - halfWidth));
    int maxX = static_cast<int>(std::floor(pos.x + halfWidth));
    int minY = static_cast<int>(std::floor(feetY));
    int maxY = static_cast<int>(std::floor(feetY + playerHeight));
    int minZ = static_cast<int>(std::floor(pos.z - halfWidth));
    int maxZ = static_cast<int>(std::floor(pos.z + halfWidth));

    return block.x >= minX && block.x <= maxX && block.y >= minY && block.y <= maxY && block.z >= minZ && block.z <= maxZ;
}

bool raycastVoxel(const chunkMap& chunks, const glm::vec3& origin, const glm::vec3& dir, float maxDist, glm::ivec3& hitVoxel, glm::ivec3& hitNormal, glm::ivec3& placeVoxel) {
	glm::ivec3 v(
		static_cast<int>(std::floor(origin.x)),
		static_cast<int>(std::floor(origin.y)),
		static_cast<int>(std::floor(origin.z))
	);

	int stepX = sign(dir.x);
	int stepY = sign(dir.y);
	int stepZ = sign(dir.z);

	float tDeltaX = (dir.x != 0.0f) ? std::abs(1.0f / dir.x) : 1e30f;
    float tDeltaY = (dir.y != 0.0f) ? std::abs(1.0f / dir.y) : 1e30f;
    float tDeltaZ = (dir.z != 0.0f) ? std::abs(1.0f / dir.z) : 1e30f;

    float tMaxX = (dir.x != 0.0f) ? ((stepX > 0 ? (v.x + 1 - origin.x) : (origin.x - v.x)) / std::abs(dir.x)) : 1e30f;
    float tMaxY = (dir.y != 0.0f) ? ((stepY > 0 ? (v.y + 1 - origin.y) : (origin.y - v.y)) / std::abs(dir.y)) : 1e30f;
    float tMaxZ = (dir.z != 0.0f) ? ((stepZ > 0 ? (v.z + 1 - origin.z) : (origin.z - v.z)) / std::abs(dir.z)) : 1e30f;

    glm::ivec3 normal(0);

    float t = 0.0f;

    while (t <= maxDist) {
        if (isSolidAtWorld(chunks, v.x, v.y, v.z)) {
            hitVoxel   = v;
            hitNormal  = normal;
            placeVoxel = v + normal;
            return true;
        }

        if (tMaxX < tMaxY && tMaxX < tMaxZ) {
            v.x += stepX;
            t = tMaxX;
            tMaxX += tDeltaX;
            normal = glm::ivec3(-stepX, 0, 0);
        } else if (tMaxY < tMaxZ) {
            v.y += stepY;
            t = tMaxY;
            tMaxY += tDeltaY;
            normal = glm::ivec3(0, -stepY, 0);
        } else {
            v.z += stepZ;
            t = tMaxZ;
            tMaxZ += tDeltaZ;
            normal = glm::ivec3(0, 0, -stepZ);
        }
    }

    return false;
}

void editVoxel(chunkMap& chunks, std::unordered_map<uint64_t, chunkMesh>& meshCache, const glm::ivec3& wp, uint8_t type) {
	if (wp.y < 0 || wp.y >= worldHeight) return;

	int cx = floorDiv(wp.x, chunkSize);
	int cy = floorDiv(wp.y, chunkSize);
	int cz = floorDiv(wp.z, chunkSize);
	int lx = wp.x - cx * chunkSize;
	int ly = wp.y - cy * chunkSize;
	int lz = wp.z - cz * chunkSize;

	auto it = chunks.find(chunkKey(cx, cy, cz));

	if (it == chunks.end()) return;

	Chunk& c = it->second;

	if (getVoxel(c, lx, ly, lz) == type) return;

	setVoxel(c, lx, ly, lz, type);

	giDirty = true;

	uint8_t& h = c.height[lx][lz];

	if (type != 0) {
		c.hasAnySolid = true;
		if (ly > h) h = static_cast<uint8_t>(ly);
		if (ly > c.maxSolidY) c.maxSolidY = static_cast<uint8_t>(ly);
	} else {
		if (ly >= h) {
			int newH = 0;

			for (int y = chunkSize - 1; y >= 0; --y) {
				if (getVoxel(c, lx, y, lz)) {
					newH = y;

					break;
				}
			}

			h = static_cast<uint8_t>(newH);
		}

		if (ly == c.maxSolidY) {
			uint8_t m = 0;

			for (int z = 0; z < chunkSize; z++) {
				for (int x = 0; x < chunkSize; x++) {
					if (c.height[x][z] > m) m = c.height[x][z];
				}
			}

			c.maxSolidY = m;
		}
	}

	meshCache.erase(chunkKey(cx, cy, cz));

	if (lx == 0) 			 meshCache.erase(chunkKey(cx - 1, cy, cz));
	if (lx == chunkSize - 1) meshCache.erase(chunkKey(cx + 1, cy, cz));
	if (lz == 0) 			 meshCache.erase(chunkKey(cx, cy, cz - 1));
	if (lz == chunkSize - 1) meshCache.erase(chunkKey(cx, cy, cz + 1));
	if (ly == 0)			 meshCache.erase(chunkKey(cx, cy - 1, cz));
	if (ly == chunkSize - 1) meshCache.erase(chunkKey(cx, cy + 1, cz));
}

void processInput(GLFWwindow* window, float deltaTime, glm::vec3& camPos, glm::vec3& camFront, const glm::vec3& up, const chunkMap& chunks) {
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    bool f1Pressed = glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS;

    if (f1Pressed && !f1WasPressed) {
    	goWireFrame = !goWireFrame;
    }

    f1WasPressed = f1Pressed;

    float tfov;

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
    	tfov = zoomFov;

    	sensitivity = zoomSensitivity;
    } else {
    	tfov = bfov;

    	sensitivity = baseSensitivity;
    }

    cfov += (tfov - cfov) * 10.0f * deltaTime;

    if (cfov < 1.0f) cfov = 1.0f;
    if (cfov > 90.0f) cfov = 90.0f;

    bool spacePressed = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
    bool justToggledFly = false;

    suppressJump = false;

    if (spacePressed && !spaceWasPressed) {
    	double now = glfwGetTime();

    	if (now - lastSpaceTap <= doubleTapTime) {
    		flyMode = !flyMode;
    		suppressJump = true;
    		justToggledFly = true;

    		if (flyMode) verticalVel = 0.0f;
    	}

    	lastSpaceTap = now;
    }

    spaceWasPressed = spacePressed;

    static bool ctrlShiftWasDown = false;
    static bool ctrlShiftToggleActive = false;

    bool ctrlShftPressed = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS && glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
    bool middleHeld = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS || ctrlShftPressed;

    if (ctrlShftPressed && !ctrlShiftWasDown) {
    	ctrlShiftToggleActive = !ctrlShiftToggleActive;
    }

    ctrlShiftWasDown = ctrlShftPressed;

    bool shouldRotateCamera = middleHeld || ctrlShiftToggleActive;

    if (shouldRotateCamera != isCameraRotateable) {
    	isCameraRotateable = shouldRotateCamera;

    	if (isCameraRotateable) {
	    	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	    	double mouseX, mouseY;
	    	glfwGetCursorPos(window, &mouseX, &mouseY);

	    	lastMouseX = static_cast<float>(mouseX);
	    	lastMouseY = static_cast<float>(mouseY);

	    	firstMouseInput = false;
	    } else {
	    	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	    }
	}

	float speed = cameraSpeed;

	static float yaw = -90.0f;
    static float pitch = -24.0f;

    bool stopLShift = false;

	if ((glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) && (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)) {
		speed = sprintSpeed;
		stopLShift = true;
	}

	float movementSpeed = speed * deltaTime;
	float moveX = 0.0f;
	float moveZ = 0.0f;

	glm::vec3 forward = glm::normalize(glm::vec3(camFront.x, 0.0f, camFront.z));
	glm::vec3 right = glm::normalize(glm::cross(camFront, up));

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    	moveX += forward.x * movementSpeed;
    	moveZ += forward.z * movementSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    	moveX -= forward.x * movementSpeed;
    	moveZ -= forward.z * movementSpeed;
    }
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		moveX -= right.x * movementSpeed;
		moveZ -= right.z * movementSpeed;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		moveX += right.x * movementSpeed;
		moveZ += right.z * movementSpeed;
	}

	glm::vec3 flyDirection(0.0f);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    	flyDirection += forward;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    	flyDirection -= forward;
    }
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		flyDirection -= right;	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		flyDirection += right;
	}

	if (flyDirection != glm::vec3(0.0f)) flyDirection = glm::normalize(flyDirection);

	if (flyMode && !justToggledFly) {
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) flyDirection.y += 1.0f;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS && !stopLShift) flyDirection.y -= 1.0f;

		flyVelocity += flyDirection * flyAcceleration * deltaTime;

		float flySpeed = glm::length(flyVelocity);

		if (flySpeed > flyMaxSpeed) flyVelocity = glm::normalize(flyVelocity) * flyMaxSpeed;
		if (flyDirection == glm::vec3(0.0f)) {
			float drag = flyDrag * deltaTime;

			if (flySpeed <= drag) {
				flyVelocity = glm::vec3(0.0f);
			} else {
				flyVelocity -= glm::normalize(flyVelocity) * drag;
			}
		}

		movePlayer(chunks, camPos, flyVelocity.x * deltaTime, flyVelocity.y * deltaTime, flyVelocity.z * deltaTime);

		verticalVel = 0.0f;
	} else {
		flyVelocity = glm::vec3(0.0f);

		if (!suppressJump && spacePressed && isGrounded) {
			verticalVel = jumpVel;
			isGrounded = false;
		}

		verticalVel -= gravity * deltaTime;

		if (verticalVel < terminalVel) verticalVel = terminalVel;
		
		movePlayer(chunks, camPos, moveX, verticalVel * deltaTime, moveZ);
		resolveInsideBlock(chunks, camPos);
	}

	if (!isCameraRotateable) {
		return;
	}

	double mouseX, mouseY;
	glfwGetCursorPos(window, &mouseX, &mouseY);

	if (firstMouseInput) {
		lastMouseX = static_cast<float>(mouseX);
		lastMouseY = static_cast<float>(mouseY);
		firstMouseInput = false;
	}

	float xOffset = static_cast<float>(mouseX) - lastMouseX;

    float yOffset = lastMouseY - static_cast<float>(mouseY);

    lastMouseX = static_cast<float>(mouseX);
    lastMouseY = static_cast<float>(mouseY);

    xOffset *= sensitivity;
    yOffset *= sensitivity;

    yaw += xOffset;
    pitch += yOffset;

    if (pitch > 89.0f)
        pitch = 89.0f;

    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 direction;

    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    camFront = glm::normalize(direction);

	return;
}

Chunk makeChunkFromGrid(int chunkX, int chunkY, int chunkZ, const float grid[noiseGridSize][noiseGridSize]) {
	Chunk chunk{};
	chunk.chunkX = chunkX;
	chunk.chunkY = chunkY;
	chunk.chunkZ = chunkZ;

	int worldY = chunkY * chunkSize;
	int heights[chunkSize][chunkSize];
	int globalHeights[chunkSize][chunkSize];

	uint8_t maxY = 0;

	bool anySolid = false;

	for (int z = 0; z < chunkSize; z++) {
		for (int x = 0; x < chunkSize; x++) {
			int globalH = static_cast<int>(std::floor(sampleHeight(grid, x, z)));
			globalHeights[x][z] = globalH;

			int localH = globalH - worldY;

			if (localH < 0) {
				heights[x][z] = -1;

				chunk.height[x][z] = 0;
			} else {
				if (localH > chunkSize - 1) localH = chunkSize - 1;

				heights[x][z] = localH;

				chunk.height[x][z] = static_cast<uint8_t>(localH);

				if (localH > maxY) maxY = static_cast<uint8_t>(localH);

				anySolid = true;
			}
		}
	}

	chunk.maxSolidY = maxY;
	chunk.hasAnySolid = anySolid;

	if (!anySolid) return chunk;

	const int steepLimit = std::max(3, worldHeight / 64);

	int8_t slope[chunkSize][chunkSize];

	int snowLineAt[chunkSize][chunkSize];
	int rockLineAt[chunkSize][chunkSize];

	float patch[chunkSize][chunkSize];

	for (int z = 0; z < chunkSize; z++) {
		for (int x = 0; x < chunkSize; x++) {
			int here = globalHeights[x][z];
			int hl = globalHeights[std::max(x - 2, 0)][z];
			int hr = globalHeights[std::min(x + 2, chunkSize - 1)][z];
			int hu = globalHeights[x][std::max(z - 2, 0)];
			int hd = globalHeights[x][std::min(z + 2, chunkSize - 1)];

			int s = std::max({ std::abs(here - hl), std::abs(here - hr), std::abs(here - hu), std::abs(here - hd) });
			slope[x][z] = static_cast<int8_t>(std::min(s, 127));

			int wx = chunkX * chunkSize + x;
			int wz = chunkZ * chunkSize + z;

			float j = (perlin2d(wx * 0.008f, wz * 0.008f) - 0.5f) * (worldHeight * 0.09f);

			snowLineAt[x][z] = baseSnowLine + static_cast<int>(j);
			rockLineAt[x][z] = baseSnowLine - worldHeight * 17 / 100 + static_cast<int>(j);
			patch[x][z] = perlin2d(wx * 0.05f + 90.0f, wz * 0.05f + 40.0f);
		}
	}

	for (int y = 0; y <= maxY; y++) {
		for (int z = 0; z < chunkSize; z++) {
			uint32_t r0 = 0, r1 = 0, r2 = 0;

			for (int x = 0; x < chunkSize; x++) {
				if (y > heights[x][z]) continue;

				int surfaceYWorld = globalHeights[x][z];
				int surfaceYLocal = surfaceYWorld - worldY;
				int depth = surfaceYLocal - y;
				int sl = slope[x][z];

				bool steep  = sl >= steepLimit;
				bool snowy  = surfaceYWorld > snowLineAt[x][z] && sl < steepLimit + 2;
				bool alpine = surfaceYWorld > rockLineAt[x][z];

				uint8_t surfaceType;

				if (snowy) surfaceType = BlockTypeSnow;
				else if (steep) surfaceType = BlockTypeStone;
				else if (alpine) surfaceType = (patch[x][z] > 0.55f) ? BlockTypeDirt : BlockTypeStone;
				else surfaceType = (patch[x][z] > 0.70f && surfaceYWorld > worldHeight * 23 / 100) ? BlockTypeDirt : BlockTypeGrass;

				uint8_t t;

				if (depth == 0) {
					t = surfaceType;
				} else if (surfaceType == BlockTypeSnow && depth < (sl < 2 ? 3 : 2)) {
					t = BlockTypeSnow;
				} else if (surfaceType == BlockTypeStone || surfaceType == BlockTypeSnow) {
					t = BlockTypeStone;
				} else if (depth <= 3) {
					t = BlockTypeDirt;
				} else {
					t = BlockTypeStone;
				}

				if (t & 1u) r0 |= 1u << x;
				if (t & 2u) r1 |= 1u << x;
				if (t & 4u) r2 |= 1u << x;
			}

			chunk.voxel[0][y][z] = r0;
			chunk.voxel[1][y][z] = r1;
			chunk.voxel[2][y][z] = r2;
		}
	}

	return chunk;
}

Chunk makeChunk(int chunkX, int chunkY, int chunkZ) {
	float grid[noiseGridSize][noiseGridSize];

	buildCoarseHeightGrid(chunkX, chunkZ, grid);

	return makeChunkFromGrid(chunkX, chunkY, chunkZ, grid);
}

static std::map<std::string, float> gSettings;
static std::string trimSpaces(const std::string& s) {
	const char* ws = " \t\r\n";

	size_t a = s.find_first_not_of(ws);

	if (a == std::string::npos) return "";

	size_t b = s.find_last_not_of(ws);

	return s.substr(a, b - a + 1);
}

static void loadSettings(const std::string& path) {
	std::ifstream file(path);
	if (!file) {
		std::cout << "SETTINGS: " << path << " not found, using built in defaults\n";

		return;
	}

	int skipped = 0;

	std::string line;

	while (std::getline(file, line)) {
		size_t cut = line.find("//");
		if (cut != std::string::npos) line.erase(cut);

		cut = line.find('#');
		if (cut != std::string::npos) line.erase(cut);

		size_t eq = line.find('=');
		if (eq == std::string::npos) continue;

		std::string left = trimSpaces(line.substr(0, eq));

		size_t sp = left.find_last_of(" \t*");

		std::string name = (sp == std::string::npos) ? left : left.substr(sp + 1);
		std::string val = line.substr(eq + 1);

		size_t semi = val.find(';');
		if (semi != std::string::npos) val.erase(semi);

		val = trimSpaces(val);
		if (name.empty() || val.empty()) continue;
		if (val == "true") {
			gSettings[name] = 1.0f;

			continue;
		}

		if (val == "false") {
			gSettings[name] = 0.0f;

			continue;
		}

		const char* start = val.c_str();

		char* end = nullptr;

		double d = std::strtod(start, &end);
		if (end == start) {
			skipped++;

			continue;
		}

		std::string rest = trimSpaces(std::string(end));
		if (!rest.empty() && rest != "f" && rest != "F") {
			skipped++;

			continue;
		}

		gSettings[name] = static_cast<float>(d);
	}

	std::cout << "SETTINGS: loaded " << gSettings.size() << " values from " << path;

	if (skipped > 0) std::cout << " (" << skipped << " lines skipped)";

	std::cout << "\n";
}

static float setting(const std::string& key, float fallback) {
	auto it = gSettings.find(key);

	return it != gSettings.end() ? it->second : fallback;
}

static int groundUnder(const chunkMap& ch, int wx, int wz, int fromY) {
	const int cx = floorDiv(wx, chunkSize);
	const int cz = floorDiv(wz, chunkSize);
	const int lx = wx - cx * chunkSize;
	const int lz = wz - cz * chunkSize;

	for (int wy = std::min(fromY, worldHeight - 1); wy >= 0; wy--) {
		const int cy = floorDiv(wy, chunkSize);

		auto it = ch.find(chunkKey(cx, cy, cz));
		if (it == ch.end()) continue;

		if (getVoxel(it->second, lx, wy - cy * chunkSize, lz) != 0) return wy;
	}

	return 0;
}

static void giFillAndLight(const chunkMap& ch, const glm::ivec3& origin, int vs, const glm::vec3& sunDir, std::vector<uint32_t>& out) {
    static const uint8_t palette[5][3] = {
        {0, 0, 0}, {107, 107, 112}, {115, 82, 51}, {77, 133, 51}, {235, 240, 247}
    };

    const int N  = giDim;
    const int NV = N * N * N;
    const Chunk* cached = nullptr;

    out.assign(NV, 0u);

    int cx0 = 0, cy0 = 0, cz0 = 0;

    bool haveCache = false;

    for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int wx = origin.x + x * vs;
                int wy = origin.y + y * vs;
                int wz = origin.z + z * vs;
                int cx = floorDiv(wx, chunkSize);
                int cy = floorDiv(wy, chunkSize);
                int cz = floorDiv(wz, chunkSize);

                if (!haveCache || cx != cx0 || cy != cy0 || cz != cz0) {
                    auto it = ch.find(chunkKey(cx, cy, cz));
                    cached = (it == ch.end()) ? nullptr : &it->second;

                    cx0 = cx; 
                    cy0 = cy; 
                    cz0 = cz; 
                    haveCache = true;
                }

                if (!cached) continue;

                uint8_t t = getVoxel(*cached, wx - cx * chunkSize, wy - cy * chunkSize, wz - cz * chunkSize);
                if (t == 0) continue;

                const uint8_t* c = palette[t];
                out[(z * N + y) * N + x] = uint32_t(c[0]) | (uint32_t(c[1]) << 8) | (uint32_t(c[2]) << 16) | (255u << 24);
            }
        }
    }

    auto isSolid = [&](int x, int y, int z) -> bool {
        if (x < 0 || x >= N || y < 0 || y >= N || z < 0 || z >= N) return false;
        
        return out[(z * N + y) * N + x] != 0u;
    };

    auto albedoAt = [&](int x, int y, int z) -> glm::vec3 {
        uint32_t v = out[(z * N + y) * N + x];

        if (v == 0u) return glm::vec3(0.0f);

        return glm::vec3(float(v & 0xFF), float((v >> 8) & 0xFF), float((v >> 16) & 0xFF)) / 255.0f;
    };

    float sunH = sunDir.y;
    float day  = sstep(-0.1f, 0.3f, sunH);

    glm::vec3 sunColor = glm::mix(glm::vec3(1.00f, 0.50f, 0.20f), glm::vec3(1.00f, 0.95f, 0.85f), sstep(0.0f, 0.4f, sunH));
    glm::vec3 skyAmb(0.45f, 0.55f, 0.70f);
    glm::vec3 grdAmb(0.20f, 0.15f, 0.10f);

    static std::vector<glm::vec3> L0, A, L, tmp;
    static std::vector<uint8_t> S;

    if ((int)L0.size() != NV) {
    	L0.resize(NV);
    	A.resize(NV);
    	L.resize(NV);
    	tmp.resize(NV);
    	S.resize(NV);
    }

    std::fill(L0.begin(), L0.end(), glm::vec3(0.0f));
    std::fill(A.begin(), A.end(), glm::vec3(0.0f));
	std::fill(L.begin(), L.end(), glm::vec3(0.0f));
    std::fill(S.begin(), S.end(), uint8_t(0));

    const int shadowSteps = 12;

    for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int i = (z * N + y) * N + x;
                if (out[i] == 0u) continue;

                S[i] = 1;
                A[i] = albedoAt(x, y, z);

                glm::vec3 n(0.0f);

                if (!isSolid(x+1, y, z)) n.x += 1.0f;
                if (!isSolid(x-1, y, z)) n.x -= 1.0f;
                if (!isSolid(x, y+1, z)) n.y += 1.0f;
                if (!isSolid(x, y-1, z)) n.y -= 1.0f;
                if (!isSolid(x, y, z+1)) n.z += 1.0f;
                if (!isSolid(x, y, z-1)) n.z -= 1.0f;
                if (glm::dot(n, n) < 1e-6f) continue;
                n = glm::normalize(n);

                float ndl = glm::max(glm::dot(n, sunDir), 0.0f);

                bool shadowed = false;
                if (ndl > 0.0f) {
                    for (int s = 1; s <= shadowSteps; s++) {
                        int qx = x + int(std::lround(sunDir.x * float(s)));
                        int qy = y + int(std::lround(sunDir.y * float(s)));
                        int qz = z + int(std::lround(sunDir.z * float(s)));
                        if (qx < 0 || qx >= N || qy < 0 || qy >= N || qz < 0 || qz >= N) break;
                        if (S[(qz * N + qy) * N + qx]) { shadowed = true; break; }
                    }
                }

                glm::vec3 ambient = glm::mix(grdAmb, skyAmb, n.y * 0.5f + 0.5f);

                float shadowF = shadowed ? 0.10f : 1.0f;

                L0[i] = A[i] * (ambient + sunColor * ndl * day * shadowF); 
                L [i] = L0[i];
            }
        }
    }

    const int   iterations = 3;
    const float bounceGain = 0.55f;

    for (int it = 0; it < iterations; it++) {
        tmp.swap(L);

        for (int z = 0; z < N; z++) {
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    int i = (z * N + y) * N + x;

                    glm::vec3 sum(0.0f);

                    if (x > 0) sum += tmp[i - 1];
                    if (x < N - 1) sum += tmp[i + 1];
                    if (y > 0) sum += tmp[i - N];
                    if (y < N - 1) sum += tmp[i + N];
                    if (z > 0) sum += tmp[i - N * N];
                    if (z < N - 1) sum += tmp[i + N * N];

                    sum *= (1.0f / 6.0f);

                    if (S[i]) {
                    	L[i] = L0[i] + A[i] * sum * bounceGain;
                    } else {
                    	L[i] = sum;
                    }
                }
            }
        }
    }

    for (int i = 0; i < NV; i++) {
        glm::vec3 c = glm::min(L[i], glm::vec3(1.0f));

        uint32_t r = uint32_t(c.x * 255.0f);
        uint32_t g = uint32_t(c.y * 255.0f);
        uint32_t b = uint32_t(c.z * 255.0f);

        out[i] = r | (g << 8) | (b << 16) | (255u << 24);
    }
}

int main() {
	std::cout << "STARTUP: START\n" << std::flush;
	std::cout << "STARTUP: Loading settings from settings.txt..." << std::flush;

	loadSettings("settings.txt");

	timeOfDay = setting("timeOfDay", timeOfDay);
    cameraSpeed = setting("cameraSpeed", cameraSpeed);
    sprintSpeed = setting("sprintSpeed", sprintSpeed);
    baseSensitivity = setting("baseSensitivity", baseSensitivity);
    bfov = setting("bfov", bfov);
    zoomFov = setting("zoomFov", zoomFov);
    zoomSensitivity = setting("zoomSensitivity", zoomSensitivity);
    dayLengthSec = setting("dayLengthSec", dayLengthSec);
    renderScale = setting("renderScale", renderScale);
    gravity = setting("gravity", gravity);
    jumpVel = setting("jumpVel", jumpVel);
    terminalVel = setting("terminalVel", terminalVel);
    doubleTapTime = setting("doubleTapTime", doubleTapTime);
    eyeHeight = setting("eyeHeight", eyeHeight);
    flyAcceleration = setting("flyAcceleration", flyAcceleration);
    flyDrag = setting("flyDrag", flyDrag);
    flyMaxSpeed = setting("flyMaxSpeed", flyMaxSpeed);

	if (!glfwInit()) {
		std::cout << "ERROR: cannot initialize GLFW.";
		return -1;
	}

	std::cout << "STARTUP: initialized glfw\n" << std::flush;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	std::cout << "STARTUP: Window hints made\n" << std::flush;

	GLFWwindow* window = glfwCreateWindow(pastWidth, pastHeight, "Loading...", nullptr, nullptr);

	std::cout << "STARTUP: Window made\n" << std::flush;

	if (!window) {
		std::cout << "ERROR: Could not make window.";
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	std::cout << "STARTUP: Made window context current\n" << std::flush;

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "ERROR: Could not initialize GLAD.";
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	std::cout << "STARTUP: initialized GLAD\n" << std::flush;

	const char* glVersion   = reinterpret_cast<const char*>(glGetString(GL_VERSION));
	const char* glVendor    = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
	const char* glRenderer  = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
	const char* glslVersion = reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION));
	
	std::cout << "STARTUP: GL_VERSION:   " << (glVersion   ? glVersion   : "?") << "\n";
	std::cout << "STARTUP: GL_VENDOR:    " << (glVendor    ? glVendor    : "?") << "\n";
	std::cout << "STARTUP: GL_RENDERER:  " << (glRenderer  ? glRenderer  : "?") << "\n";
	std::cout << "STARTUP: GLSL_VERSION: " << (glslVersion ? glslVersion : "?") << "\n";

	glViewport(0, 0, pastWidth, pastHeight);

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    std::cout << "STARTUP: Set Framebuffer size callback\n" << std::flush;

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	GLuint shaderProgram = glCreateProgram();

	int success;
    char infolog[512];

	glShaderSource(vertexShader, 1, &vertexSrc, nullptr);
	glCompileShader(vertexShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, sizeof(infolog), NULL, infolog);
        printf("Vertex Shader Ran into a problem:\n%s\n", infolog);
    }

	std::cout << "STARTUP: Made Vertex Shader\n" << std::flush;

	glShaderSource(fragmentShader, 1, &fragmentSrc, nullptr);
	glCompileShader(fragmentShader);

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, sizeof(infolog), NULL, infolog);
        printf("Fragment Shader Ran into a problem:\n%s\n", infolog);
    }


	std::cout << "STARTUP: Made Frament Shader\n" << std::flush;

	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	std::cout << "STARTUP: Made Shader Program\n" << std::flush;

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, sizeof(infolog), NULL, infolog);
        printf("Program linking Ran into a problem:\n%s\n", infolog);
    }

    if (success) {
    	std::cout << "STARTUP: No shader errors.\n" << std::flush;
    }

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	const char* texPaths[] = {
		"textures/dirt.png",
		"textures/grass.png",
		"textures/stone.png",
		"textures/snow.png"
	};

	GLuint texArray = loadTextureArray(texPaths, 4, 128);

    // float vertices[] = {
    // 	// position                  uv      layer face

	//     // Front
	//     -0.5f,-0.5f, 0.5f,          0,0,     0,0,
	//      0.5f,-0.5f, 0.5f,          1,0,     0,0,
	//      0.5f, 0.5f, 0.5f,          1,1,     0,0,
	//     -0.5f, 0.5f, 0.5f,          0,1,     0,0,

	//     // Back
	//      0.5f,-0.5f,-0.5f,          0,0,     0,0,
	//     -0.5f,-0.5f,-0.5f,          1,0,     0,0,
	//     -0.5f, 0.5f,-0.5f,          1,1,     0,0,
	//      0.5f, 0.5f,-0.5f,          0,1,     0,0,

	//     // Left
	//     -0.5f,-0.5f,-0.5f,          0,0,     0,0,
	//     -0.5f,-0.5f, 0.5f,          1,0,     0,0,
	//     -0.5f, 0.5f, 0.5f,          1,1,     0,0,
	//     -0.5f, 0.5f,-0.5f,          0,1,     0,0,

	//     // Right
	//      0.5f,-0.5f, 0.5f,          0,0,     0,0,
	//      0.5f,-0.5f,-0.5f,          1,0,     0,0,
	//      0.5f, 0.5f,-0.5f,          1,1,     0,0,
	//      0.5f, 0.5f, 0.5f,          0,1,     0,0,

	//     // Top
	//     -0.5f, 0.5f, 0.5f,          0,0,     0,0,
	//      0.5f, 0.5f, 0.5f,          1,0,     0,0,
	//      0.5f, 0.5f,-0.5f,          1,1,     0,0,
	//     -0.5f, 0.5f,-0.5f,          0,1,     0,0,

	//     // Bottom
	//     -0.5f,-0.5f,-0.5f,          0,0,     0,0,
	//      0.5f,-0.5f,-0.5f,          1,0,     0,0,
	//      0.5f,-0.5f, 0.5f,          1,1,     0,0,
	//     -0.5f,-0.5f, 0.5f,          0,1,     0,0
    // };

    // unsigned int indices[] = {
    // 	0,1,2, 2,3,0,       // front
	//     4,5,6, 6,7,4,       // back
	//     8,9,10, 10,11,8,    // left
	//     12,13,14, 14,15,12, // right
	//     16,17,18, 18,19,16, // top
	//     20,21,22, 22,23,20  // bottom
    // };

    chunkMap chunks;

    std::unordered_map<uint64_t, chunkMesh> meshCache;

    std::vector<drawElementsCommand> commands;
    std::vector<GLsizei> mdCounts;
    std::vector<const void*> mdIndices;

    std::unordered_map<uint64_t, uint32_t> chunkSlots;
    std::vector<uint32_t> freeSlots;

    GLsync slotFences[maxChunkSlots] = { nullptr };

    uint32_t nextFreeSlot = 0;

	std::cout << "STARTUP: Premade stuff\n" << std::flush;

    GLuint vao, vbo, ebo;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, maxChunkSlots * vboSlotBytes, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, maxChunkSlots * eboSlotBytes, nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, x));
    glEnableVertexAttribArray(0); 
 
    glVertexAttribPointer(1, 2, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, u));
    glEnableVertexAttribArray(1); 
 
    glVertexAttribPointer(2, 1, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, layer));
    glEnableVertexAttribArray(2); 
 
    glVertexAttribPointer(3, 1, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, face));
    glEnableVertexAttribArray(3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLuint lineVao, lineVbo;

    glGenVertexArrays(1, &lineVao);
    glGenBuffers(1, &lineVbo);

    glBindVertexArray(lineVao);

    glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
    glBufferData(GL_ARRAY_BUFFER, 24 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    GLuint sceneFbo, sceneTex, sceneDepthTex, sceneNormalTex;

    glGenFramebuffers(1, &sceneFbo);
    glGenTextures(1, &sceneTex);
    glGenTextures(1, &sceneDepthTex);
    glGenTextures(1, &sceneNormalTex);

    glBindTexture(GL_TEXTURE_2D, sceneTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, pastWidth, pastHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	
    glBindTexture(GL_TEXTURE_2D, sceneNormalTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, pastWidth, pastHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, sceneDepthTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, pastWidth, pastHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, sceneNormalTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sceneDepthTex, 0);

	GLenum sceneAttach[2] = {
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1
	};

	glDrawBuffers(2, sceneAttach);

	GLuint accumFbo[2], accumTex[2];
	GLuint bloomFbo[2], bloomTex[2];

	GLenum drawBuffers[2] = {
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1
	};

	glGenFramebuffers(2, bloomFbo);
	glGenTextures(2, bloomTex);

	int accumIdx = 0;
	int accumFrame = 0;

	glGenFramebuffers(2, accumFbo);
	glGenTextures(2, accumTex);

	for (int i = 0; i < 2; i++) {
	    glBindTexture(GL_TEXTURE_2D, accumTex[i]);
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, pastWidth, pastHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glBindFramebuffer(GL_FRAMEBUFFER, accumFbo[i]);
	    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTex[i], 0);
	}

	for (int i = 0; i < 2; i++) {
		glBindTexture(GL_TEXTURE_2D, bloomTex[i]);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, pastWidth, pastHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	    glBindFramebuffer(GL_FRAMEBUFFER, bloomFbo[i]);
	    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bloomTex[i], 0);
	}

	glm::vec3 camPos(16.0f, 35.0f, 45.0f);
    glm::vec3 camFront(0.0f, -0.4f, -0.9f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    std::srand(1337);

    std::vector<glm::vec3> ssaoKernel;
    std::vector<glm::vec4> ssaoNoise;
    std::vector<glm::vec2> ssgiOffsets;
    std::vector<float> blurWeights;

    float tanHalfFov = 0.0f;
    float aspect = 1.0f;

    glm::vec3 cameraRight(1.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraForward(0.0f, 0.0f, -1.0f);

    ssaoKernel.reserve(12);

    for (int i = 0; i < 12; i++) {
    	glm::vec3 s(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f, static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f, static_cast<float>(std::rand()) / RAND_MAX);

    	if (glm::dot(s, s) < 1e-6) {
    		s = glm::vec3(0.0f, 0.0f, 1.0f);
    	}

    	s = glm::normalize(s);

    	float t = float(i) / 11.0f;
    	float scale = 0.1f + 0.9f * t * t;

    	ssaoKernel.push_back(s * scale);
    }

    ssaoNoise.reserve(16);

    for (int i = 0; i < 16; i++) {
    	float x = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    	float y = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;

    	glm::vec2 n(x, y);

    	if (glm::dot(n, n) < 1e-6f) {
    		n = glm::vec2(1.0f, 0.0f);
    	}

    	n = glm::normalize(n);

    	ssaoNoise.emplace_back(n.x, n.y, 0.0f, 0.0f);
    }

    ssgiOffsets.reserve(8);

	for (int i = 0; i < 8; i++) {
	    float a = (static_cast<float>(i) / 8.0f) * 6.28318530718f;

	    ssgiOffsets.emplace_back(std::cos(a), std::sin(a));
	}

	blurWeights = {
	    0.227027f,
	    0.194594f,
	    0.121621f,
	    0.054054f,
	    0.016216f
	};

	int aoW0 = std::max(1, pastWidth / 2);
	int aoH0 = std::max(1, pastHeight / 2);

	GLuint aoFbo, aoTex, aoBlurFbo, aoBlurTex, ssaoNoiseTex;

	glGenFramebuffers(1, &aoFbo);
	glGenFramebuffers(1, &aoBlurFbo);
	glGenTextures(1, &aoTex);
	glGenTextures(1, &aoBlurTex);
	glGenTextures(1, &ssaoNoiseTex);

	for (auto [fbo, tex] : { std::pair{aoFbo, aoTex}, std::pair{aoBlurFbo, aoBlurTex} }) {
	    glBindTexture(GL_TEXTURE_2D, tex);
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, aoW0, aoH0, 0, GL_RGBA, GL_FLOAT, nullptr);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
	}

	glBindTexture(GL_TEXTURE_2D, ssaoNoiseTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 4, 4, 0, GL_RGBA, GL_FLOAT, ssaoNoise.data());

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

    int playerChunkX = static_cast<int>(std::floor(camPos.x / static_cast<float>(chunkSize)));
	int playerChunkZ = static_cast<int>(std::floor(camPos.z / static_cast<float>(chunkSize)));

    auto buildVoxelTbo = [&](const chunkMap& ch, int pcx, int pcz) {
    	std::vector<uint32_t> data(chunksPerColumn * 25 * 1024, 0u) ;

    	for (int cy = 0; cy < chunksPerColumn; cy++) {
    		for (int rz = 0; rz < 5; rz++) {
    			for (int rx = 0; rx < 5; rx++) {
    				int cx = pcx - 2 + rx;
    				int cz = pcz - 2 + rz;
    				int slot = cy * 25 + rz * 5 + rx;
    				int base = slot * 1024;

    				auto it = ch.find(chunkKey(cx, cy, cz));
    				if (it == ch.end()) continue;

    				const Chunk& c = it->second;
    				for (int y = 0; y < 32; y++) {
    					for (int z = 0; z < 32; z++) {
    						data[base + y * 32 + z] = c.voxel[0][y][z] | c.voxel[1][y][z] | c.voxel[2][y][z];
    					}
    				}
    			}   			
    		}
    	}

    	return data;
    };

    GLuint voxelTbo, voxelTboTex;

    glGenBuffers(1, &voxelTbo);
    glGenTextures(1, &voxelTboTex);

    auto uploadVoxelTbo = [&]() {
    	auto data = buildVoxelTbo(chunks, playerChunkX, playerChunkZ);

    	glBindBuffer(GL_TEXTURE_BUFFER, voxelTbo);
    	glBufferData(GL_TEXTURE_BUFFER, data.size() * sizeof(uint32_t), data.data(), GL_DYNAMIC_DRAW);

    	glBindTexture(GL_TEXTURE_BUFFER, voxelTboTex);
    	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, voxelTbo);
    };

    uploadVoxelTbo();

    GLuint brickTbo, brickTboTex;

    glGenBuffers(1, &brickTbo);
    glGenTextures(1, &brickTboTex);

    auto buildBrickTbo = [&](const chunkMap& ch, int pcx, int pcz) {
    	std::vector<uint32_t> data(brickDimX * brickDimY * brickDimZ, 0u);

    	int originX = (pcx - renderRadius) * chunkSize;
    	int originZ = (pcz - renderRadius) * chunkSize;

    	for (const auto& [k, c] : ch) {
    		int cwx = c.chunkX * chunkSize;
    		int cwy = c.chunkY * chunkSize;
    		int cwz = c.chunkZ * chunkSize;

    		for (int y = 0; y <= c.maxSolidY; y++) {
	            for (int z = 0; z < chunkSize; z++) {
	                uint32_t row = c.voxel[0][y][z] | c.voxel[1][y][z] | c.voxel[2][y][z];
	                if (!row) continue;

	                for (int x = 0; x < chunkSize; x++) {
	                    if (!(row & (1u << x))) continue;

	                    int bx = (cwx + x - originX) / brickSize;
	                    int by = (cwy + y)            / brickSize;
	                    int bz = (cwz + z - originZ) / brickSize;

	                    if (bx < 0 || bx >= brickDimX) continue;
	                    if (by < 0 || by >= brickDimY) continue;
	                    if (bz < 0 || bz >= brickDimZ) continue;

	                    data[(by * brickDimZ + bz) * brickDimX + bx] = 1u;
	                }
	            }
	        }
    	}

    	return data;
    };

    auto uploadBrickTbo = [&]() {
    	auto data = buildBrickTbo(chunks, playerChunkX, playerChunkZ);

    	glBindBuffer(GL_TEXTURE_BUFFER, brickTbo);
    	glBufferData(GL_TEXTURE_BUFFER, data.size() * sizeof(uint32_t), data.data(), GL_DYNAMIC_DRAW);

    	glBindTexture(GL_TEXTURE_BUFFER, brickTboTex);
    	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, brickTbo);
    };

    uploadBrickTbo();

    std::cout << "STARTUP: Made Buffers and ETC.\n" << std::flush;

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    std::cout << "STARTUP: Enabled Depth Test, And Culling\n" << std::flush;

    GLint mvpLoc = glGetUniformLocation(shaderProgram, "mvp");
    GLint texAtlasLoc = glGetUniformLocation(shaderProgram, "texAtlas");
    GLint sunDirLoc = glGetUniformLocation(shaderProgram, "sunDir");
    GLint uModeLoc = glGetUniformLocation(shaderProgram, "uMode");
    GLint uColorLoc = glGetUniformLocation(shaderProgram, "uColor");
    GLint invViewProjLoc = glGetUniformLocation(shaderProgram, "invViewProj");
    GLint camPosLoc = glGetUniformLocation(shaderProgram, "camPos");
    GLint timeOfDayLoc = glGetUniformLocation(shaderProgram, "timeOfDay");
    GLint maxHeightTerrainLoc = glGetUniformLocation(shaderProgram, "maxHeightTerrain");
    GLint voxelTboLoc = glGetUniformLocation(shaderProgram, "voxelTbo");
    GLint rmPlayerXLoc = glGetUniformLocation(shaderProgram, "playerChunkX");
    GLint rmPlayerZLoc = glGetUniformLocation(shaderProgram, "playerChunkZ");
    GLint brickTboLoc = glGetUniformLocation(shaderProgram, "brickTbo");
    GLint lightingModeLoc = glGetUniformLocation(shaderProgram, "lightingMode");
    GLint brickDimXLoc = glGetUniformLocation(shaderProgram, "brickDimX");
	GLint brickDimYLoc = glGetUniformLocation(shaderProgram, "brickDimY");
	GLint brickDimZLoc = glGetUniformLocation(shaderProgram, "brickDimZ");
	GLint uSceneLoc = glGetUniformLocation(shaderProgram, "uScene");
	GLint uAccumLoc = glGetUniformLocation(shaderProgram, "uAccum");
	GLint uBlendLoc = glGetUniformLocation(shaderProgram, "uBlend");
	GLint frameIdxLoc = glGetUniformLocation(shaderProgram, "uFrameIndex");
	GLint bloomInputLoc = glGetUniformLocation(shaderProgram, "uBloomInput");
	GLint bloomThresholdLoc = glGetUniformLocation(shaderProgram, "uBloomThreshold");
	GLint bloomStrengthLoc = glGetUniformLocation(shaderProgram, "uBloomStrength");
	GLint bloomTexelSizeLoc = glGetUniformLocation(shaderProgram, "uBloomTexelSize");
	GLint sceneDepthLoc = glGetUniformLocation(shaderProgram, "uSceneDepth");
	GLint sceneNormalLoc = glGetUniformLocation(shaderProgram, "uSceneNormal");
	GLint ssaoNoiseLoc = glGetUniformLocation(shaderProgram, "uSsaoNoise");
	GLint ssaoTexelSizeLoc = glGetUniformLocation(shaderProgram, "uSsaoTexelSize");
	GLint ssaoRadiusLoc = glGetUniformLocation(shaderProgram, "uSsaoRadius");
	GLint ssaoBiasLoc = glGetUniformLocation(shaderProgram, "uSsaoBias");
	GLint ssaoIntensityLoc = glGetUniformLocation(shaderProgram, "uSsaoIntensity");
	GLint ssgiOffsetsLoc = glGetUniformLocation(shaderProgram, "uSsgiOffsets[0]");
	GLint ssaoResultLoc = glGetUniformLocation(shaderProgram, "uSsaoResult");
	GLint ssaoInputLoc = glGetUniformLocation(shaderProgram, "uSsaoInput");
	GLint viewProjLoc = glGetUniformLocation(shaderProgram, "viewProj");
	GLint shadowFactorLoc = glGetUniformLocation(shaderProgram, "uShadowFactor");
	GLint giStrengthLoc = glGetUniformLocation(shaderProgram, "uGiStrength");
	GLint aeLumLoc = glGetUniformLocation(shaderProgram, "uLumTex");
    GLint aeExpLoc = glGetUniformLocation(shaderProgram, "uExposureTex");
    GLint aeAdaptLoc = glGetUniformLocation(shaderProgram, "uAdaptFactor");
    GLint giOriginsLoc = glGetUniformLocation(shaderProgram, "giOrigins");
    GLint giVoxelSizesLoc = glGetUniformLocation(shaderProgram, "giVoxelSizes");

    std::cout << "STARTUP: Made LOOKUP locations for glUniform\n" << std::flush;

	double time = 0;
	double lastFrame = 0;

	int width, height;

	auto allocSlot = [&](uint64_t key) -> uint32_t {
		auto it = chunkSlots.find(key);
		if (it != chunkSlots.end()) return it->second;

		uint32_t s = UINT32_MAX;
		if (!freeSlots.empty()) {
			for (auto freeIt = freeSlots.begin(); freeIt != freeSlots.end(); ++freeIt) {
				uint32_t candidateSlot = *freeIt;

				GLsync fence = slotFences[candidateSlot];

				if (fence == nullptr) {
					s = candidateSlot;

					freeSlots.erase(freeIt);

					break;
				}

				GLenum status = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
				if (status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED) {
					glDeleteSync(fence);

					slotFences[candidateSlot] = nullptr;

					s = candidateSlot;

					freeSlots.erase(freeIt);

					break;
				}
			}

			if (s == UINT32_MAX) {
				s = freeSlots.front();

				freeSlots.erase(freeSlots.begin());
				if (slotFences[s] != nullptr) {
					glClientWaitSync(slotFences[s], GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000);

					glDeleteSync(slotFences[s]);

					slotFences[s] = nullptr;
				}
			}
		} else {
			if (nextFreeSlot >= (uint32_t)maxChunkSlots) {
				std::cerr << "ERROR: slot overflow";
				return UINT32_MAX;
			}

			s = nextFreeSlot++;
		}

		chunkSlots[key] = s;

		return s;
	};

	auto freeSlot = [&](uint64_t key) {
		auto it = chunkSlots.find(key);
		if (it == chunkSlots.end()) return;

		freeSlots.push_back(it->second);

		chunkSlots.erase(it);
	};

	constexpr uint32_t vertsPerSlot = vboSlotBytes / sizeof(vertex);

	auto uploadMeshToSlot = [&](uint32_t slot, const chunkMesh& mesh) -> bool {
		if (mesh.vertices.empty() || mesh.indices.empty()) return false;

		size_t vBytes = mesh.vertices.size() * sizeof(vertex);
		size_t eBytes = mesh.indices.size() * sizeof(uint32_t);

		if (vBytes > vboSlotBytes || eBytes > eboSlotBytes) {
			std::cerr << "ERROR: mesh too big for slot: " << vBytes << "/" << vboSlotBytes << " bytes, " << eBytes << "/" << eboSlotBytes << " indices\n";

			return false;
		}

		uint32_t baseVertex = slot * vertsPerSlot;

		std::vector<uint32_t> shiftedInds(mesh.indices.size());

		for (size_t i = 0; i < mesh.indices.size(); i++) {
			shiftedInds[i] = mesh.indices[i] + baseVertex;
		}

		glBindVertexArray(vao);
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	    glBufferSubData(GL_ARRAY_BUFFER, slot * vboSlotBytes, vBytes, mesh.vertices.data());

	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, slot * eboSlotBytes, eBytes, shiftedInds.data());

	    glBindVertexArray(0);

	    return true;
	};

	std::unordered_set<uint64_t> desired;
	std::vector<uint64_t> newlyAdded;
	std::unordered_set<uint64_t> uploadFailed;

	auto rebuildChunks = [&]() {
		desired.clear();
		newlyAdded.clear();
		uploadFailed.clear();
		desired.reserve((renderRadius * 2 + 1) * (renderRadius * 2 + 1) * chunksPerColumn);

		for (int cy = 0; cy < chunksPerColumn; cy++) {
			for (int cz = playerChunkZ - renderRadius; cz <= playerChunkZ + renderRadius; cz++) {
				for (int cx = playerChunkX - renderRadius; cx <= playerChunkX + renderRadius; cx++) {
					desired.insert(chunkKey(cx, cy, cz));
				}
			}
		}

		for (auto it = chunks.begin(); it != chunks.end(); ) {
			if (desired.count(it->first) == 0) {
				meshCache.erase(it->first);

				freeSlot(it->first);

				it = chunks.erase(it);
			} else {
				++it;
			}
		}

		for (int cz = playerChunkZ - renderRadius; cz <= playerChunkZ + renderRadius; cz++) {
			for (int cx = playerChunkX - renderRadius; cx <= playerChunkX + renderRadius; cx++) {
				bool anythingMissing = false;

				for (int cy = 0; cy < chunksPerColumn; cy++) {
					if (chunks.count(chunkKey(cx, cy, cz)) == 0) {
						anythingMissing = true; 

						break;
					}
				}

				if (!anythingMissing) continue;

				float grid[noiseGridSize][noiseGridSize];

				buildCoarseHeightGrid(cx, cz, grid);

				for (int cy = 0; cy < chunksPerColumn; cy++) {
					uint64_t k = chunkKey(cx, cy, cz);
					if (chunks.count(k) != 0) continue;

					if (cy * chunkSize > worldHeight - chunkSize) {
						Chunk empty{};
						empty.chunkX = cx;
						empty.chunkY = cy;
						empty.chunkZ = cz;
						empty.hasAnySolid = false;

						chunks.emplace(k, empty);
						allocSlot(k);
						newlyAdded.push_back(k);

						continue;
					}

					chunks.emplace(k, makeChunkFromGrid(cx, cy, cz, grid));
					allocSlot(k);
					newlyAdded.push_back(k);
				}
			}
		}

		for (uint64_t k : newlyAdded) {
	        int cx = keyToCx(k);
			int cy = keyToCy(k);
			int cz = keyToCz(k);

	        meshCache.erase(chunkKey(cx, cy, cz));
	        meshCache.erase(chunkKey(cx - 1, cy, cz));
	        meshCache.erase(chunkKey(cx + 1, cy, cz));
	        meshCache.erase(chunkKey(cx, cy, cz - 1));
	        meshCache.erase(chunkKey(cx, cy, cz + 1));
	        meshCache.erase(chunkKey(cx, cy - 1, cz));
	        meshCache.erase(chunkKey(cx, cy + 1, cz));
	    }

	    for (uint64_t k : desired) {
	        if (meshCache.count(k) != 0) continue;

	        int cx = keyToCx(k);
			int cy = keyToCy(k);
			int cz = keyToCz(k);

	        const Chunk& c = chunks.at(k);

	        uint32_t slot = chunkSlots[k];

	        if (!c.hasAnySolid) {
	        	meshCache.try_emplace(k);

	        	continue;
	        }

	        auto [it, insterted] = meshCache.try_emplace(k);

	        auto nb = [&](int ncx, int ncy, int ncz) -> const Chunk* {
	            auto cit = chunks.find(chunkKey(ncx, ncy, ncz));
	            return cit != chunks.end() ? &cit->second : nullptr;
	        };

	        buildChunkMesh(it->second, c, cx, cy, cz, nb(cx - 1, cy, cz), nb(cx + 1, cy, cz), nb(cx, cy - 1, cz), nb(cx, cy + 1, cz), nb(cx, cy, cz - 1), nb(cx, cy, cz + 1));

	        if (!uploadMeshToSlot(slot, it->second)) {
	        	uploadFailed.insert(k);
	        }
	    }

	    commands.clear();
	    commands.reserve(meshCache.size());

		for (const auto& [k, mesh] : meshCache) {
			if (mesh.indices.empty()) continue;
			if (uploadFailed.count(k)) continue;

			uint32_t slot = chunkSlots[k];
			const Chunk& c = chunks.at(k);

			drawElementsCommand cmd{};

			cmd.count = static_cast<GLuint>(mesh.indices.size());
			cmd.instanceCount = 1;
			cmd.firstIndex = slot * eboSlotBytes;
			cmd.baseVertex = 0;
			cmd.baseInstance = 0;
			cmd.aabbMin = glm::vec3(c.chunkX * chunkSize, c.chunkY * chunkSize, c.chunkZ * chunkSize);
			cmd.aabbMax = cmd.aabbMin + glm::vec3(chunkSize, chunkSize, chunkSize);

			commands.push_back(cmd);
		}
	};

	rebuildChunks();

	for (size_t i = 0; i < commands.size() && i < 3; i++) {
    	std::cout << "cmd " << i
    	          << " count="       << commands[i].count
    	          << " firstIndex="  << commands[i].firstIndex
    	          << " baseVertex="  << commands[i].baseVertex
    	          << " baseInstance="<< commands[i].baseInstance
    	          << "\n";
	}

	glUseProgram(shaderProgram);

	glUniform1i(texAtlasLoc, 0);
	glUniform1i(maxHeightTerrainLoc, worldHeight);
	glUniform1i(brickDimXLoc, brickDimX);
	glUniform1i(brickDimYLoc, brickDimY);
	glUniform1i(brickDimZLoc, brickDimZ);

	glUniform1f(ssaoRadiusLoc, setting("ssaoRadius", 2.5f));
	glUniform1f(ssaoBiasLoc, setting("ssaoBias", 0.003f));
	glUniform1f(ssaoIntensityLoc, setting("ssaoIntensity", 1.5f));
	glUniform1f(shadowFactorLoc, setting("shadowFactor", 0.10f));
	glUniform1f(giStrengthLoc, setting("giStrength", 0.24f));

	glUniform3fv(glGetUniformLocation(shaderProgram, "uSsaoKernel[0]"), (GLsizei)ssaoKernel.size(), (const GLfloat*)ssaoKernel.data());

	glUniform1fv(glGetUniformLocation(shaderProgram, "uBlurWeights[0]"), (GLsizei)blurWeights.size(), blurWeights.data());

	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, ssaoNoiseTex);

	glUniform1i(ssaoNoiseLoc, 6);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D_ARRAY, texArray);

    GLuint autoExpLum = 0, autoExpLumFbo = 0;

    glGenTextures(1, &autoExpLum);
    glBindTexture(GL_TEXTURE_2D, autoExpLum);

    glTexStorage2D(GL_TEXTURE_2D, 9, GL_R16F, 256, 256);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &autoExpLumFbo);

    glBindFramebuffer(GL_FRAMEBUFFER, autoExpLumFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, autoExpLum, 0);

    GLuint autoExpTex[2] = { 
    	0, 0 
    };

    GLuint autoExpFbo[2] = {
    	0, 0 
	};

    glGenTextures(2, autoExpTex);
    glGenFramebuffers(2, autoExpFbo);

    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, autoExpTex[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R16F, 1, 1);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, autoExpFbo[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, autoExpTex[i], 0);
        glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    int autoExpCur = 0;

    float exposureSpeed = setting("exposureSpeed", 1.5f);

    glUseProgram(shaderProgram);

    glUniform1f(glGetUniformLocation(shaderProgram, "uExposureKey"), setting("exposureKey", 0.18f));
    glUniform1f(glGetUniformLocation(shaderProgram, "uExposureBias"), setting("exposureBias", 1.0f));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLuint giSsbo = 0;

    glGenBuffers(1, &giSsbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, giSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, giCascades * giVoxelsPerCascade * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, giSsbo);

    std::vector<uint32_t> giCpu;

    glm::ivec3 giOrigin[giCascades] = {
    	glm::ivec3(0),
    	glm::ivec3(0),
    	glm::ivec3(0)
    };

    int giVoxelSizeArr[giCascades] = { 1, 2, 4 };

   	glUniform1i(glGetUniformLocation(shaderProgram, "uGiDebug"), (int)setting("giDebug", 0.0f));

	std::cout << "STARTUP: EOF\n" << std::flush;

	while (!glfwWindowShouldClose(window)) {
		time = glfwGetTime();

		float deltaTime = static_cast<float>(time - lastFrame);

		lastFrame = time;

		if (deltaTime > 0.05f) deltaTime = 0.05f;

		processInput(window, deltaTime, camPos, camFront, up, chunks);

		glm::ivec3 highlightHit, highlightNormal, highlightPlace;

		bool hasHightlight = false;

		static bool lmbWasDown = false;
		static bool rmbWasDown = false;

		bool lmbDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
		bool rmbDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

		if (isCameraRotateable) {

			hasHightlight = raycastVoxel(chunks, camPos, camFront, 6.0f, highlightHit, highlightNormal, highlightPlace);

			if (hasHightlight) {
				if (lmbDown && !lmbWasDown) {
					editVoxel(chunks, meshCache, highlightHit, 0);

					chunksNeedUpdate = true;
				}

				if (rmbDown && !rmbWasDown && !blockOverlapsPlayer(camPos, highlightPlace)) {
					editVoxel(chunks, meshCache, highlightPlace, 1);

					chunksNeedUpdate = true;
				}
			}
		}

		lmbWasDown = lmbDown;
		rmbWasDown = rmbDown;

		timeOfDay += deltaTime / dayLengthSec;
		if (timeOfDay > 1.0f) timeOfDay -= 1.0f;

		float remap = timeOfDay < 0.6f ? timeOfDay * (0.5f / 0.6f) : 0.5f + (timeOfDay - 0.6f) * (0.5f / 0.4f);
		float sunAng = remap * 2.0f * PI;

		glm::vec3 sunDir = glm::normalize(glm::vec3(std::cos(sunAng), std::sin(sunAng), 0.0f));

		glfwGetFramebufferSize(window, &width, &height);

		static int prevWidth = width;
		static int prevHeight = height;
		static int bloomWidth = width;
		static int bloomHeight = height;
		static bool go = false;

		if (!go || width != prevWidth || height != prevHeight) {
			prevWidth = width;
			prevHeight = height;

			go = true;

			pastHeight = std::max(1, (int)(height * renderScale));
			pastWidth = std::max(1, (int)(width * renderScale));

			bloomWidth = pastWidth / 4;
			bloomHeight = pastHeight / 4;

			// Scene
		    glBindTexture(GL_TEXTURE_2D, sceneTex);
		    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, pastWidth, pastHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

		   	glBindTexture(GL_TEXTURE_2D, sceneNormalTex);
		   	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, pastWidth, pastHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

		   	glBindTexture(GL_TEXTURE_2D, sceneDepthTex);
		   	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, pastWidth, pastHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

		    // Accumulation
		    for (int i = 0; i < 2; i++) {
		        glBindTexture(GL_TEXTURE_2D, accumTex[i]);
		        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, pastWidth, pastHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
		    }

		    // Bloom
		    for (int i = 0; i < 2; i++) {
		        glBindTexture(GL_TEXTURE_2D, bloomTex[i]);
		        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, bloomWidth, bloomHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
		    }

		    int aoWidth = std::max(1, pastWidth / 2);
		    int aoHeight = std::max(1, pastHeight / 2);

		    aoW0 = aoWidth;
		    aoH0 = aoHeight;

		    for (GLuint t : { aoTex, aoBlurTex }) {
		    	glBindTexture(GL_TEXTURE_2D, t);
		    	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, aoWidth, aoHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
		    }

		    auto clearFbo = [](GLuint fbo) {
		    	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		    	glClearColor(0, 0, 0, 0);
		    	glClear(GL_COLOR_BUFFER_BIT);
		    };

		    clearFbo(sceneFbo);

		   	for (int i = 0; i <= 1; i++) {
		   		clearFbo(accumFbo[i]);
		   		clearFbo(bloomFbo[i]);
		   	}

		   	clearFbo(aoFbo);
		   	clearFbo(aoBlurFbo);

		   	glBindFramebuffer(GL_FRAMEBUFFER, 0);

			accumFrame = 0;
			accumIdx = 0;
		}

		glBindTexture(GL_TEXTURE_2D, 0);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glm::mat4 projection = glm::perspective(glm::radians(cfov), static_cast<float>(width) / static_cast<float>(height), 0.1f, 800.0f);
		glm::mat4 view = glm::lookAt(camPos, camPos + camFront, up);
		glm::mat4 model(1.0f);
		glm::mat4 mvp = projection * view * model;
		glm::mat4 invVp = glm::inverse(projection * view);

		glUniformMatrix4fv(viewProjLoc, 1, GL_FALSE, glm::value_ptr(projection * view));

		static float smoothedBlend = 1.0f;
		static glm::vec3 lastPos = camPos + glm::vec3(1e6f);
		static glm::vec3 lastFront = camFront;
		static float lastFov = cfov;
		static glm::vec3 lastSunDir = sunDir;

		glm::vec3 dp = camPos - lastPos;
		glm::vec3 df = camFront - lastFront;
		glm::vec3 ds = sunDir - lastSunDir;

		bool moved = glm::dot(dp, dp) > 1e-6f || glm::dot(df, df) > 1e-8f || std::abs(cfov - lastFov) > 1e-4f;

		lastPos = camPos;
		lastFront = camFront;
		lastFov = cfov;
		lastSunDir = sunDir;

		float targetBlend = moved ? 1.0f : 0.12f;
		// float k = (targetBlend > smoothedBlend) ? 0.6f : 0.15f;

		smoothedBlend = moved ? 1.0f : smoothedBlend + (targetBlend - smoothedBlend) * 0.15f;

		accumFrame++;

		float blend = smoothedBlend;

		glUniform3f(sunDirLoc, sunDir.x, sunDir.y, sunDir.z);

		int newChunkX = static_cast<int>(std::floor(camPos.x / static_cast<float>(chunkSize)));
		int newChunkZ = static_cast<int>(std::floor(camPos.z / static_cast<float>(chunkSize)));

		static int lastTboChunkX = INT32_MIN;
		static int lastTboChunkZ = INT32_MIN;

		if (chunksNeedUpdate || newChunkX != lastTboChunkX || newChunkZ != lastTboChunkZ) {
			playerChunkX = newChunkX;
			playerChunkZ = newChunkZ;

		    rebuildChunks();
		    uploadVoxelTbo();
		    uploadBrickTbo();

			lastTboChunkX = newChunkX;
			lastTboChunkZ = newChunkZ;

			chunksNeedUpdate = false;
		}

		static int lightingMode = 0;
		static bool tabWasDown = false;

		bool tabDown = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;

		if (tabDown && !tabWasDown) {
			lightingMode = (lightingMode + 1) % 2;

			accumFrame = 0;
			accumIdx = 0;
			giDirty = true;

			for (int i = 0; i < 2; i++) {
				glBindFramebuffer(GL_FRAMEBUFFER, accumFbo[i]);

				glClearColor(0, 0, 0, 0);
				glClear(GL_COLOR_BUFFER_BIT);
			}

			glBindFramebuffer(GL_FRAMEBUFFER, aoFbo);

			glClearColor(1, 0, 0, 0);
			glClear(GL_COLOR_BUFFER_BIT);

			std::cout << "lighting mode = " << (lightingMode == 0 ? "normal" : "ray marching") << "\n";
		}

		tabWasDown = tabDown;

		glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo);
		glViewport(0, 0, pastWidth, pastHeight);

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		glUseProgram(shaderProgram);
		glUniform1i(uModeLoc, 2);
		glUniform1i(lightingModeLoc, lightingMode);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_BUFFER, voxelTboTex);
		glUniform1i(voxelTboLoc, 1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_BUFFER, brickTboTex);
		glUniform1i(brickTboLoc, 2);

		glUniform1i(rmPlayerXLoc, playerChunkX);
		glUniform1i(rmPlayerZLoc, playerChunkZ);
		glUniform1i(frameIdxLoc, accumFrame);

		glUniform2fv(ssgiOffsetsLoc, (GLsizei)ssgiOffsets.size(), (const GLfloat*)ssgiOffsets.data());

		glm::mat4 skyIdentity(1.0f);

		glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(skyIdentity));

		glUniformMatrix4fv(invViewProjLoc, 1, GL_FALSE, glm::value_ptr(invVp));

		glUniform3f(camPosLoc, camPos.x, camPos.y, camPos.z);

		glUniform1f(timeOfDayLoc, timeOfDay);

		float skyQuad[] = {
			-1, -1, 0, 1, -1, 0, 1, 1, 0,
			-1, -1, 0, 1, 1, 0, -1, 1, 0
		};

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		glBindVertexArray(lineVao);

		glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(skyQuad), nullptr, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(skyQuad), skyQuad);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);

		if (goWireFrame) {
		    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		} else {
		    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}

		frustum Frustum = extractFrustum(projection * view);

		vis.clear();
		vis.reserve(commands.size());

		for (const auto& cmd : commands) {
			if (!aabbInFrustum(Frustum, cmd.aabbMin, cmd.aabbMax)) continue;

			glm::vec3 center = (cmd.aabbMin + cmd.aabbMax) * 0.5f;
			glm::vec3 d = center - camPos;

			vis.push_back({
				&cmd,
				glm::dot(d, d)
			});
		}

		std::sort(vis.begin(), vis.end(), [](const visCmd& a, const visCmd& b){
			return a.distSq < b.distSq;
		});

		mdCounts.clear();
		mdIndices.clear();

		for (const auto& v : vis) {
			mdCounts.push_back(static_cast<GLsizei>(v.cmd->count));
			mdIndices.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(v.cmd->firstIndex)));
		}

		glUseProgram(shaderProgram);

		glUniform1i(uModeLoc, 0);

		glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glBindVertexArray(vao);

		if (!mdCounts.empty()) {
			glMultiDrawElements(GL_TRIANGLES, mdCounts.data(), GL_UNSIGNED_INT, mdIndices.data(), static_cast<GLsizei>(mdCounts.size()));

			GLsync frameFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

			for (const auto& v : vis) {
				uint64_t k = chunkKey(v.cmd->aabbMin.x / chunkSize, v.cmd->aabbMin.y / chunkSize, v.cmd->aabbMin.z / chunkSize);

				auto it = chunkSlots.find(k);
				if (it != chunkSlots.end()) {
					uint32_t slot = it->second;

					if (slotFences[slot] != nullptr) {
						glDeleteSync(slotFences[slot]);
					}

					slotFences[slot] = frameFence;
				}
			}
		}

		glBindVertexArray(0);

		glDisable(GL_BLEND);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		if (hasHightlight) {
			float e = 0.002f;
	        float x0 = (float)highlightHit.x - e;
	        float x1 = (float)highlightHit.x + 1.0f + e;
	        float y0 = (float)highlightHit.y - e;
	        float y1 = (float)highlightHit.y + 1.0f + e;
	        float z0 = (float)highlightHit.z - e;
	        float z1 = (float)highlightHit.z + 1.0f + e;
	        float corners[8][3] = {
	            {x0,y0,z0}, {x1,y0,z0}, {x1,y0,z1}, {x0,y0,z1},
	            {x0,y1,z0}, {x1,y1,z0}, {x1,y1,z1}, {x0,y1,z1}
	        };
 
	        int edges[12][2] = {
	            {0,1},{1,2},{2,3},{3,0},
	            {4,5},{5,6},{6,7},{7,4},
	            {0,4},{1,5},{2,6},{3,7}
	        };
 
	        float verts[72];
 
	        int vi = 0;
 
	        for (int k = 0; k < 12; k++) {
		       	for (int j = 0; j < 2; j++) {
		       		int c = edges[k][j];

		       		verts[vi++] = corners[c][0];
		       		verts[vi++] = corners[c][1];
		       		verts[vi++] = corners[c][2];
		       	}
	        }

	        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

			glUniform4f(uColorLoc, 0.0f, 0.0f, 0.0f, 1.0f);

			glBindVertexArray(lineVao);

			glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
			glBufferData(GL_ARRAY_BUFFER, sizeof(verts), nullptr, GL_STREAM_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);

			glLineWidth(1.0f);
			glDrawArrays(GL_LINES, 0, 24);
		}

		{
			GLuint visible = static_cast<GLuint>(mdCounts.size());
			GLuint total = static_cast<GLuint>(commands.size());
			GLuint triangles = 0;

			for (const auto& [k, mesh] : meshCache) triangles += mesh.indices.size() / 3;

			updFpsCounter(window, time, visible, total, triangles);
		}

		glm::mat4 blendIdentity(1.0f);

		 if (lightingMode == 1) {
			glBindFramebuffer(GL_FRAMEBUFFER, aoFbo);
			glViewport(0, 0, aoW0, aoH0);

			glDisable(GL_DEPTH_TEST);

			glUseProgram(shaderProgram);
			glUniform1i(uModeLoc, 8);

			glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(blendIdentity));

			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_2D, sceneTex);

			glUniform1i(uSceneLoc, 3);

			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_2D, sceneDepthTex);

			glUniform1i(sceneDepthLoc, 4);

			glActiveTexture(GL_TEXTURE5);
			glBindTexture(GL_TEXTURE_2D, sceneNormalTex);

			glUniform1i(sceneNormalLoc, 5);

			glActiveTexture(GL_TEXTURE6);
			glBindTexture(GL_TEXTURE_2D, ssaoNoiseTex);

			glUniform1i(ssaoNoiseLoc, 6);

			float ssaoQuad[] = {
				-1, -1, 0, 1, -1, 0, 1, 1, 0,
				-1, -1, 0, 1, 1, 0, -1, 1, 0
			};

			glBindVertexArray(lineVao);

			glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
			glBufferData(GL_ARRAY_BUFFER, sizeof(ssaoQuad), nullptr, GL_STREAM_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(ssaoQuad), ssaoQuad);

			glDrawArrays(GL_TRIANGLES, 0, 6);

			glEnable(GL_DEPTH_TEST);
		}

		if (lightingMode == 1) {
		    glBindFramebuffer(GL_FRAMEBUFFER, aoBlurFbo);
		    glViewport(0, 0, aoW0, aoH0);
		    glDisable(GL_DEPTH_TEST);

		    glUseProgram(shaderProgram);
		    glUniform1i(uModeLoc, 10);

		    glActiveTexture(GL_TEXTURE3);
		    glBindTexture(GL_TEXTURE_2D, aoTex);
		    glUniform1i(ssaoInputLoc, 3);

		    float fsQuad[] = {
		        -1, -1, 0, 1, -1, 0, 1, 1, 0,
		        -1, -1, 0, 1,  1, 0, -1, 1, 0
		    };

		    glBindVertexArray(lineVao);
		    glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
		    glBufferData(GL_ARRAY_BUFFER, sizeof(fsQuad), nullptr, GL_STREAM_DRAW);
		    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(fsQuad), fsQuad);
		    glDrawArrays(GL_TRIANGLES, 0, 6);

		    glBindVertexArray(0);
		    glEnable(GL_DEPTH_TEST);
		}

		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, aoBlurTex);

		glUniform1i(ssaoResultLoc, 7);

		glBindFramebuffer(GL_FRAMEBUFFER, accumFbo[1 - accumIdx]);
		glViewport(0, 0, pastWidth, pastHeight);

		glUniform1i(uModeLoc, 4);

		glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(blendIdentity));
		glUniform1f(uBlendLoc, blend);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, sceneTex);

		glUniform1i(uSceneLoc, 3);

		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, accumTex[accumIdx]);

		glUniform1i(uAccumLoc, 4);

		float fullscreenQuad[] = {
			-1, -1, 0, 1, -1, 0, 1, 1, 0, 
			-1, -1, 0, 1, 1, 0, -1, 1, 0
		};

		glBindVertexArray(lineVao);
		glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(fullscreenQuad), nullptr, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(fullscreenQuad), fullscreenQuad);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glBindVertexArray(0);

		accumIdx = 1 - accumIdx;

		if (lightingMode == 1) {
            const int px = (int)std::floor(camPos.x);
            const int py = (int)std::floor(camPos.y);
            const int pz = (int)std::floor(camPos.z);

            glm::ivec3 giWant[giCascades];

            bool dirty[giCascades];
            bool anyDirty = false;

            for (int c = 0; c < giCascades; c++) {
            	int vs = giVoxelSizeArr[c];
            	int half = giDim * vs / 2;
            	int maxY = std::max(0, worldHeight - giDim * vs);

            	giWant[c] = glm::ivec3(floorDiv(px - half, vs) * vs, std::clamp(py - half, 0, maxY), floorDiv(pz - half, vs) * vs);

            	dirty[c] = giDirty || std::abs(giWant[c].x - giOrigin[c].x) >= vs * 16 || std::abs(giWant[c].y - giOrigin[c].y) >= vs * 16 || std::abs(giWant[c].z - giOrigin[c].z) >= vs * 16;
            	if (dirty[c]) anyDirty = true;
            }
            
            if (anyDirty) {
            	glBindBuffer(GL_SHADER_STORAGE_BUFFER, giSsbo);

            	for (int c = 0; c < giCascades; c++) {
            		if (!dirty[c]) continue;

            		giOrigin[c] = giWant[c];

            		giFillAndLight(chunks, giOrigin[c], giVoxelSizeArr[c], sunDir, giCpu);

            		glBufferSubData(GL_SHADER_STORAGE_BUFFER, c * giVoxelsPerCascade * sizeof(uint32_t), giCpu.size() * sizeof(uint32_t), giCpu.data());
            	}

            	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            	glUniform3iv(giOriginsLoc, giCascades, (const GLint*)giOrigin);

            	glUniform1fv(giVoxelSizesLoc, giCascades, (const GLfloat*)giVoxelSizeArr);

            	giDirty = false;
            }
        }

		if (lightingMode == 1) {
			GLboolean aeDepthWas = glIsEnabled(GL_DEPTH_TEST);
	        GLboolean aeBlendWas = glIsEnabled(GL_BLEND);

	        GLint aeViewport[4];

	        glGetIntegerv(GL_VIEWPORT, aeViewport);

	        glDisable(GL_DEPTH_TEST);
	        glDisable(GL_BLEND);

	        glUseProgram(shaderProgram);

	        glBindVertexArray(lineVao);
	        glBindBuffer(GL_ARRAY_BUFFER, lineVbo);

	        float aeQuad[] = {
				-1, -1, 0, 1, -1, 0, 1, 1, 0, 
				-1, -1, 0, 1, 1, 0, -1, 1, 0
			};

			glBufferData(GL_ARRAY_BUFFER, sizeof(aeQuad), nullptr, GL_STREAM_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(aeQuad), aeQuad);

			glBindFramebuffer(GL_FRAMEBUFFER, autoExpLumFbo);

			glViewport(0, 0, 256, 256);

			glUniform1i(uModeLoc, 11);

			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_2D, accumTex[accumIdx]);

			glUniform1i(uAccumLoc, 4);

			glDrawArrays(GL_TRIANGLES, 0, 6);

			glActiveTexture(GL_TEXTURE8);
			glBindTexture(GL_TEXTURE_2D, autoExpLum);

			glGenerateMipmap(GL_TEXTURE_2D);

			int autoExpNext = 1 - autoExpCur;

			glBindFramebuffer(GL_FRAMEBUFFER, autoExpFbo[autoExpNext]);

			glViewport(0, 0, 1, 1);

			glUniform1i(uModeLoc, 12);
			glUniform1i(aeLumLoc, 8);

			glActiveTexture(GL_TEXTURE9);
			glBindTexture(GL_TEXTURE_2D, autoExpTex[autoExpCur]);

			glUniform1i(aeExpLoc, 9);

			glUniform1f(aeAdaptLoc, 1.0f - std::exp(-deltaTime * exposureSpeed));

			glDrawArrays(GL_TRIANGLES, 0, 6);

			autoExpCur = autoExpNext;

			if (aeDepthWas) glEnable(GL_DEPTH_TEST);
			if (aeBlendWas) glEnable(GL_BLEND);

			glViewport(aeViewport[0], aeViewport[1], aeViewport[2], aeViewport[3]);

		}

		glBindFramebuffer(GL_FRAMEBUFFER, bloomFbo[0]);
		glViewport(0, 0, bloomWidth, bloomHeight);

		glUseProgram(shaderProgram);

		glUniform1i(uModeLoc, 6);

		glActiveTexture(GL_TEXTURE5);

		glBindTexture(GL_TEXTURE_2D, accumTex[accumIdx]);
		glUniform1i(bloomInputLoc, 5);

		glUniform1f(bloomThresholdLoc, 1.0f);

		glBindVertexArray(lineVao);
		glBindBuffer(GL_ARRAY_BUFFER, lineVbo);

		glBufferData(GL_ARRAY_BUFFER, sizeof(fullscreenQuad), nullptr, GL_STREAM_DRAW);

		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(fullscreenQuad), fullscreenQuad);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		glBindFramebuffer(GL_FRAMEBUFFER, bloomFbo[1]);
		glViewport(0, 0, bloomWidth, bloomHeight);

		glUniform1i(uModeLoc, 7);

		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, bloomTex[0]);
		glUniform1i(bloomInputLoc, 5);

		glUniform2f(bloomTexelSizeLoc, 1.0f / static_cast<float>(bloomWidth), 0.0f);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		glBindFramebuffer(GL_FRAMEBUFFER, bloomFbo[0]);
		glViewport(0, 0, bloomWidth, bloomHeight);

		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, bloomTex[1]);
		glUniform1i(bloomInputLoc, 5);

		glUniform2f(bloomTexelSizeLoc, 0.0f, 1.0f / static_cast<float>(bloomHeight));

		glDrawArrays(GL_TRIANGLES, 0, 6);

		glBindVertexArray(0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(0, 0, width, height);

		glUniform1i(uModeLoc, 0);
		
		glDisable(GL_DEPTH_TEST);

		glUseProgram(shaderProgram);
		glUniform1i(uModeLoc, 5);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, accumTex[accumIdx]);
		glUniform1i(uAccumLoc, 3);

		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, bloomTex[0]);

		glUniform1i(bloomInputLoc, 5);
		glUniform1f(bloomStrengthLoc, setting("bloomStrength", 0.20f));

		glActiveTexture(GL_TEXTURE9);
		glBindTexture(GL_TEXTURE_2D, autoExpTex[autoExpCur]);

		glUniform1i(aeExpLoc, 9);

		glm::mat4 displayIdentity(1.0f);
		glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(displayIdentity));

		float displayQuad[] = {
		    -1, -1, 0, 1, -1, 0, 1, 1, 0,
		    -1, -1, 0, 1, 1, 0, -1, 1, 0
		};

		glBindVertexArray(lineVao);
		glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(displayQuad), nullptr, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(displayQuad), displayQuad);
		
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glUniform1i(uModeLoc, 1);

		glm::mat4 crosshairIdentity(1.0f);
		glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(crosshairIdentity));

		glUniform4f(uColorLoc, 1.0f, 1.0f, 1.0f, 1.0f);

		float s = 0.02f;
		float aspect = static_cast<float>(width) / static_cast<float>(height);
		float ch[12] = {
			-s / aspect, 0.0f, 0.0f,
			s / aspect, 0.0f, 0.0f,
			0.0f, -s, 0.0f,
			0.0f, s, 0.0f
		};

		glBindVertexArray(lineVao);

		glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(ch), nullptr, GL_STREAM_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(ch), ch);

		glLineWidth(2.0f);
		glDrawArrays(GL_LINES, 0, 4);

		glUseProgram(shaderProgram);
		glUniform1i(uModeLoc, 9);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, sceneTex);
		
		glBindVertexArray(0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}