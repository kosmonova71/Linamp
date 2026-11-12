// GLSL helper functions for MilkDrop compatibility
// This file contains all the helper functions used by the GLSL generator

#define MEGABUF_SIZE {{MEGABUF_SIZE}}

float md_rand(vec2 uv) {
    return fract(sin(dot(uv, vec2(12.9898, 78.233)) + u_rand_seed) * 43758.5453);
}

float megabuf_read(float idx) {
    int i = int(mod(idx, float(MEGABUF_SIZE)));
    int pixel_idx = i / 4;
    vec4 pixel = texelFetch(u_megabuf, pixel_idx);
    float channel = mod(float(i), 4.0);
    if (channel < 0.5) return pixel.r;
    if (channel < 1.5) return pixel.g;
    if (channel < 2.5) return pixel.b;
    return pixel.a;
}

float invsqrt(float x) { return inversesqrt(x); }

float fact(float n) {
    float r = 1.0;
    for(int i=2; i<=int(n); i++) r *= float(i);
    return r;
}

float nchoosek(float n, float k) {
    return fact(n) / (fact(k) * fact(n-k));
}

float wrap(float x, float min, float max) {
    return min + mod(x - min, max - min);
}

float cwrap(float x, float min, float max) {
    float range = max - min;
    float m = mod(x - min, 2.0 * range);
    return min + (m > range ? 2.0 * range - m : m);
}

float atan2_safe(float y, float x) { return atan(y, x + 1e-10); }

// --- MODERN GENERATIVE HELPERS ---
vec3 plasma(vec2 p, float time) {
    float v1 = sin(p.x * 10.0 + time);
    float v2 = sin(10.0 * (p.x * sin(time / 2.0) + p.y * cos(time / 3.0)) + time);
    float v3 = sin(sqrt(100.0 * (p.x * p.x + p.y * p.y) + 1.0) + time);
    float v4 = sin(p.x * 5.0 + p.y * 3.0 - time * 1.5) * 0.5;
    float v = v1 + v2 + v3 + v4;
    return vec3(sin(v), sin(v + 2.0), sin(v + 4.0)) * 0.5 + 0.5;
}

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = vec4(x0.xy, x0.xy) + vec4(C.xx, C.zz);
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

vec2 curlNoise(vec2 p) {
    const float e = 0.01;
    float n1 = snoise(vec2(p.x, p.y + e));
    float n2 = snoise(vec2(p.x, p.y - e));
    float n3 = snoise(vec2(p.x + e, p.y));
    float n4 = snoise(vec2(p.x - e, p.y));
    return vec2(n1 - n2, n4 - n3) / (2.0 * e);
}

// SDF Primitives
float sdCircle(vec2 p, float r) { return length(p) - r; }

float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdStar(vec2 p, float r, int n, float m) {
    float an = 3.141593/float(n);
    float en = 3.141593/m;
    vec2  acs = vec2(cos(an),sin(an));
    vec2  ecs = vec2(cos(en),sin(en));
    float bn = mod(atan(p.x,p.y),2.0*an) - an;
    p = length(p)*vec2(cos(bn),abs(sin(bn)));
    p -= r*acs;
    p += ecs*clamp( -dot(p,ecs), 0.0, r*sin(an)/ecs.y);
    return length(p)*sign(p.x);
}

// CLAMP_F helper overloads
// Some generated shader code calls CLAMP_F with a single argument.
// Keep a 1-arg version to avoid arity mismatch on strict GLSL compilers.
#define CLAMP_F(x) clamp(x, -1e6, 1e6)
#define CLAMP_F3(x, lo, hi) clamp(x, lo, hi)

// MilkDrop compatibility aliases (HLSL-style names used in some presets)
#define texsize vec4(1.0/u_resolution.x, 1.0/u_resolution.y, u_resolution.x, u_resolution.y)
#define g_fTexSize texsize
#define sampler_main u_texture
#define rand_preset u_rand_seed
#define rand_frame rand_preset
#define vec2vec2 vec2

// MilkDrop if() function - returns b if cond is non-zero, else c
float if_(float cond, float a, float b) { return (abs(cond) > 1e-5) ? a : b; }

// HLSL-style fallback functions (also handled by HLSL-to-GLSL translator)
float frac(float x) { return fract(x); }
float lerp(float a, float b, float t) { return mix(a, b, t); }
float saturate(float x) { return clamp(x, 0.0, 1.0); }

// MilkDrop 2.x texsize helpers for noise textures (noise texs are 64x64)
#define texsize_noise_lq vec4(1.0/64.0, 1.0/64.0, 64.0, 64.0)
#define texsize_noise_mq vec4(1.0/64.0, 1.0/64.0, 64.0, 64.0)
#define texsize_noise_hq vec4(1.0/64.0, 1.0/64.0, 64.0, 64.0)
#define texsize_noisevol_lq vec4(1.0/32.0, 1.0/32.0, 1.0/32.0, 32.0)
#define texsize_noisevol_hq vec4(1.0/32.0, 1.0/32.0, 1.0/32.0, 32.0)

// Feedback sampler alias
#define sampler_fc_main u_prev_frame
#define sampler_fw_main u_prev_frame
#define sampler_pw_main u_texture
#define sampler_cells u_texture

// GetBlur1_Impl: simple 3-tap blur (internal implementation)
vec3 GetBlur1_Impl(sampler2D samp, vec2 uv, vec2 texelSize) {
    vec3 c = texture(samp, uv).rgb;
    c += texture(samp, uv + vec2(texelSize.x, 0.0)).rgb;
    c += texture(samp, uv - vec2(texelSize.x, 0.0)).rgb;
    c += texture(samp, uv + vec2(0.0, texelSize.y)).rgb;
    c += texture(samp, uv - vec2(0.0, texelSize.y)).rgb;
    return c * 0.2;
}

// GetBlur3_Impl: wider 5-tap blur (internal implementation)
vec3 GetBlur3_Impl(sampler2D samp, vec2 uv, vec2 texelSize) {
    vec3 c = texture(samp, uv).rgb;
    c += texture(samp, uv + vec2(texelSize.x * 2.0, 0.0)).rgb;
    c += texture(samp, uv - vec2(texelSize.x * 2.0, 0.0)).rgb;
    c += texture(samp, uv + vec2(0.0, texelSize.y * 2.0)).rgb;
    c += texture(samp, uv - vec2(0.0, texelSize.y * 2.0)).rgb;
    c += texture(samp, uv + vec2(texelSize.x, texelSize.y)).rgb;
    c += texture(samp, uv - vec2(texelSize.x, texelSize.y)).rgb;
    c += texture(samp, uv + vec2(texelSize.x, -texelSize.y)).rgb;
    c += texture(samp, uv - vec2(texelSize.x, -texelSize.y)).rgb;
    return c * (1.0 / 9.0);
}

// lum: luminance of a vec3 color (MilkDrop spec: dot(a, float3(0.32, 0.49, 0.29)))
float lum(vec3 c) {
    return dot(c, vec3(0.32, 0.49, 0.29));
}

// MilkDrop legacy blur helper wrappers
// These provide the vec3 GetBlur1/2/3(vec2 uv) API expected by presets
vec2 _getTexelSize() {
    return vec2(1.0 / u_resolution.x, 1.0 / u_resolution.y);
}

// GetBlur1 legacy wrapper - returns RGB of simple blur
vec3 GetBlur1(vec2 uv) {
    return GetBlur1_Impl(u_prev_frame, uv, _getTexelSize());
}

// GetBlur2 legacy wrapper - returns RGB of wider blur (using GetBlur3_Impl)
vec3 GetBlur2(vec2 uv) {
    return GetBlur3_Impl(u_prev_frame, uv, _getTexelSize());
}

// GetBlur3 legacy wrapper - returns RGB of widest blur
vec3 GetBlur3(vec2 uv) {
    return GetBlur3_Impl(u_prev_frame, uv, _getTexelSize());
}

// GetPixel: sample the previous frame at given UV coordinates and return RGB
vec3 GetPixel(vec2 uv) {
    return texture(u_prev_frame, uv).rgb;
}

// lum: luminance - accepts float, vec2, and vec3 for preset compatibility
float lum(float c) {
    return c;
}

float lum(vec2 c) {
    return dot(c, vec2(0.299, 0.587));
}

float lum(vec4 c) {
    return lum(c.rgb);
}

