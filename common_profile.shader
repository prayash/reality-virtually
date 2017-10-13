////////////////////////////////////////////////
// CommonProfile Shader v2

#import <metal_stdlib>

using namespace metal;

#ifndef __SCNMetalDefines__
#define __SCNMetalDefines__

enum {
    SCNVertexSemanticPosition,
    SCNVertexSemanticNormal,
    SCNVertexSemanticTangent,
    SCNVertexSemanticColor,
    SCNVertexSemanticBoneIndices,
    SCNVertexSemanticBoneWeights,
    SCNVertexSemanticTexcoord0,
    SCNVertexSemanticTexcoord1,
    SCNVertexSemanticTexcoord2,
    SCNVertexSemanticTexcoord3,
    SCNVertexSemanticTexcoord4,
    SCNVertexSemanticTexcoord5,
    SCNVertexSemanticTexcoord6,
    SCNVertexSemanticTexcoord7
};

// This structure hold all the informations that are constant through a render pass
// In a shader modifier, it is given both in vertex and fragment stage through an argument named "scn_frame".
struct SCNSceneBuffer {
    float4x4    viewTransform;
    float4x4    inverseViewTransform; // transform from view space to world space
    float4x4    projectionTransform;
    float4x4    viewProjectionTransform;
    float4x4    viewToCubeTransform; // transform from view space to cube texture space (canonical Y Up space)
    float4x4    lastFrameViewProjectionTransform;
    float4      ambientLightingColor;
    float4		fogColor;
    float3		fogParameters; // x:-1/(end-start) y:1-start*x z:exp
    float2      inverseResolution;
    float       time;
    float       sinTime;
    float       cosTime;
    float       random01;
    float       motionBlurIntensity;
    // new in macOS 10.12 and iOS 10
    float       environmentIntensity;
    float4x4    inverseProjectionTransform;
    float4x4    inverseViewProjectionTransform;
    // new in macOS 10.13 and iOS 11
    float2      nearFar; // x: near, y: far
    float4      viewportSize; // xy:size, zw:size / tan(fov/2)
};

// In custom shaders or in shader modifiers, you also have access to node relative information.
// This is done using an argument named "scn_node", which must be a struct with only the necessary fields
// among the following list:
//
// float4x4 modelTransform;
// float4x4 inverseModelTransform;
// float4x4 modelViewTransform;
// float4x4 inverseModelViewTransform;
// float4x4 normalTransform; // This is the inverseTransposeModelViewTransform, need for normal transformation
// float4x4 modelViewProjectionTransform;
// float4x4 inverseModelViewProjectionTransform;
// float2x3 boundingBox;
// float2x3 worldBoundingBox;

#endif /* defined(__SCNMetalDefines__) */


//
// Utility
//

// Tool function

namespace scn {
    
    // MARK: - Matrix/Vector utils
    
    inline float3x3 mat3(float4x4 mat4)
    {
        return float3x3(mat4[0].xyz, mat4[1].xyz, mat4[2].xyz);
    }
    
    inline float3 mat4_mult_float3_normalized(float4x4 matrix, float3 src)
    {
        float3 dst  =  src.xxx * matrix[0].xyz;
        dst         += src.yyy * matrix[1].xyz;
        dst         += src.zzz * matrix[2].xyz;
        return normalize(dst);
    }
    
    inline float3 mat4_mult_float3(float4x4 matrix, float3 src)
    {
        float3 dst  =  src.xxx * matrix[0].xyz;
        dst         += src.yyy * matrix[1].xyz;
        dst         += src.zzz * matrix[2].xyz;
        return dst;
    }

    inline float3 matrix_rotate(float4x4 mat, float3 dir)
    {
        return  dir.xxx * mat[0].xyz +
                dir.yyy * mat[1].xyz +
                dir.zzz * mat[2].xyz;
    }

    inline float4 matrix_transform(float4x4 mat, float3 pos)
    {
        return  pos.xxxx * mat[0] +
                pos.yyyy * mat[1] +
                pos.zzzz * mat[2] +
                           mat[3];
    }

    inline void generate_basis(float3 inR, thread float3 *outS, thread float3 *outT)
    {
        //float3 dir = abs(inR.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
        float3 dir = mix( float3(1.,0.,0.), float3(0.,0.,1.), step(0.999, abs(inR.z)) );
        *outS = normalize(cross(dir, inR));
        *outT = cross(inR, *outS);
    }
    
    // MARK: - Blending operators
    
    inline float3 blend_add(float3 base, float3 blend)
    {
        return min(base + blend, 1.0);
    }
    
    inline float3 blend_lighten(float3 base, float3 blend)
    {
        return max(blend, base);
    }
    
    inline float3 blend_screen(float3 base, float3 blend)
    {
        return (1.0 - ((1.0 - base) * (1.0 - blend)));
    }

    // MARK: - Math
    
    inline half sq(half f) {
        return f * f;
    }

    inline float sq(float f) {
        return f * f;
    }
    
    // MARK: - SIMD Extensions
    
    inline vector_float2 barycentric_mix(vector_float2 __x, vector_float2 __y, vector_float2 __z, vector_float3 __t) { return __t.x * __x + __t.y * __y + __t.z * __z; }
    inline vector_float3 barycentric_mix(vector_float3 __x, vector_float3 __y, vector_float3 __z, vector_float3 __t) { return __t.x * __x + __t.y * __y + __t.z * __z; }
    inline vector_float4 barycentric_mix(vector_float4 __x, vector_float4 __y, vector_float4 __z, vector_float3 __t) { return __t.x * __x + __t.y * __y + __t.z * __z; }
    
    static inline float rect(float2 lt, float2 rb, float2 uv)
    {
        float2 borders = step(lt, uv) * step(uv, rb);
        return borders.x * borders.y;
    }
    
    inline half4 debugColorForCascade(int cascade)
    {
        switch (cascade) {
            case 0:
            return half4(1.h, 0.h, 0.h, 1.h);
            case 1:
            return half4(0.9, 0.5, 0., 1.);
            case 2:
            return half4(1., 1., 0., 1.);
            case 3:
            return half4(0., 1., 0., 1.);
            default:
            return half4(0., 0., 0., 1.);
        }
    }
    
    inline float grid(float2 lt, float2 rb, float2 gridSize, float thickness, float2 uv)
    {
        float insideRect = rect(lt, rb + thickness, uv);
        float2 gt = thickness * gridSize;
        float2 lines = step(abs(lt - fract(uv * gridSize)), gt);
        return insideRect * (lines.x + lines.y);
    }
    
    // MARK: - Colors
    
    inline float luminance(float3 color)
    {
        // `color` assumed to be in the linear sRGB color space
        // https://en.wikipedia.org/wiki/Relative_luminance
        return color.r * 0.212671 + color.g * 0.715160 + color.b * 0.072169;
    }
    
    inline float srgb_to_linear(float c)
    {
        return (c <= 0.04045f) ? c / 12.92f : powr((c + 0.055f) / 1.055f, 2.4f);
    }
    
    inline half srgb_to_linear_fast(half c)
    {
        return powr(c, 2.2h);
    }
    
    inline half3 srgb_to_linear_fast(half3 c)
    {
        return powr(c, 2.2h);
    }
    
    inline half srgb_to_linear(half c)
    {
        // return (c <= 0.04045h) ? c / 12.92h : powr((c + 0.055h) / 1.055h, 2.4h);
        return (c <= 0.04045h) ? (c * 0.0773993808h) :  powr(0.9478672986h * c + 0.05213270142h, 2.4h);
    }
    
    inline float3 srgb_to_linear(float3 c)
    {
        return float3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
    }
    
    inline float linear_to_srgb(float c)
    {
        return (c < 0.0031308f) ? (12.92f * c) : (1.055f * powr(c, 1.f/2.4f) - 0.055f);
    }
    
    inline float3 linear_to_srgb(float3 v) { // we do not saturate since linear extended values can be fed in
        return float3(linear_to_srgb(v.x), linear_to_srgb(v.y), linear_to_srgb(v.z));
    }
    
}

// MARK: GL helpers

template <typename T>
inline T dFdx(T v) {
    return dfdx(v);
}

// Y is up in GL and down in Metal
template <typename T>
inline T dFdy(T v) {
    return -dfdy(v);
}

// MARK: -

inline float4 texture2DProj(texture2d<float> tex, sampler smp, float4 uv)
{
    return tex.sample(smp, uv.xy / uv.w);
}

constexpr sampler shadow_sampler(coord::normalized, filter::linear, mip_filter::none, address::clamp_to_edge, compare_func::greater_equal);

inline float shadow2DProj(depth2d<float> tex, float4 uv)
{
    float3 uvp = uv.xyz / uv.w;
    return tex.sample_compare(shadow_sampler, uvp.xy, uvp.z);
}

inline float shadow2DProj(sampler smp, depth2d<float> tex, float4 uv)
{
    float3 uvp = uv.xyz / uv.w;
    return tex.sample_compare(smp, uvp.xy, uvp.z);
}

inline float shadow2DArrayProj(depth2d_array<float> tex, float4 uv, uint slice)
{
    float3 uvp = uv.xyz / uv.w;
    return tex.sample_compare(shadow_sampler, uvp.xy, slice, uvp.z);
}

// MARK Shadow

inline float ComputeShadow(float3 worldPos, float4x4 shadowMatrix, depth2d<float> shadowMap)
{
    //project into light space
    float4 lightScreen =  shadowMatrix * float4(worldPos, 1.0);
    
    // ensure receiver after the shadow projection box are not in shadow (when no caster == 1. instead of infinite)
    lightScreen.z = min(lightScreen.z, 0.9999f * lightScreen.w);
    
    float shadow = shadow2DProj(shadow_sampler, shadowMap, lightScreen);

    // Is this useful ?
    shadow *= step(0., lightScreen.w);
    
    return shadow;
}

inline float ComputeSoftShadow(float3 worldPos, float4x4 shadowMatrix, depth2d<float> shadowMap, constant float4* shadowKernel, int sampleCount, float shadowRadius)
{
    //project into light space
    float4 lightScreen =  shadowMatrix * float4(worldPos, 1.0);
    
    // ensure receiver after the shadow projection box are not in shadow (when no caster == 1. instead of infinite)
    lightScreen.z = min(lightScreen.z, 0.9999f * lightScreen.w);
    
    // penumbra
    float filteringSizeFactor = shadowRadius * lightScreen.w;
        
    //smooth all samples
    float totalAccum = 0.0;
    for(int i=0; i < sampleCount; i++){
        totalAccum += shadow2DProj(shadowMap, lightScreen + (shadowKernel[i] * filteringSizeFactor));
    }
    float shadow = totalAccum / float(sampleCount);
    
    // Is this useful ?
    shadow *= step(0., lightScreen.w);

    return shadow;
}

inline float ComputeBlendedCascadedShadow(float3 worldPos, constant float4x4* shadowMatrices, int cascadeCount, depth2d_array<float> shadowMaps, float cascadeBlendingFactor)
{
    float shadow = 0.f;
    float opacitySum = 0.f;

    for (int c = 0; c < cascadeCount; ++c) {
        
        float4 lightScreen =  shadowMatrices[c] * float4(worldPos, 1.0);
        
        // ensure receiver after the shadow projection box are not in shadow (when no caster == 1. instead of infinite)
        lightScreen.z = min(lightScreen.z, 0.9999f * lightScreen.w);
        
        // move in [-1..1] range
        float2 o = lightScreen.xy * 2.f - 1.f;
        // float2 o = (lightScreen.xy / lightScreen.w) * 2.f - 1.f; // do we need to test after projection ???
        const float edge = 1.f - cascadeBlendingFactor;
        // could also do a smoothstep
        o = 1.f - saturate((abs(o) - edge) / cascadeBlendingFactor);
        float opacity = o.x * o.y; //min(o.x, o.y);
        
        if (opacity > 0.f) { // this cascade should be considered
            
            float alpha = opacity * (1.f - opacitySum);
            
            shadow += shadow2DArrayProj(shadowMaps, lightScreen, c) * alpha;
            opacitySum += alpha;

        }
        
        if (opacitySum >= 1.f) // fully opaque shadow (no more blending needed) -> bail out
            break;
    }
    
    if (opacitySum)
        shadow /= opacitySum; // normalization
    
    return shadow;
}

inline float ComputeCascadedShadow(float3 worldPos, constant float4x4* shadowMatrices, int cascadeCount, depth2d_array<float> shadowMaps)
{
    for (int c = 0; c < cascadeCount; ++c) {
        
        float4 lightScreen =  shadowMatrices[c] * float4(worldPos, 1.0);
        
        // ensure receiver after the shadow projection box are not in shadow (when no caster == 1. instead of infinite)
        lightScreen.z = min(lightScreen.z, 0.9999f * lightScreen.w);
        
        // move in [-1..1] range
        float2 o = lightScreen.xy * 2.f - 1.f;
        // float2 o = (lightScreen.xy / lightScreen.w) * 2.f - 1.f; // do we need to test after projection ???
        float opacity = step(abs(o.x), 1.f) * step(abs(o.y), 1.f);
        
        if (opacity > 0.f) { // this cascade should be considered
            
            return shadow2DArrayProj(shadowMaps, lightScreen, c);
        }
    }
    
    return 0.f;
}

inline float ComputeCascadedSoftShadow(float3 worldPos, constant float4x4* shadowMatrices, int cascadeCount, depth2d_array<float> shadowMaps, bool enableCascadeBlending, float cascadeBlendingFactor, constant float4* shadowKernel, int sampleCount, float shadowRadius)
{
    float shadow = 0.f;
    float opacitySum = 0.f;
    
    for (int c = 0; c < cascadeCount; ++c) {
        
        float4 lightScreen =  shadowMatrices[c] * float4(worldPos, 1.0);
        
        // ensure receiver after the shadow projection box are not in shadow (when no caster == 1. instead of infinite)
        lightScreen.z = min(lightScreen.z, 0.9999f * lightScreen.w);
        
        // move in [-1..1] range
        float2 o = lightScreen.xy * 2.f - 1.f;
        // float2 o = (lightScreen.xy / lightScreen.w) * 2.f - 1.f; // do we need to test after projection ???
        float opacity = 1.f;
        if (enableCascadeBlending) {
            const float edge = 1.f - cascadeBlendingFactor;
            // could also do a smoothstep
            o = 1.f - saturate((abs(o) - edge) / cascadeBlendingFactor);
            opacity = o.x * o.y; //min(o.x, o.y);
        } else {
            opacity = step(abs(o.x), 1.f) * step(abs(o.y), 1.f);
        }
        
        if (opacity > 0.f) { // this cascade should be considered
            
            float alpha = opacity * (1.f - opacitySum);
            
            // penumbra
            float filteringSizeFactor = shadowRadius * lightScreen.w; //shadowRadius * lightScreen.w;//(distLight - lightDepth)*shadowRadius / lightDepth ;
            
            //smooth all samples
            float totalAccum = 0.0;
            for (int i=0; i < sampleCount; i++) {
                totalAccum += shadow2DArrayProj(shadowMaps, lightScreen + (shadowKernel[i] * filteringSizeFactor), c);
            }
            
            //    float shadow = totalAccum;
            shadow += (totalAccum / float(sampleCount)) * alpha;
            
            // shadow += shadow2DArrayProj(shadowMaps, lightScreen, c) * alpha;
            opacitySum += alpha;
            
        }
        
        if (opacitySum >= 1.f) // fully opaque shadow (no more blending needed) -> bail out
            break;
    }
    
    if (opacitySum)
    shadow /= opacitySum; // normalization
    
    return shadow;
}

inline float4 ComputeCascadedShadowDebug(float3 worldPos, constant float4x4* shadowMatrices, int cascadeCount, depth2d_array<float> shadowMaps, bool enableCascadeBlending, float cascadeBlendingFactor)
{
    float shadow = 0.f;
    float opacitySum = 0.f;
    half4 debugColorSum = 0.f;
    
    for (int c = 0; c < cascadeCount; ++c) {
        
        float4 lightScreen =  shadowMatrices[c] * float4(worldPos, 1.0);
        
        // ensure receiver after the shadow projection box are not in shadow (when no caster == 1. instead of infinite)
        lightScreen.z = min(lightScreen.z, 0.9999f * lightScreen.w);
        
        // move in [-1..1] range
        float2 o = lightScreen.xy * 2.f - 1.f;
        // float2 o = (lightScreen.xy / lightScreen.w) * 2.f - 1.f; // do we need to test after projection ???
        float opacity = 1.f;
        if (enableCascadeBlending) {
            const float edge = 1.f - cascadeBlendingFactor;
            // could also do a smoothstep
            o = 1.f - saturate((abs(o) - edge) / cascadeBlendingFactor);
            opacity = o.x * o.y; //min(o.x, o.y);
        } else {
            opacity = step(abs(o.x), 1.f) * step(abs(o.y), 1.f);
        }
        
        if (opacity > 0.f) { // this cascade should be considered
            
            float alpha = opacity * (1.f - opacitySum);
            
            { // cascade debug + grid
                float2 texPos = lightScreen.xy / lightScreen.w;
                float2 gridSize = float2(shadowMaps.get_width(), shadowMaps.get_height()) / 8;
                float gd = scn::grid(float2(0.f), float2(1.f), gridSize, 0.001f, texPos);
                half4 gridCol = mix(scn::debugColorForCascade(c), half4(0.f), half4(gd > 0.f));
                debugColorSum += gridCol * alpha;
            }
            
            shadow += shadow2DArrayProj(shadowMaps, lightScreen, c) * alpha;
            opacitySum += alpha;
            
        }
        
        if (opacitySum >= 1.f) // fully opaque shadow (no more blending needed) -> bail out
            break;
    }
    
    if (opacitySum)
        shadow /= opacitySum; // normalization
    
    return float4(float3(debugColorSum.rgb), shadow);
}





// Inputs

typedef struct {

#ifdef USE_MODELTRANSFORM
    float4x4 modelTransform;
#endif
#ifdef USE_INVERSEMODELTRANSFORM
    float4x4 inverseModelTransform;
#endif
#ifdef USE_MODELVIEWTRANSFORM
    float4x4 modelViewTransform;
#endif
#ifdef USE_INVERSEMODELVIEWTRANSFORM
    float4x4 inverseModelViewTransform;
#endif
#ifdef USE_NORMALTRANSFORM
    float4x4 normalTransform;
#endif
#ifdef USE_MODELVIEWPROJECTIONTRANSFORM
    float4x4 modelViewProjectionTransform;
#endif
#ifdef USE_INVERSEMODELVIEWPROJECTIONTRANSFORM
    float4x4 inverseModelViewProjectionTransform;
#endif
#ifdef USE_MOTIONBLUR
    float4x4 lastFrameModelViewProjectionTransform;
    float motionBlurIntensity;
#endif
#ifdef USE_BOUNDINGBOX
    float2x3 boundingBox;
#endif
#ifdef USE_WORLDBOUNDINGBOX
    float2x3 worldBoundingBox;
#endif
#ifdef USE_NODE_OPACITY
    float nodeOpacity;
#endif
#ifdef USE_DOUBLE_SIDED
    float orientationPreserved;
#endif
#if defined(USE_PROBES_LIGHTING) && (USE_PROBES_LIGHTING == 2)
    sh2_coefficients shCoefficients;
#elif defined(USE_PROBES_LIGHTING) && (USE_PROBES_LIGHTING == 3)
    sh3_coefficients shCoefficients;
#endif
#ifdef USE_SKINNING // need to be last since we may cut the buffer size based on the real bone number
    float4 skinningJointMatrices[765]; // Consider having a separate buffer ?
#endif
} commonprofile_node;

typedef struct {
    float3 position         [[attribute(SCNVertexSemanticPosition)]];
#ifdef HAS_NORMAL
    float3 normal           [[attribute(SCNVertexSemanticNormal)]];
#endif
#ifdef USE_TANGENT
    float4 tangent          [[attribute(SCNVertexSemanticTangent)]];
#endif
#ifdef USE_VERTEX_COLOR
    float4 color            [[attribute(SCNVertexSemanticColor)]];
#endif
#ifdef USE_SKINNING
    float4 skinningWeights  [[attribute(SCNVertexSemanticBoneWeights)]];
    uint4  skinningJoints   [[attribute(SCNVertexSemanticBoneIndices)]];
#endif
#ifdef NEED_IN_TEXCOORD0
    float2 texcoord0        [[attribute(SCNVertexSemanticTexcoord0)]];
#endif
#ifdef NEED_IN_TEXCOORD1
    float2 texcoord1        [[attribute(SCNVertexSemanticTexcoord1)]];
#endif
#ifdef NEED_IN_TEXCOORD2
    float2 texcoord2        [[attribute(SCNVertexSemanticTexcoord2)]];
#endif
#ifdef NEED_IN_TEXCOORD3
    float2 texcoord3        [[attribute(SCNVertexSemanticTexcoord3)]];
#endif
#ifdef NEED_IN_TEXCOORD4
    float2 texcoord4        [[attribute(SCNVertexSemanticTexcoord4)]];
#endif
#ifdef NEED_IN_TEXCOORD5
    float2 texcoord5        [[attribute(SCNVertexSemanticTexcoord5)]];
#endif
#ifdef NEED_IN_TEXCOORD6
    float2 texcoord6        [[attribute(SCNVertexSemanticTexcoord6)]];
#endif
#ifdef NEED_IN_TEXCOORD7
    float2 texcoord7        [[attribute(SCNVertexSemanticTexcoord7)]];
#endif
} scn_vertex_t; // __attribute__((scn_per_frame));

typedef struct {
    float4 fragmentPosition [[position]]; // The window relative coordinate (x, y, z, 1/w) values for the fragment
#ifdef USE_POINT_RENDERING
    float fragmentSize [[point_size]];
#endif
#ifdef USE_VERTEX_COLOR
    float4 vertexColor;
#endif
#ifdef USE_PER_VERTEX_LIGHTING
    float3 diffuse;
#ifdef USE_SPECULAR
    float3 specular;
#endif
#endif
#if defined(USE_POSITION) && (USE_POSITION == 2)
    float3 position;
#endif
#if defined(USE_NORMAL) && (USE_NORMAL == 2) && (defined(HAS_NORMAL) || defined(USE_OPENSUBDIV))
    float3 normal;
#endif
#if defined(USE_TANGENT) && (USE_TANGENT == 2)
    float3 tangent;
#endif
#if defined(USE_BITANGENT) && (USE_BITANGENT == 2)
    float3 bitangent;
#endif
#ifdef USE_DISPLACEMENT_MAP
    float2 displacementTexcoord;   // Displacement texture coordinates
#endif
#ifdef USE_NODE_OPACITY
    float nodeOpacity;
#endif
#ifdef USE_DOUBLE_SIDED
    float orientationPreserved;
#endif
#ifdef USE_TEXCOORD
    
#endif
    
#ifdef USE_EXTRA_VARYINGS
    
#endif
    
#ifdef USE_MOTIONBLUR
    float3 velocity;// [[ center_no_perspective ]];
#endif
#ifdef USE_OUTLINE
	float outlineHash [[ flat ]];
#endif
} commonprofile_io;

struct SCNShaderSurface {
    float3 view;                // Direction from the point on the surface toward the camera (V)
    float3 position;            // Position of the fragment
    float3 normal;              // Normal of the fragment (N)
    float3 geometryNormal;      // Normal of the fragment - not taking into account normal map
    float2 normalTexcoord;      // Normal texture coordinates
    float3 tangent;             // Tangent of the fragment
    float3 bitangent;           // Bitangent of the fragment
    float4 ambient;             // Ambient property of the fragment
    float2 ambientTexcoord;     // Ambient texture coordinates
    float4 diffuse;             // Diffuse property of the fragment. Alpha contains the opacity.
    float2 diffuseTexcoord;     // Diffuse texture coordinates
    float4 specular;            // Specular property of the fragment
    float2 specularTexcoord;    // Specular texture coordinates
    float4 emission;            // Emission property of the fragment
    float2 emissionTexcoord;    // Emission texture coordinates
    float4 selfIllumination;            // selfIllumination property of the fragment
    float2 selfIlluminationTexcoord;    // selfIllumination texture coordinates
    float4 multiply;            // Multiply property of the fragment
    float2 multiplyTexcoord;    // Multiply texture coordinates
    float4 transparent;         // Transparent property of the fragment
    float2 transparentTexcoord; // Transparent texture coordinates
    float4 reflective;          // Reflective property of the fragment
    float  metalness;           // Metalness
    float2 metalnessTexcoord;   // Metalness texture coordinates
    float  roughness;           // Roughness
    float2 roughnessTexcoord;   // Roughness texture coordinates
    float shininess;            // Shininess property of the fragment.
    float fresnel;              // Fresnel property of the fragment.
    float ambientOcclusion;     // Ambient occlusion term of the fragment
    float3 _normalTS;           // UNDOCUMENTED in tangent space
#ifdef USE_SURFACE_EXTRA_DECL
    
#endif
};

struct SCNShaderLightingContribution {
    float3 ambient;
    float3 diffuse;
    float3 specular;
    float3 modulate;
};

// Structure to gather property of a light, packed to give access in a light shader modifier
struct SCNShaderLight {
    float4 intensity; // lowp, light intensity
    float3 direction; // mediump, vector from the point toward the light
    float  _att;
    float3 _spotDirection; // lowp, vector from the point to the light for point and spot, dist attenuations
    float  _distance; // mediump, distance from the point to the light (same coord. than range)
};

#ifdef USE_PBR

inline SCNPBRSurface SCNShaderSurfaceToSCNPBRSurface(SCNShaderSurface surface)
{
    SCNPBRSurface s;
    
    s.n = surface.normal;
    s.v = surface.view;
    s.albedo = surface.diffuse.xyz;
    
#ifdef USE_EMISSION
    s.emission = surface.emission.xyz;
#else
    s.emission = float3(0.);
#endif
#ifdef USE_SELFILLUMINATION
    s.selfIllumination = surface.selfIllumination.xyz;
#else
    s.selfIllumination = float3(0.);
#endif
    
    s.metalness = surface.metalness;
    s.roughness = surface.roughness;
    s.ao = surface.ambientOcclusion;
    return s;
}

static float4 scn_pbr_combine(SCNPBRSurface                      pbr_surface,
                              SCNShaderLightingContribution      lighting,
                              texture2d<float, access::sample>   specularDFG,
                              texturecube<float, access::sample> specularLD,
#ifdef USE_PROBES_LIGHTING
#if defined(USE_PROBES_LIGHTING) && (USE_PROBES_LIGHTING == 2)
                              sh2_coefficients                   shCoefficients,
#elif defined(USE_PROBES_LIGHTING) && (USE_PROBES_LIGHTING == 3)
                              sh3_coefficients                   shCoefficients,
#endif
#else
                              texturecube<float, access::sample> irradiance,
#endif
                              constant SCNSceneBuffer&           scn_frame)
{
#ifdef USE_PROBES_LIGHTING
    float3 pbr_color = scn_pbr_color_IBL(pbr_surface, specularDFG, specularLD, shCoefficients, scn_frame.viewToCubeTransform, scn_frame.environmentIntensity);
#else
    float3 pbr_color = scn_pbr_color_IBL(pbr_surface, specularDFG, specularLD, irradiance, scn_frame.viewToCubeTransform, scn_frame.environmentIntensity);
#endif
    
    float4 color;
    color.rgb = (lighting.ambient * pbr_surface.ao + lighting.diffuse) * pbr_surface.albedo.rgb + lighting.specular + pbr_color;
    
#ifdef USE_EMISSION
    color.rgb += pbr_surface.emission.rgb;
#endif
    
    return color;
}

static void scn_pbr_lightingContribution(SCNShaderSurface                   surface,
                                         SCNShaderLight                     light,
                                         constant SCNSceneBuffer&           scn_frame,
                                         thread float3&                     lightingContributionDiffuse,
                                         thread float3&                     lightingContributionSpecular)
{
    SCNPBRSurface pbr_surface = SCNShaderSurfaceToSCNPBRSurface(surface);
    
    float3 diffuseOut, specularOut;
    scn_pbr_lightingContribution_pointLight(light.direction, pbr_surface.n, pbr_surface.v, pbr_surface.albedo, pbr_surface.metalness, pbr_surface.roughness, diffuseOut, specularOut);
    
    float3 lightFactor = light.intensity.rgb * light._att;
    lightingContributionDiffuse += diffuseOut * lightFactor;
    lightingContributionSpecular += specularOut * lightFactor;
}

#else // ifdef USE_PBR

inline float4 illuminate(SCNShaderSurface surface, SCNShaderLightingContribution lighting)
{
    float4 color = {0.,0.,0., surface.diffuse.a};
    
    float3 D = lighting.diffuse;
  
#if defined(USE_AMBIENT_LIGHTING) && (defined(LOCK_AMBIENT_WITH_DIFFUSE) || defined(USE_AMBIENT_AS_AMBIENTOCCLUSION))
    D += lighting.ambient * surface.ambientOcclusion;
#endif
    
#ifdef USE_SELFILLUMINATION
    D += surface.selfIllumination.rgb;
#endif

    // Do we want to clamp there ????

    color.rgb = surface.diffuse.rgb * D;
    #ifdef USE_SPECULAR
        float3 S = lighting.specular;
    #elif defined(USE_REFLECTIVE)
        float3 S = float3(0.);
    #endif
    #ifdef USE_REFLECTIVE
        S += surface.reflective.rgb * surface.ambientOcclusion;
    #endif
    #ifdef USE_SPECULAR
        S *= surface.specular.rgb;
    #endif
    #if defined(USE_SPECULAR) || defined(USE_REFLECTIVE)
        color.rgb += S;
    #endif
#if defined(USE_AMBIENT) && !defined(USE_AMBIENT_AS_AMBIENTOCCLUSION)
    color.rgb += surface.ambient.rgb * lighting.ambient;
#endif
#ifdef USE_EMISSION
    color.rgb += surface.emission.rgb;
#endif
#ifdef USE_MULTIPLY
    color.rgb *= surface.multiply.rgb;
#endif
#ifdef USE_MODULATE
    color.rgb *= lighting.modulate;
#endif
    return color;
}
#endif

struct  commonprofile_lights {
#ifdef USE_LIGHTING
        float4 color0;
    float4 direction0;

#endif
};


struct SCNShaderGeometry
{
    float4 position;
    float3 normal;
    float4 tangent;
    float4 color;
    float pointSize;
    float2 texcoords[8]; // MAX_UV
};

struct commonprofile_uniforms {
    float4 diffuseColor;
    float4 specularColor;
    float4 ambientColor;
    float4 emissionColor;
    float4 selfIlluminationColor;
    float4 reflectiveColor;
    float4 multiplyColor;
    float4 transparentColor;
    float metalness;
    float roughness;
    
    float diffuseIntensity;
    float specularIntensity;
    float normalIntensity;
    float ambientIntensity;
    float emissionIntensity;
    float selfIlluminationIntensity;
    float reflectiveIntensity;
    float multiplyIntensity;
    float transparentIntensity;
    float metalnessIntensity;
    float roughnessIntensity;
    float displacementIntensity;
    
    float materialShininess;
    float selfIlluminationOcclusion;
    float transparency;
    float3 fresnel; // x: ((n1-n2)/(n1+n2))^2 y:1-x z:exponent

#ifdef TEXTURE_TRANSFORM_COUNT
    float4x4 textureTransforms[TEXTURE_TRANSFORM_COUNT];
#endif

#if defined(USE_REFLECTIVE_CUBEMAP)
//    float4x4 u_viewToCubeWorld;
#endif
};

// Shader modifiers declaration (only enabled if one modifier is present)
#ifdef USE_SHADER_MODIFIERS

#endif

#ifdef USE_OPENSUBDIV



struct osd_packed_vertex {
    packed_float3 position;
#if defined(OSD_USER_VARYING_DECLARE_PACKED)
    OSD_USER_VARYING_DECLARE_PACKED
#endif
};

#endif


#ifdef USE_DISPLACEMENT_MAP
static void applyDisplacement( texture2d<float> displacementTexture, sampler displacementTextureSampler, float2 displacementTexcoord, thread SCNShaderGeometry &geometry, constant commonprofile_uniforms& scn_commonprofile )
{
#ifdef USE_DISPLACEMENT_TEXTURE_COMPONENT
	float altitude = displacementTexture.sample(displacementTextureSampler, displacementTexcoord)[USE_DISPLACEMENT_TEXTURE_COMPONENT];
#ifdef USE_DISPLACEMENT_INTENSITY
	altitude *= scn_commonprofile.displacementIntensity;
#endif
#if defined(USE_NORMAL) && (defined(HAS_NORMAL) || defined(USE_OPENSUBDIV))
	float3 bitangent = geometry.tangent.w * normalize(cross(geometry.tangent.xyz, geometry.normal.xyz));
	geometry.position.xyz += geometry.normal * altitude;
	
	float3 offset = float3(1./displacementTexture.get_width(), 1./displacementTexture.get_height(), 0.);
	float3 h;
	h.x = displacementTexture.sample(displacementTextureSampler, displacementTexcoord)[USE_DISPLACEMENT_TEXTURE_COMPONENT];
	h.y = displacementTexture.sample(displacementTextureSampler, displacementTexcoord+offset.xz)[USE_DISPLACEMENT_TEXTURE_COMPONENT];
	h.z = displacementTexture.sample(displacementTextureSampler, displacementTexcoord-offset.zy)[USE_DISPLACEMENT_TEXTURE_COMPONENT];
	
#ifdef USE_DISPLACEMENT_INTENSITY
	h *= scn_commonprofile.displacementIntensity;
#endif
	
	float3 n = normalize( float3( (h.x - h.y)/offset.x, 1., (h.x - h.z)/offset.y) );
	geometry.normal = geometry.tangent.xyz * n.x + geometry.normal.xyz * n.y + bitangent.xyz * n.z;
	geometry.tangent.xyz = normalize(cross(bitangent, geometry.normal));
#endif // USE_NORMAL
#else // USE_DISPLACEMENT_TEXTURE_COMPONENT
	float3 displacement = displacementTexture.sample(displacementTextureSampler, displacementTexcoord).rgb;
#ifdef USE_DISPLACEMENT_INTENSITY
	displacement *= scn_commonprofile.displacementIntensity;
#endif
#if defined(USE_NORMAL) && (defined(HAS_NORMAL) || defined(USE_OPENSUBDIV))
	float3 bitangent = geometry.tangent.w * normalize(cross(geometry.tangent.xyz, geometry.normal.xyz));
	geometry.position.xyz += geometry.tangent.xyz * displacement.x + geometry.normal.xyz * displacement.y + bitangent.xyz * displacement.z;
	
	float3 offset = float3(1./u_displacementTexture.get_width(), 1./u_displacementTexture.get_height(), 0.);
	float3 a = displacementTexture.sample(displacementTextureSampler, displacementTexcoord).rgb;
	float3 b = displacementTexture.sample(displacementTextureSampler, displacementTexcoord+offset.xz).rgb;
	float3 c = displacementTexture.sample(displacementTextureSampler, displacementTexcoord+offset.zy).rgb;
	
#ifdef USE_DISPLACEMENT_INTENSITY
	a *= scn_commonprofile.displacementIntensity;
	b *= scn_commonprofile.displacementIntensity;
	c *= scn_commonprofile.displacementIntensity;
#endif
	
	b += offset.xzz;
	c -= offset.zzy;
	float3 n = (normalize( cross( b-a, c-a ) ));
	geometry.normal = geometry.tangent.xyz * n.x + geometry.normal.xyz * n.y + bitangent.xyz * n.z;
	geometry.tangent.xyz = normalize(cross(bitangent, geometry.normal));
#endif // USE_NORMAL
#endif // USE_DISPLACEMENT_TEXTURE_COMPONENT
}
#endif // USE_DISPLACEMENT_MAP

#ifdef USE_OUTLINE
static inline float hash(float2 p)
{
	const float2 kMod2 = float2(443.8975f, 397.2973f);
	p  = fract(p * kMod2);
	p += dot(p.xy, p.yx+19.19f);
	return fract(p.x * p.y);
}
#endif

// Vertex shader function

#ifndef USE_TESSELATION

vertex commonprofile_io commonprofile_vert(scn_vertex_t                       in                [[ stage_in ]]
                                           , constant SCNSceneBuffer&         scn_frame         [[ buffer(0) ]]
#ifdef USE_INSTANCING
                                           // we use device here to override the 64Ko limit of constant buffers on NV hardware
                                           , device commonprofile_node*       scn_nodeInstances [[ buffer(1) ]]
#else
                                           , constant commonprofile_node&     scn_node          [[ buffer(1) ]]
#endif
#ifdef USE_PER_VERTEX_LIGHTING
                                           , constant commonprofile_lights&   scn_lights        [[ buffer(2) ]]
#endif
// used for texture transform and materialShininess in case of perVertexLighting
                                           , constant commonprofile_uniforms& scn_commonprofile [[ buffer(3) ]]
                                           , uint                             scn_vertexID      [[ vertex_id ]]
                                           , uint                             scn_instanceID    [[ instance_id ]]

#ifdef USE_POINT_RENDERING
                                           // x:pointSize, y:minimumScreenSize, z:maximumScreenSize
                                           , constant float3&                 scn_pointSize     [[ buffer(4) ]]
#endif

#ifdef USE_DISPLACEMENT_MAP
										   , texture2d<float>                 u_displacementTexture        [[ texture(0) ]]
										   , sampler                          u_displacementTextureSampler [[ sampler(0) ]]
#endif
#ifdef USE_VERTEX_EXTRA_ARGUMENTS

#endif
                                           )
{
#ifdef USE_INSTANCING
    device commonprofile_node& scn_node = scn_nodeInstances[scn_instanceID];
#endif

    SCNShaderGeometry _geometry;
    // OPTIM in could be already float4?
    _geometry.position = float4(in.position, 1.f);
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
    _geometry.normal = in.normal;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    _geometry.tangent = in.tangent;
#endif
#ifdef NEED_IN_TEXCOORD0
    _geometry.texcoords[0] = in.texcoord0;
#endif
#ifdef NEED_IN_TEXCOORD1
    _geometry.texcoords[1] = in.texcoord1;
#endif
#ifdef NEED_IN_TEXCOORD2
    _geometry.texcoords[2] = in.texcoord2;
#endif
#ifdef NEED_IN_TEXCOORD3
    _geometry.texcoords[3] = in.texcoord3;
#endif
#ifdef NEED_IN_TEXCOORD4
    _geometry.texcoords[4] = in.texcoord4;
#endif
#ifdef NEED_IN_TEXCOORD5
    _geometry.texcoords[5] = in.texcoord5;
#endif
#ifdef NEED_IN_TEXCOORD6
    _geometry.texcoords[6] = in.texcoord6;
#endif
#ifdef NEED_IN_TEXCOORD7
    _geometry.texcoords[7] = in.texcoord7;
#endif
#ifdef HAS_VERTEX_COLOR
    _geometry.color = in.color;
#elif USE_VERTEX_COLOR
    _geometry.color = float4(1.);
#endif
#ifdef USE_POINT_RENDERING
    _geometry.pointSize = scn_pointSize.x;
#endif
    
#ifdef USE_TEXCOORD
    
#endif

#ifdef USE_SKINNING
    {
        float3 pos = 0.f;
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
        float3 nrm = 0.f;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
        float3 tgt = 0.f;
#endif
        for (int i = 0; i < MAX_BONE_INFLUENCES; ++i) {
#if MAX_BONE_INFLUENCES == 1
            float weight = 1.f;
#else
            float weight = in.skinningWeights[i];
            if (weight <= 0.f)
                continue;
 
#endif
            int idx = int(in.skinningJoints[i]) * 3;
            float4x4 jointMatrix = float4x4(scn_node.skinningJointMatrices[idx],
                                            scn_node.skinningJointMatrices[idx+1],
                                            scn_node.skinningJointMatrices[idx+2],
                                            float4(0., 0., 0., 1.));
            
            pos += (_geometry.position * jointMatrix).xyz * weight;
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
            nrm += _geometry.normal * scn::mat3(jointMatrix) * weight;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
            tgt += _geometry.tangent.xyz * scn::mat3(jointMatrix) * weight;
#endif
        }
        
        _geometry.position.xyz = pos;
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
        _geometry.normal = nrm;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
        _geometry.tangent.xyz = tgt;
#endif
    }
#endif // USE_SKINNING
    
    commonprofile_io out;
    

#ifdef USE_DISPLACEMENT_MAP
	applyDisplacement(u_displacementTexture, u_displacementTextureSampler, _displacementTexcoord, _geometry, scn_commonprofile);
#endif
	
#ifdef USE_GEOMETRY_MODIFIER
// DoGeometryModifier START

// DoGeometryModifier END
#endif
    
    // Transform the geometry elements in view space
#if defined(USE_POSITION) || (defined(USE_NORMAL) && defined(HAS_NORMAL)) || defined(USE_TANGENT) || defined(USE_BITANGENT) || defined(USE_INSTANCING)
    SCNShaderSurface _surface;
#endif
#if defined(USE_POSITION) || defined(USE_INSTANCING)
    _surface.position = (scn_node.modelViewTransform * _geometry.position).xyz;
#endif
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
    _surface.normal = normalize(scn::mat3(scn_node.normalTransform) * _geometry.normal);
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    _surface.tangent = normalize(scn::mat3(scn_node.normalTransform) * _geometry.tangent.xyz);
    _surface.bitangent = _geometry.tangent.w * cross(_surface.tangent, _surface.normal); // no need to renormalize since tangent and normal should be orthogonal
    // old code : _surface.bitangent =  normalize(cross(_surface.normal,_surface.tangent));
#endif
    
    //if USE_VIEW is 2 we may also need to set _surface.view. todo: make USE_VIEW a mask
#ifdef USE_VIEW
    _surface.view = normalize(-_surface.position);
#endif
    
#ifdef USE_PER_VERTEX_LIGHTING
    // Lighting
    SCNShaderLightingContribution _lightingContribution;
    _lightingContribution.diffuse = 0.;
  #ifdef USE_SPECULAR
    _lightingContribution.specular = 0.;
    _surface.shininess = scn_commonprofile.materialShininess;
  #endif
    
    out.diffuse = _lightingContribution.diffuse;
  #ifdef USE_SPECULAR
    out.specular = _lightingContribution.specular;
  #endif
#endif
    
#if defined(USE_POSITION) && (USE_POSITION == 2)
    out.position = _surface.position;
#endif
#if defined(USE_NORMAL) && (USE_NORMAL == 2) && defined(HAS_NORMAL)
    out.normal = _surface.normal;
#endif
#if defined(USE_TANGENT) && (USE_TANGENT == 2)
    out.tangent = _surface.tangent;
#endif
#if defined(USE_BITANGENT) && (USE_BITANGENT == 2)
    out.bitangent = _surface.bitangent;
#endif
#ifdef USE_VERTEX_COLOR
    out.vertexColor = _geometry.color;
#endif
#ifdef USE_TEXCOORD

#endif
    
#if defined(USE_POSITION) || defined(USE_INSTANCING)
    out.fragmentPosition = scn_frame.projectionTransform * float4(_surface.position, 1.);
#elif defined(USE_MODELVIEWPROJECTIONTRANSFORM) // this means that the geometry are still in model space : we can transform it directly to NDC space
    out.fragmentPosition = scn_node.modelViewProjectionTransform * _geometry.position;
#endif
#ifdef USE_NODE_OPACITY
    out.nodeOpacity = scn_node.nodeOpacity;
#endif
#ifdef USE_DOUBLE_SIDED
    out.orientationPreserved = scn_node.orientationPreserved;
#endif
#ifdef USE_POINT_RENDERING
    float screenSize = _geometry.pointSize / out.fragmentPosition.w;
    out.fragmentSize = clamp(screenSize, scn_pointSize.y, scn_pointSize.z);
#endif
    
#ifdef USE_MOTIONBLUR
    float4 lastFrameFragmentPosition = scn_node.lastFrameModelViewProjectionTransform * _geometry.position;
    out.velocity.xy = lastFrameFragmentPosition.xy * float2(1., -1.);
    out.velocity.z = lastFrameFragmentPosition.w;
#endif
#ifdef USE_OUTLINE
	out.outlineHash = hash(scn_node.modelTransform[3].xy)+1.f/255.f;
#endif
    return out;
}

#else // #ifndef USE_TESSELATION

#if __METAL_VERSION__ >= 120
struct scn_patch_t {
    patch_control_point<scn_vertex_t> controlPoints;
};

#ifdef USE_OPENSUBDIV
#if OSD_IS_ADAPTIVE
[[ patch(quad, VERTEX_CONTROL_POINTS_PER_PATCH) ]]
#endif
#else // USE_OPENSUBDIV
[[ patch(triangle, 3) ]]
#endif // USE_OPENSUBDIV
vertex commonprofile_io commonprofile_post_tessellation_vert(
#ifdef USE_OPENSUBDIV
#if OSD_IS_ADAPTIVE
#if USE_STAGE_IN
                                                             PatchInput                         patchInput                   [[ stage_in ]]
#else
                                                             OsdVertexBufferSet                 patchInput
#endif
                                                             , float2                           patchCoord                   [[ position_in_patch ]]
                                                             , uint                             patchID                      [[ patch_id ]]
                                                             , constant float&                  osdTessellationLevel         [[ buffer(TESSELLATION_LEVEL_BUFFER_INDEX) ]]
#else // OSD_IS_ADAPTIVE
                                                             device unsigned const*             osdIndicesBuffer             [[ buffer(INDICES_BUFFER_INDEX) ]]
                                                             , device osd_packed_vertex const*  osdVertexBuffer              [[ buffer(VERTEX_BUFFER_INDEX) ]]
                                                             , uint                             vertexID                     [[ vertex_id ]]
#endif // OSD_IS_ADAPTIVE
#if defined(OSD_FVAR_WIDTH)
                                                             , device float const*              osdFaceVaryingData           [[ buffer(OSD_FVAR_DATA_BUFFER_INDEX) ]]
                                                             , device int const*                osdFaceVaryingIndices        [[ buffer(OSD_FVAR_INDICES_BUFFER_INDEX) ]]
#if OSD_IS_ADAPTIVE
                                                             , device packed_int3 const*        osdFaceVaryingPatchParams    [[ buffer(OSD_FVAR_PATCHPARAM_BUFFER_INDEX) ]]
                                                             , constant packed_int4&            osdFaceVaryingPatchArray     [[ buffer(OSD_FVAR_PATCH_ARRAY_BUFFER_INDEX) ]]
#endif
#endif //defined(OSD_FVAR_WIDTH)
#else // USE_OPENSUBDIV
                                                             scn_patch_t                        in                           [[ stage_in ]]
                                                             , float3                           patchCoord                   [[ position_in_patch ]]
#endif // USE_OPENSUBDIV
                                                             , constant SCNSceneBuffer&         scn_frame                    [[ buffer(0) ]]
#ifdef USE_INSTANCING
                                                             // we use device here to override the 64Ko limit of constant buffers on NV hardware
                                                             , device commonprofile_node*       scn_nodeInstances            [[ buffer(1) ]]
#else
                                                             , constant commonprofile_node&     scn_node                     [[ buffer(1) ]]
#endif
#ifdef USE_PER_VERTEX_LIGHTING
                                                             , constant commonprofile_lights&   scn_lights                   [[ buffer(2) ]]
#endif
                                                             // used for texture transform and materialShininess in case of perVertexLighting
                                                             , constant commonprofile_uniforms& scn_commonprofile            [[ buffer(3) ]]
                                                             , uint                             scn_instanceID               [[ instance_id ]]
#ifdef USE_VERTEX_EXTRA_ARGUMENTS
                                                             
#endif
#ifdef USE_DISPLACEMENT_MAP
                                                             , texture2d<float>                 u_displacementTexture        [[ texture(0) ]]
                                                             , sampler                          u_displacementTextureSampler [[ sampler(0) ]]
#endif
                                                             )
{
#ifdef USE_INSTANCING
    device commonprofile_node& scn_node = scn_nodeInstances[scn_instanceID];
#endif
    uint scn_vertexID; // we need scn_vertexID if a geometry modifier is used
    scn_vertexID = 0;
    SCNShaderGeometry _geometry;

#ifdef USE_OPENSUBDIV
#if OSD_IS_ADAPTIVE
#if USE_STAGE_IN
    int3 patchParam = patchInput.patchParam;
#else
    int3 patchParam = patchInput.patchParamBuffer[patchID];
#endif
    
    int refinementLevel = OsdGetPatchRefinementLevel(patchParam);
    float tessellationLevel = min(osdTessellationLevel, (float)OSD_MAX_TESS_LEVEL) / exp2((float)refinementLevel - 1);
    
    OsdPatchVertex patchVertex = OsdComputePatch(tessellationLevel, patchCoord, patchID, patchInput);
    
#if defined(OSD_FVAR_WIDTH)
    int patchIndex = OsdGetPatchIndex(patchID);
    OsdInterpolateFaceVarings(_geometry, patchCoord.xy, patchIndex, osdFaceVaryingIndices, osdFaceVaryingData, osdFaceVaryingPatchParams, osdFaceVaryingPatchArray);
#endif
    
    _geometry.position = float4(patchVertex.position, 1.f);
    
#if defined(USE_NORMAL)
    _geometry.normal = patchVertex.normal;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    _geometry.tangent = float4(patchVertex.tangent, -1.f);
    //_geometry.bitangent = patchVertex.bitangent;
#endif
#if defined(NEED_IN_TEXCOORD0) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[0] = patchVertex.texcoord0;
#endif
#if defined(NEED_IN_TEXCOORD1) && (OSD_TEXCOORD1_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[1] = patchVertex.texcoord1;
#endif
#if defined(NEED_IN_TEXCOORD2) && (OSD_TEXCOORD2_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[2] = patchVertex.texcoord2;
#endif
#if defined(NEED_IN_TEXCOORD3) && (OSD_TEXCOORD3_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[3] = patchVertex.texcoord3;
#endif
#if defined(NEED_IN_TEXCOORD4) && (OSD_TEXCOORD4_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[4] = patchVertex.texcoord4;
#endif
#if defined(NEED_IN_TEXCOORD5) && (OSD_TEXCOORD5_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[5] = patchVertex.texcoord5;
#endif
#if defined(NEED_IN_TEXCOORD6) && (OSD_TEXCOORD6_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[6] = patchVertex.texcoord6;
#endif
#if defined(NEED_IN_TEXCOORD7) && (OSD_TEXCOORD7_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[7] = patchVertex.texcoord7;
#endif
#if defined(HAS_VERTEX_COLOR) && (OSD_COLOR_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.color = patchVertex.color;
#elif defined(USE_VERTEX_COLOR)
    _geometry.color = float4(1.);
#endif
    
#else //OSD_IS_ADAPTIVE

#if OSD_PATCH_QUADS
    const uint primitiveIndex = vertexID / 6;
#ifdef USE_NORMAL
    float3 p0 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 4 + 0]].position;
    float3 p1 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 4 + 1]].position;
    float3 p2 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 4 + 2]].position;
    float3 normal = normalize(cross(p2 - p1, p0 - p1));
#endif
    const uint triangleIndices[6] = { 0, 1, 2, 0, 2, 3 };
    const uint quadVertexIndex = triangleIndices[vertexID % 6];
    osd_packed_vertex osdVertex = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 4 + quadVertexIndex]];
#elif OSD_PATCH_TRIANGLES
    const uint primitiveIndex = vertexID / 3;
#ifdef USE_NORMAL
    float3 p0 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 3 + 0]].position;
    float3 p1 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 3 + 1]].position;
    float3 p2 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 3 + 2]].position;
    float3 normal = normalize(cross(p2 - p1, p0 - p1));
#endif
    osd_packed_vertex osdVertex = osdVertexBuffer[osdIndicesBuffer[vertexID]];
#endif
    
    float3 position = osdVertex.position;
    
#if defined(OSD_FVAR_WIDTH)
    int patchIndex = OsdGetPatchIndex(primitiveIndex);

#if OSD_PATCH_QUADS
    float2 quadUVs[4] = { float2(0,0), float2(1,0), float2(1,1), float2(0,1) };
    OsdInterpolateFaceVarings(_geometry, quadUVs[quadVertexIndex], patchIndex, osdFaceVaryingIndices, osdFaceVaryingData);
#elif OSD_PATCH_TRIANGLES
    //TODO
#endif
#endif
    
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    //_geometry.tangent = float4(osdVertex.tangent, 1);
    //_geometry.bitangent = osdVertex.bitangent;
#endif
#if defined(NEED_IN_TEXCOORD0) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[0] = osdVertex.texcoord0;
#endif
#if defined(NEED_IN_TEXCOORD1) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[1] = osdVertex.texcoord1;
#endif
#if defined(NEED_IN_TEXCOORD2) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[2] = osdVertex.texcoord2;
#endif
#if defined(NEED_IN_TEXCOORD3) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[3] = osdVertex.texcoord3;
#endif
#if defined(NEED_IN_TEXCOORD4) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[4] = osdVertex.texcoord4;
#endif
#if defined(NEED_IN_TEXCOORD5) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[5] = osdVertex.texcoord5;
#endif
#if defined(NEED_IN_TEXCOORD5) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[6] = osdVertex.texcoord6;
#endif
#if defined(NEED_IN_TEXCOORD7) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[7] = osdVertex.texcoord7;
#endif
#if defined(HAS_VERTEX_COLOR) && (OSD_COLOR_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.color = osdVertex.color;
#elif defined(USE_VERTEX_COLOR)
    _geometry.color = float4(1.);
#endif
    
    _geometry.position = float4(position, 1.f);
#ifdef USE_NORMAL
    _geometry.normal = normal;
#endif
 
#endif //OSD_IS_ADAPTIVE
    
#else // USE_OPENSUBDIV
    
#if defined(TESSELLATION_SMOOTHING_MODE_PN_TRIANGLE) || defined(TESSELLATION_SMOOTHING_MODE_PHONG)
    
    float3 P0 = in.controlPoints[0].position;
    float3 P1 = in.controlPoints[1].position;
    float3 P2 = in.controlPoints[2].position;
    
    float3 N0 = in.controlPoints[0].normal;
    float3 N1 = in.controlPoints[1].normal;
    float3 N2 = in.controlPoints[2].normal;
    
#if defined(TESSELLATION_SMOOTHING_MODE_PN_TRIANGLE)
    float3 position, normal;
    scn_smooth_geometry_pn_triangle(position, normal, patchCoord, P0, P1, P2, N0, N1, N2);
#elif defined(TESSELLATION_SMOOTHING_MODE_PHONG)
    float3 position, normal;
    scn_smooth_geometry_phong(position, normal, patchCoord, P0, P1, P2, N0, N1, N2);
#endif
    
    _geometry.position = float4(position, 1.f);
#ifdef USE_NORMAL
    _geometry.normal = normal;
#endif

#else // GEOMETRY_SMOOTHING
    
    // OPTIM in could be already float4?
    _geometry.position = float4(scn::barycentric_mix(in.controlPoints[0].position, in.controlPoints[1].position, in.controlPoints[2].position, patchCoord), 1.f);
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
    _geometry.normal = normalize(scn::barycentric_mix(in.controlPoints[0].normal, in.controlPoints[1].normal, in.controlPoints[2].normal, patchCoord));
#endif
    
#endif // GEOMETRY_SMOOTHING
    
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    _geometry.tangent = normalize(scn::barycentric_mix(in.controlPoints[0].tangent, in.controlPoints[1].tangent, in.controlPoints[2].tangent, patchCoord));
#endif
#ifdef NEED_IN_TEXCOORD0
    _geometry.texcoords[0] = scn::barycentric_mix(in.controlPoints[0].texcoord0, in.controlPoints[1].texcoord0, in.controlPoints[2].texcoord0, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD1
    _geometry.texcoords[1] = scn::barycentric_mix(in.controlPoints[0].texcoord1, in.controlPoints[1].texcoord1, in.controlPoints[2].texcoord1, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD2
    _geometry.texcoords[2] = scn::barycentric_mix(in.controlPoints[0].texcoord2, in.controlPoints[1].texcoord2, in.controlPoints[2].texcoord2, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD3
    _geometry.texcoords[3] = scn::barycentric_mix(in.controlPoints[0].texcoord3, in.controlPoints[1].texcoord3, in.controlPoints[2].texcoord3, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD4
    _geometry.texcoords[4] = scn::barycentric_mix(in.controlPoints[0].texcoord4, in.controlPoints[1].texcoord4, in.controlPoints[2].texcoord4, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD5
    _geometry.texcoords[5] = scn::barycentric_mix(in.controlPoints[0].texcoord5, in.controlPoints[1].texcoord5, in.controlPoints[2].texcoord5, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD6
    _geometry.texcoords[6] = scn::barycentric_mix(in.controlPoints[0].texcoord6, in.controlPoints[1].texcoord6, in.controlPoints[2].texcoord6, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD7
    _geometry.texcoords[7] = scn::barycentric_mix(in.controlPoints[0].texcoord7, in.controlPoints[1].texcoord7, in.controlPoints[2].texcoord7, patchCoord);
#endif
#ifdef HAS_VERTEX_COLOR
    _geometry.color = scn::barycentric_mix(in.controlPoints[0].color, in.controlPoints[1].color, in.controlPoints[2].color, patchCoord);
#elif USE_VERTEX_COLOR
    _geometry.color = float4(1.);
#endif

#endif // USE_OPENSUBDIV
    
#ifdef USE_TEXCOORD
    
#endif
	
#ifdef USE_DISPLACEMENT_MAP
	applyDisplacement(u_displacementTexture, u_displacementTextureSampler, _displacementTexcoord, _geometry, scn_commonprofile);
#endif
    
#if defined(USE_SKINNING) && !defined(USE_OPENSUBDIV)
    {
        float3 pos[3] = {0.f, 0.f, 0.f};
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
        float3 nrm[3] = {0.f, 0.f, 0.f};
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
        float3 tgt[3] = {0.f, 0.f, 0.f};
#endif
        for (int controlPointIndex = 0; controlPointIndex < 3; ++controlPointIndex) {
            for (int i = 0; i < MAX_BONE_INFLUENCES; ++i) {
#if MAX_BONE_INFLUENCES == 1
                float weight = 1.f;
#else
                float weight = in.controlPoints[controlPointIndex].skinningWeights[i];
                if (weight <= 0.f)
                    continue;
                
#endif
                int idx = int(in.controlPoints[controlPointIndex].skinningJoints[i]) * 3;
                float4x4 jointMatrix = float4x4(scn_node.skinningJointMatrices[idx],
                                                scn_node.skinningJointMatrices[idx+1],
                                                scn_node.skinningJointMatrices[idx+2],
                                                float4(0., 0., 0., 1.));
                
                pos[controlPointIndex] += (_geometry.position * jointMatrix).xyz * weight;
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
                nrm[controlPointIndex] += _geometry.normal * scn::mat3(jointMatrix) * weight;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
                tgt[controlPointIndex] += _geometry.tangent.xyz * scn::mat3(jointMatrix) * weight;
#endif
            }
        }
        
        _geometry.position.xyz = scn::barycentric_mix(pos[0], pos[1], pos[2], patchCoord);
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
        _geometry.normal = scn::barycentric_mix(nrm[0], nrm[1], nrm[2], patchCoord);
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
        _geometry.tangent.xyz = scn::barycentric_mix(tgt[0], tgt[1], tgt[2], patchCoord);
#endif
    }
#endif // defined(USE_SKINNING) && !defined(USE_OPENSUBDIV)
    
    commonprofile_io out;
    
#ifdef USE_DISPLACEMENT_MAP
    out.displacementTexcoord = _displacementTexcoord;
#endif
    
#ifdef USE_GEOMETRY_MODIFIER
    // DoGeometryModifier START
    
    // DoGeometryModifier END
#endif
    
    // Transform the geometry elements in view space
#if defined(USE_POSITION) || defined(USE_NORMAL) || defined(USE_TANGENT) || defined(USE_BITANGENT) || defined(USE_INSTANCING)
    SCNShaderSurface _surface;
#endif
#if defined(USE_POSITION) || defined(USE_INSTANCING)
    _surface.position = (scn_node.modelViewTransform * _geometry.position).xyz;
#endif
#if defined(USE_NORMAL) && (defined(HAS_NORMAL) || defined(USE_OPENSUBDIV))
    _surface.normal = normalize(scn::mat3(scn_node.normalTransform) * _geometry.normal);
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    _surface.tangent = normalize(scn::mat3(scn_node.normalTransform) * _geometry.tangent.xyz);
    _surface.bitangent = _geometry.tangent.w * cross(_surface.tangent, _surface.normal); // no need to renormalize since tangent and normal should be orthogonal
    // old code : _surface.bitangent =  normalize(cross(_surface.normal,_surface.tangent));
#endif
    
    //if USE_VIEW is 2 we may also need to set _surface.view. todo: make USE_VIEW a mask
#ifdef USE_VIEW
    _surface.view = normalize(-_surface.position);
#endif
    
#ifdef USE_PER_VERTEX_LIGHTING
    // Lighting
    SCNShaderLightingContribution _lightingContribution;
    _lightingContribution.diffuse = 0.;
  #ifdef USE_SPECULAR
    _lightingContribution.specular = 0.;
    _surface.shininess = scn_commonprofile.materialShininess;
  #endif
    
    out.diffuse = _lightingContribution.diffuse;
  #ifdef USE_SPECULAR
    out.specular = _lightingContribution.specular;
  #endif
#endif
    
#if defined(USE_POSITION) && (USE_POSITION == 2)
    out.position = _surface.position;
#endif
#if defined(USE_NORMAL) && (USE_NORMAL == 2) && (defined(HAS_NORMAL) || defined(USE_OPENSUBDIV))
    out.normal = _surface.normal;
#endif
#if defined(USE_TANGENT) && (USE_TANGENT == 2)
    out.tangent = _surface.tangent;
#endif
#if defined(USE_BITANGENT) && (USE_BITANGENT == 2)
    out.bitangent = _surface.bitangent;
#endif
#ifdef USE_VERTEX_COLOR
    out.vertexColor = _geometry.color;
#endif
#ifdef USE_TEXCOORD
    
#endif
    
#if defined(USE_POSITION) || defined(USE_INSTANCING)
    out.fragmentPosition = scn_frame.projectionTransform * float4(_surface.position, 1.);
#elif defined(USE_MODELVIEWPROJECTIONTRANSFORM) // this means that the geometry are still in model space : we can transform it directly to NDC space
    out.fragmentPosition = scn_node.modelViewProjectionTransform * _geometry.position;
#endif
    
#ifdef USE_NODE_OPACITY
    out.nodeOpacity = scn_node.nodeOpacity;
#endif
#ifdef USE_DOUBLE_SIDED
    out.orientationPreserved = scn_node.orientationPreserved;
#endif
#ifdef USE_MOTIONBLUR
	float4 lastFrameFragmentPosition = scn_node.lastFrameModelViewProjectionTransform * _geometry.position;
	out.velocity.xy = lastFrameFragmentPosition.xy * float2(1., -1.);
	out.velocity.z = lastFrameFragmentPosition.w;
#endif
#ifdef USE_OUTLINE
	out.outlineHash = hash(scn_node.modelTransform[3].xy)+1.f/255.f;
#endif
    return out;
}
#endif // __METAL_VERSION__
#endif // #ifndef USE_TESSELATION


struct SCNOutput
{
    float4 color [[ color(0) ]];
#ifdef USE_MOTIONBLUR
    half4 motionblur [[ color(1) ]];
#endif
};

// Fragment shader function
fragment SCNOutput commonprofile_frag(commonprofile_io                 in                         [[stage_in]],
                                  constant commonprofile_uniforms& scn_commonprofile          [[buffer(0)]],
                                  constant SCNSceneBuffer&         scn_frame                  [[buffer(1)]]
#ifdef USE_PER_PIXEL_LIGHTING
                                  , constant commonprofile_lights& scn_lights                 [[buffer(2)]]
#endif
#ifdef USE_EMISSION_MAP
                                  , texture2d<float>              u_emissionTexture           [[texture(0)]]
                                  , sampler                       u_emissionTextureSampler    [[sampler(0)]]
#endif
#ifdef USE_AMBIENT_MAP
                                  , texture2d<float>              u_ambientTexture            [[texture(1)]]
                                  , sampler                       u_ambientTextureSampler     [[sampler(1)]]
#endif
#ifdef USE_DIFFUSE_MAP
                                  , texture2d<float>              u_diffuseTexture            [[texture(2)]]
                                  , sampler                       u_diffuseTextureSampler     [[sampler(2)]]
#endif
#ifdef USE_SPECULAR_MAP
                                  , texture2d<float>              u_specularTexture           [[texture(3)]]
                                  , sampler                       u_specularTextureSampler    [[sampler(3)]]
#endif
#ifdef USE_REFLECTIVE_MAP
                                  , texture2d<float>              u_reflectiveTexture         [[texture(4)]]
                                  , sampler                       u_reflectiveTextureSampler  [[sampler(4)]]
#elif defined(USE_REFLECTIVE_CUBEMAP)
                                  , texturecube<float>            u_reflectiveTexture         [[texture(4)]]
                                  , sampler                       u_reflectiveTextureSampler  [[sampler(4)]]
#endif
#ifdef USE_TRANSPARENT_MAP
                                  , texture2d<float>              u_transparentTexture        [[texture(5)]]
                                  , sampler                       u_transparentTextureSampler [[sampler(5)]]
#endif
#ifdef USE_MULTIPLY_MAP
                                  , texture2d<float>              u_multiplyTexture           [[texture(6)]]
                                  , sampler                       u_multiplyTextureSampler    [[sampler(6)]]
#endif
#ifdef USE_NORMAL_MAP
                                  , texture2d<float>              u_normalTexture             [[texture(7)]]
                                  , sampler                       u_normalTextureSampler      [[sampler(7)]]
#endif
#ifdef USE_SELFILLUMINATION_MAP
                                  , texture2d<float>              u_selfIlluminationTexture           [[texture(8)]]
                                  , sampler                       u_selfIlluminationTextureSampler    [[sampler(8)]]
#endif
#ifdef USE_PBR
#ifdef USE_METALNESS_MAP
                                  , texture2d<float>              u_metalnessTexture          [[texture(3)]]
                                  , sampler                       u_metalnessTextureSampler   [[sampler(3)]]
#endif
#ifdef USE_ROUGHNESS_MAP
                                  , texture2d<float>              u_roughnessTexture          [[texture(4)]]
                                  , sampler                       u_roughnessTextureSampler   [[sampler(4)]]
#endif
#ifdef USE_DISPLACEMENT_MAP
                                  , texture2d<float>              u_displacementTexture        [[ texture(12) ]]
                                  , sampler                       u_displacementTextureSampler [[ sampler(12) ]]
#endif
#if !defined(USE_SELFILLUMINATION_MAP)
                                  , texturecube<float>            u_irradianceTexture         [[texture(8)]]
#endif
                                  , texturecube<float>            u_radianceTexture           [[texture(9)]]
                                  , texture2d<float>              u_specularDFGTexture        [[texture(10)]]
#endif
#ifdef USE_SSAO
                                  , texture2d<float>              u_ssaoTexture               [[texture(11)]]
#endif
                                  , constant commonprofile_node&  scn_node                    [[buffer(3)]]
#ifdef USE_FRAGMENT_EXTRA_ARGUMENTS

#endif
#if defined(USE_DOUBLE_SIDED)
                                  , bool                          isFrontFacing               [[front_facing]]
#endif
#ifdef USE_POINT_RENDERING
                                  , float2                        pointCoord                  [[point_coord]]
#endif
                                  
                                  )
{
    SCNShaderSurface _surface;
#ifdef USE_TEXCOORD

#endif
    _surface.ambientOcclusion = 1.f; // default to no AO
#ifdef USE_AMBIENT_MAP
    #ifdef USE_AMBIENT_AS_AMBIENTOCCLUSION
        _surface.ambientOcclusion = u_ambientTexture.sample(u_ambientTextureSampler, _surface.ambientTexcoord).r;
        #ifdef USE_AMBIENT_INTENSITY
            _surface.ambientOcclusion = saturate(mix(1.f, _surface.ambientOcclusion, scn_commonprofile.ambientIntensity));
        #endif
    #else // AMBIENT_MAP
        _surface.ambient = u_ambientTexture.sample(u_ambientTextureSampler, _surface.ambientTexcoord);
        #ifdef USE_AMBIENT_INTENSITY
            _surface.ambient *= scn_commonprofile.ambientIntensity;
        #endif
    #endif // USE_AMBIENT_AS_AMBIENTOCCLUSION
#if defined(USE_AMBIENT_TEXTURE_COMPONENT)
    _surface.ambient = float4(_surface.ambient[USE_AMBIENT_TEXTURE_COMPONENT]);
#endif

#elif defined(USE_AMBIENT_COLOR)
    _surface.ambient = scn_commonprofile.ambientColor;
#elif defined(USE_AMBIENT)
    _surface.ambient = float4(0.);
#endif
#if defined(USE_AMBIENT) && defined(USE_VERTEX_COLOR)
    _surface.ambient *= in.vertexColor;
#endif
#if  defined(USE_SSAO)
    _surface.ambientOcclusion = u_ssaoTexture.sample( sampler(filter::linear), in.fragmentPosition.xy * scn_frame.inverseResolution.xy ).x;
#endif
    
#ifdef USE_DIFFUSE_MAP
    _surface.diffuse = u_diffuseTexture.sample(u_diffuseTextureSampler, _surface.diffuseTexcoord);
#if defined(USE_DIFFUSE_TEXTURE_COMPONENT)
    _surface.diffuse = float4(_surface.diffuse[USE_DIFFUSE_TEXTURE_COMPONENT]);
#endif
#ifdef USE_DIFFUSE_INTENSITY
    _surface.diffuse.rgb *= scn_commonprofile.diffuseIntensity;
#endif
#elif defined(USE_DIFFUSE_COLOR)
    _surface.diffuse = scn_commonprofile.diffuseColor;
#else
    _surface.diffuse = float4(0.f,0.f,0.f,1.f);
#endif
#if defined(USE_DIFFUSE) && defined(USE_VERTEX_COLOR)
    _surface.diffuse *= in.vertexColor;
#endif
#ifdef USE_SPECULAR_MAP
    _surface.specular = u_specularTexture.sample(u_specularTextureSampler, _surface.specularTexcoord);
#if defined(USE_SPECULAR_TEXTURE_COMPONENT)
    _surface.specular = float4(_surface.specular[USE_SPECULAR_TEXTURE_COMPONENT]);
#endif
#ifdef USE_SPECULAR_INTENSITY
    _surface.specular *= scn_commonprofile.specularIntensity;
#endif
#elif defined(USE_SPECULAR_COLOR)
    _surface.specular = scn_commonprofile.specularColor;
#elif defined(USE_SPECULAR)
    _surface.specular = float4(0.f);
#endif
#ifdef USE_EMISSION_MAP
    _surface.emission = u_emissionTexture.sample(u_emissionTextureSampler, _surface.emissionTexcoord);
#if defined(USE_EMISSION_TEXTURE_COMPONENT)
    _surface.emission = float4(_surface.emission[USE_EMISSION_TEXTURE_COMPONENT]);
#endif
#ifdef USE_EMISSION_INTENSITY
    _surface.emission *= scn_commonprofile.emissionIntensity;
#endif
#elif defined(USE_EMISSION_COLOR)
    _surface.emission = scn_commonprofile.emissionColor;
#elif defined(USE_EMISSION)
    _surface.emission = float4(0.);
#endif
#ifdef USE_SELFILLUMINATION_MAP
    _surface.selfIllumination = u_selfIlluminationTexture.sample(u_selfIlluminationTextureSampler, _surface.selfIlluminationTexcoord);
#if defined(USE_SELFILLUMINATION_TEXTURE_COMPONENT)
    _surface.selfIllumination = float4(_surface.selfIllumination[USE_SELFILLUMINATION_TEXTURE_COMPONENT]);
#endif
#ifdef USE_SELFILLUMINATION_INTENSITY
    _surface.selfIllumination *= scn_commonprofile.selfIlluminationIntensity;
#endif
#elif defined(USE_SELFILLUMINATION_COLOR)
    _surface.selfIllumination = scn_commonprofile.selfIlluminationColor;
#elif defined(USE_SELFILLUMINATION)
    _surface.selfIllumination = float4(0.);
#endif
#ifdef USE_MULTIPLY_MAP
    _surface.multiply = u_multiplyTexture.sample(u_multiplyTextureSampler, _surface.multiplyTexcoord);
#if defined(USE_MULTIPLY_TEXTURE_COMPONENT)
    _surface.multiply = float4(_surface.multiply[USE_MULTIPLY_TEXTURE_COMPONENT]);
#endif
#ifdef USE_MULTIPLY_INTENSITY
    _surface.multiply = mix(float4(1.), _surface.multiply, scn_commonprofile.multiplyIntensity);
#endif
#elif defined(USE_MULTIPLY_COLOR)
    _surface.multiply = scn_commonprofile.multiplyColor;
#elif defined(USE_MULTIPLY)
    _surface.multiply = float4(1.);
#endif
#ifdef USE_TRANSPARENT_MAP
    _surface.transparent = u_transparentTexture.sample(u_transparentTextureSampler, _surface.transparentTexcoord);
#if defined(USE_TRANSPARENT_TEXTURE_COMPONENT)
    _surface.transparent = float4(_surface.transparent[USE_TRANSPARENT_TEXTURE_COMPONENT]);
#endif
#ifdef USE_TRANSPARENT_INTENSITY
    _surface.transparent *= scn_commonprofile.transparentIntensity;
#endif
#elif defined(USE_TRANSPARENT_COLOR)
    _surface.transparent = scn_commonprofile.transparentColor;
#elif defined(USE_TRANSPARENT)
    _surface.transparent = float4(1.);
#endif
    
#ifdef USE_METALNESS_MAP
#if defined(USE_METALNESS_TEXTURE_COMPONENT)
    _surface.metalness = float4(u_metalnessTexture.sample(u_metalnessTextureSampler, _surface.metalnessTexcoord))[USE_METALNESS_TEXTURE_COMPONENT];
#else
    _surface.metalness = u_metalnessTexture.sample(u_metalnessTextureSampler, _surface.metalnessTexcoord).r;
#endif
#ifdef USE_METALNESS_INTENSITY
    _surface.metalness *= scn_commonprofile.metalnessIntensity;
#endif
#elif defined(USE_METALNESS_COLOR)
    _surface.metalness = scn_commonprofile.metalness;
#else
    _surface.metalness = 0;
#endif
    
#ifdef USE_ROUGHNESS_MAP
#if defined(USE_ROUGHNESS_TEXTURE_COMPONENT)
    _surface.roughness = float4(u_roughnessTexture.sample(u_roughnessTextureSampler, _surface.roughnessTexcoord))[USE_ROUGHNESS_TEXTURE_COMPONENT];
#else
    _surface.roughness = u_roughnessTexture.sample(u_roughnessTextureSampler, _surface.roughnessTexcoord).r;
#endif
#ifdef USE_ROUGHNESS_INTENSITY
    _surface.roughness *= scn_commonprofile.roughnessIntensity;
#endif
#elif defined(USE_ROUGHNESS_COLOR)
    _surface.roughness = scn_commonprofile.roughness;
#else
    _surface.roughness = 0;
#endif
#if (defined USE_POSITION) && (USE_POSITION == 2)
    _surface.position = in.position;
#endif
#if (defined USE_NORMAL) && (USE_NORMAL == 2)
#if defined(HAS_NORMAL) || defined(USE_OPENSUBDIV)
#ifdef USE_DOUBLE_SIDED
    _surface.geometryNormal = normalize(in.normal.xyz) * in.orientationPreserved * ((float(isFrontFacing) * 2.f) - 1.f);
#else
    _surface.geometryNormal = normalize(in.normal.xyz);
#endif
#else // need to generate the normal from the derivatives
    _surface.geometryNormal = normalize( cross(dfdy( _surface.position ), dfdx( _surface.position ) ));
#endif
    _surface.normal = _surface.geometryNormal;
#endif
#if defined(USE_TANGENT) && (USE_TANGENT == 2)
    _surface.tangent = in.tangent;
#endif
#if defined(USE_BITANGENT) && (USE_BITANGENT == 2)
    _surface.bitangent = in.bitangent;
#endif
#if (defined USE_VIEW) && (USE_VIEW == 2)
    _surface.view = normalize(-in.position);
#endif
#if defined(USE_POSITION)
//    {
//    float3 p = in.position;
//    float3 dpdx = dfdx(p);
//    float3 dpdy = dfdy(p);
//        _surface.normal.rgb = normalize( cross(dpdx, dpdy) );
//    }
#endif
    
#if defined(USE_NORMAL_MAP)
    {
        float3x3 ts2vs = float3x3(_surface.tangent, _surface.bitangent, _surface.normal);
#ifdef USE_NORMAL_MAP
        _surface._normalTS = u_normalTexture.sample(u_normalTextureSampler, _surface.normalTexcoord).rgb;
#if defined(USE_NORMAL_TEXTURE_COMPONENT)
        _surface._normalTS.xy = _surface._normalTS.xy * 2.f - 1.f;
        _surface._normalTS.z = sqrt(1.f - length_squared(_surface._normalTS.xy));
#else
        _surface._normalTS = _surface._normalTS * 2.f - 1.f;
#endif
#ifdef USE_NORMAL_INTENSITY
        _surface._normalTS = mix(float3(0.f, 0.f, 1.f), _surface._normalTS, scn_commonprofile.normalIntensity);
#endif
#else
        _surface._normalTS = float3(0.f, 0.f, 1.f);
#endif
        _surface.normal.rgb = normalize(ts2vs * _surface._normalTS.xyz );
    }
#else
    _surface._normalTS = float3(0.f, 0.f, 1.f);
#endif
#ifdef USE_REFLECTIVE_MAP
    float3 refl = reflect( -_surface.view, _surface.normal );
    float m = 2.f * sqrt( refl.x*refl.x + refl.y*refl.y + (refl.z+1.f)*(refl.z+1.f));
    _surface.reflective = u_reflectiveTexture.sample(u_reflectiveTextureSampler, float2(float2(refl.x,-refl.y) / m) + 0.5f);
#if defined(USE_REFLECTIVE_TEXTURE_COMPONENT)
    _surface.reflective = float4(_surface.reflective[USE_REFLECTIVE_TEXTURE_COMPONENT]);
#endif
#ifdef USE_REFLECTIVE_INTENSITY
    _surface.reflective *= scn_commonprofile.reflectiveIntensity;
#endif
#elif defined(USE_REFLECTIVE_CUBEMAP)
    float3 refl = reflect( _surface.position, _surface.normal );
    _surface.reflective = u_reflectiveTexture.sample(u_reflectiveTextureSampler, scn::mat4_mult_float3(scn_frame.viewToCubeTransform, refl)); // sample the cube map in world space
#ifdef USE_REFLECTIVE_INTENSITY
    _surface.reflective *= scn_commonprofile.reflectiveIntensity;
#endif
#elif defined(USE_REFLECTIVE_COLOR)
    _surface.reflective = scn_commonprofile.reflectiveColor;
#elif defined(USE_REFLECTIVE)
    _surface.reflective = float4(0.);
#endif
#ifdef USE_FRESNEL
    _surface.fresnel = scn_commonprofile.fresnel.x + scn_commonprofile.fresnel.y * pow(1.f - saturate(dot(_surface.view, _surface.normal)), scn_commonprofile.fresnel.z);
    _surface.reflective *= _surface.fresnel;
#endif
#ifdef USE_SHININESS
    _surface.shininess = scn_commonprofile.materialShininess;
#endif

#ifdef USE_SURFACE_MODIFIER
// DoSurfaceModifier START

// DoSurfaceModifier END
#endif
    // Lighting
    SCNShaderLightingContribution _lightingContribution = {0};
    
    
    // Lighting
#ifdef USE_AMBIENT_LIGHTING
    _lightingContribution.ambient = scn_frame.ambientLightingColor.rgb;
#endif
    
#ifdef USE_LIGHTING
#ifdef USE_PER_PIXEL_LIGHTING
    _lightingContribution.diffuse = float3(0.);
#ifdef USE_MODULATE
    _lightingContribution.modulate = float3(1.);
#endif
#ifdef USE_SPECULAR
    _lightingContribution.specular = float3(0.);
#endif
{
    SCNShaderLight _light;
    _light.intensity = scn_lights.color0;
    _light.direction = scn_lights.direction0.xyz;
    _light._att = 1.;
    _light.intensity.rgb *= _light._att * max(0.f, dot(_surface.normal, _light.direction));
    _lightingContribution.diffuse += _light.intensity.rgb;
}

#else // USE_PER_PIXEL_LIGHTING
    _lightingContribution.diffuse = in.diffuse;
#ifdef USE_SPECULAR
    _lightingContribution.specular = in.specular;
#endif
#endif
#ifdef AVOID_OVERLIGHTING
    _lightingContribution.diffuse = saturate(_lightingContribution.diffuse);
#ifdef USE_SPECULAR
    _lightingContribution.specular = saturate(_lightingContribution.specular);
#endif // USE_SPECULAR
#endif // AVOID_OVERLIGHTING
#else // USE_LIGHTING
    _lightingContribution.diffuse = float3(1.);
#endif // USE_LIGHTING

    // Combine
    SCNOutput _output;
#ifdef USE_PBR
    SCNPBRSurface pbr_surface = SCNShaderSurfaceToSCNPBRSurface(_surface);
    pbr_surface.selfIlluminationOcclusion = scn_commonprofile.selfIlluminationOcclusion;
#ifdef USE_PROBES_LIGHTING
    _output.color = scn_pbr_combine(pbr_surface, _lightingContribution, u_specularDFGTexture, u_radianceTexture, scn_node.shCoefficients, scn_frame);
#elif defined(USE_SELFILLUMINATION_MAP)
    _output.color = scn_pbr_combine(pbr_surface, _lightingContribution, u_specularDFGTexture, u_radianceTexture, u_radianceTexture, scn_frame);
#else
    _output.color = scn_pbr_combine(pbr_surface, _lightingContribution, u_specularDFGTexture, u_radianceTexture, u_irradianceTexture, scn_frame);
#endif
    _output.color.a = _surface.diffuse.a;
#else
    _output.color = illuminate(_surface, _lightingContribution);
#endif
    
#ifdef USE_FOG
    float fogFactor = pow(clamp(length(_surface.position.xyz) * scn_frame.fogParameters.x + scn_frame.fogParameters.y, 0., scn_frame.fogColor.a), scn_frame.fogParameters.z);
    _output.color.rgb = mix(_output.color.rgb, scn_frame.fogColor.rgb * _output.color.a, fogFactor);
#endif

#ifndef DIFFUSE_PREMULTIPLIED
    _output.color.rgb *= _surface.diffuse.a;
#endif

#ifdef USE_TRANSPARENT // Either a map or a color
    
#ifdef USE_TRANSPARENCY
    _surface.transparent *= scn_commonprofile.transparency;
#endif
    
#ifdef USE_TRANSPARENCY_RGBZERO
#ifdef USE_NODE_OPACITY
    _output.color *= in.nodeOpacity;
#endif
    // compute luminance
    _surface.transparent.a = (_surface.transparent.r * 0.212671f) + (_surface.transparent.g * 0.715160f) + (_surface.transparent.b * 0.072169f);
    _output.color *= (float4(1.f) - _surface.transparent);
#else // ALPHA_ONE
#ifdef USE_NODE_OPACITY
    _output.color *= (in.nodeOpacity * _surface.transparent.a);
#else
    _output.color *= _surface.transparent.a;
#endif
#endif
#else
#ifdef USE_TRANSPARENCY // TRANSPARENCY without TRANSPARENT slot (nodeOpacity + diffuse.a)
#ifdef USE_NODE_OPACITY
    _output.color *= (in.nodeOpacity * scn_commonprofile.transparency);
#else
    _output.color *= scn_commonprofile.transparency;
#endif // NODE_OPACITY
#endif
#endif
    
#ifdef USE_FRAGMENT_MODIFIER
// DoFragmentModifier START

// DoFragmentModifier END
#endif
    
#ifdef DISABLE_LINEAR_RENDERING
    _output.color.rgb = scn::linear_to_srgb(_output.color.rgb);
#endif
    
#ifdef USE_DISCARD
    if (_output.color.a == 0.) // we could set a different limit here
        discard_fragment();
#endif

#ifdef USE_POINT_RENDERING
    if ((dfdx(pointCoord.x) < 0.5f) && (length_squared(pointCoord * 2.f - 1.f) > 1.f)) {
        discard_fragment();
    }
#endif
    
#ifdef USE_MOTIONBLUR
    //_output.motionblur.xy = half2(in.fragmentPosition.xy*scn_frame.inverseResolution.xy*2.-1.);
    _output.motionblur.xy = (half2(in.fragmentPosition.xy*scn_frame.inverseResolution.xy*2.-1.) - half2(in.velocity.xy / in.velocity.z)) * scn_frame.motionBlurIntensity;
    _output.motionblur.z = length(_output.motionblur.xy);
    _output.motionblur.w = half(-_surface.position.z);
#endif

#ifdef USE_OUTLINE
	_output.color.rgb = in.outlineHash;
#endif
	
    return _output;
}
