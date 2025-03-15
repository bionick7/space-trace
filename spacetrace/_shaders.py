'''
	Custom shaders for spacetrace.
    vertex and fragment shaders in GLSL 3.3
'''

trajectory_shader_vs = """
#version 330

layout (location = 0) in vec3 point;
layout (location = 1) in float time;
layout (location = 2) in vec3 direction;

smooth out float y_offset;
smooth out float time_out;

uniform vec2 window_size;
uniform mat4 mvp;

const float STRIP_THICKNESS = 3.0;  // pixels
const float EPSILON = 0.01;

void main() {
	vec4 clip_pos = mvp * vec4(point, 1);
	vec4 direction_clip_delta = mvp * vec4(point + direction * EPSILON, 1);
	vec2 direction_clip = (direction_clip_delta.xy / direction_clip_delta.w - clip_pos.xy / clip_pos.w) / EPSILON;

	vec2 orth_direction_clip = normalize(direction_clip).yx * vec2(1,-1);
	vec2 ndc_offset = STRIP_THICKNESS / window_size;

    y_offset = -1.0 + float(gl_VertexID % 2) * 2.0;
	clip_pos.xy += y_offset * clip_pos.w * ndc_offset * orth_direction_clip;
	gl_Position = clip_pos;

	time_out = time;
}
"""

trajectory_shader_fs = """
#version 330

// Input vertex attributes (from vertex shader)
smooth in float y_offset;
smooth in float time_out;

// Input uniform values
uniform vec4 color;
uniform float current_t;

// Output fragment color
out vec4 finalColor;

void main() {
    finalColor = color;
    finalColor.a *= smoothstep(1.0f, 0.0f, abs(y_offset));
    if (current_t < time_out) {
        discard;
    }
}
"""

planet_shader_vs = """
#version 330

in vec3 vertexPosition;
in vec2 vertexTexCoord;

out vec2 fragTexCoord;

uniform mat4 mvp;

const float PI = 3.14159265359;

void main() {
    fragTexCoord = vertexTexCoord.yx;  // Not setup yet
    //fragTexCoord = vec2(
    //    atan(vertexPosition.x, vertexPosition.z) / (2 * PI) + 0.5,
    //	- 0.5 - asin(vertexPosition.y / length(vertexPosition)) / PI
    //);
    
	gl_Position = mvp * vec4(vertexPosition, 1.0);
}

"""

planet_shader_fs = """
#version 330

// Input uniform values
uniform vec4 fragColor;

uniform sampler2D albedoMap;
uniform sampler2D mraMap;
uniform sampler2D normalMap;

uniform vec2 tiling;
uniform vec2 offset;

uniform int useTexAlbedo;
uniform int useTexNormal;
uniform int useTexMRA;

uniform vec4  albedoColor;
uniform vec4  emissiveColor;
uniform float normalValue;
uniform float metallicValue;
uniform float roughnessValue;

//uniform vec3 light_direction

// Output fragment color
in vec2 fragTexCoord;
out vec4 finalColor;

void main() {
    vec3 albedo = texture(albedoMap, fragTexCoord).rgb;    
    finalColor = vec4(albedo, 1.0);
}
"""
