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
