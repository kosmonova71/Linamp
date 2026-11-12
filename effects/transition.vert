#version 330 core
// Transition Vertex Shader
// Standard vertex shader for all transition effects

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

out vec2 v_uv;

void main() {
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
