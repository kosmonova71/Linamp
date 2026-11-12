#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_texture1;
uniform sampler2D u_texture2;
uniform float u_progress;
void main() {
    FragColor = mix(texture(u_texture1, v_uv), texture(u_texture2, v_uv), u_progress);
}

