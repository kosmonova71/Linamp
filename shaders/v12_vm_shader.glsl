#version 330 core

// V12 GPU VM - Bytecode interpreter shader
// Executes Milkdrop preset bytecode on GPU (OpenGL 3.3 compatible)

// Input vertex attributes
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;

// Output to fragment shader
out vec2 v_texcoord;

// Uniforms
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_audio;
uniform float u_frame;
uniform int u_per_frame_pc;
uniform int u_per_pixel_pc;

// Bytecode and data buffers (UBO for OpenGL 3.3 compatibility)
#define MAX_BYTECODE 1024
#define MAX_CONSTANTS 256
#define MAX_VARIABLES 128

uniform int u_bytecode[MAX_BYTECODE];
uniform float u_constants[MAX_CONSTANTS];
uniform float u_vars[MAX_VARIABLES];

// Opcodes
const int OP_PUSH_CONST = 1;
const int OP_PUSH_VAR = 2;
const int OP_STORE = 3;
const int OP_DUP = 4;
const int OP_SWAP = 5;
const int OP_POP = 6;
const int OP_ADD = 10;
const int OP_SUB = 11;
const int OP_MUL = 12;
const int OP_DIV = 13;
const int OP_MOD = 14;
const int OP_POW = 15;
const int OP_NEG = 16;
const int OP_GT = 20;
const int OP_LT = 21;
const int OP_GE = 22;
const int OP_LE = 23;
const int OP_EQ = 24;
const int OP_NE = 25;
const int OP_AND = 30;
const int OP_OR = 31;
const int OP_NOT = 32;
const int OP_TERNARY = 40;
const int OP_SIN = 50;
const int OP_COS = 51;
const int OP_TAN = 52;
const int OP_ASIN = 53;
const int OP_ACOS = 54;
const int OP_ATAN = 55;
const int OP_SQRT = 56;
const int OP_ABS = 57;
const int OP_FLOOR = 58;
const int OP_CEIL = 59;
const int OP_ROUND = 60;
const int OP_SIGN = 61;
const int OP_EXP = 62;
const int OP_LOG = 63;
const int OP_LOG10 = 64;
const int OP_MIN = 65;
const int OP_MAX = 66;
const int OP_RAND = 67;
const int OP_NOOP = 255;
const int OP_HALT = 254;

// Stack for VM execution
float stack[256];
int sp = 0;

// Helper: push value onto stack
void push(float val) {
    stack[sp++] = val;
}

// Helper: pop value from stack
float pop() {
    return stack[--sp];
}

// Helper: peek at top of stack
float peek() {
    return stack[sp - 1];
}

// VM execution engine
float run_vm(int pc_start) {
    int pc = pc_start;
    int max_iterations = 512;  // Prevent infinite loops
    
    for (int i = 0; i < max_iterations && pc < MAX_BYTECODE; i++) {
        int op = u_bytecode[pc++];
        
        switch (op) {
            // Stack operations
            case OP_PUSH_CONST: {
                int const_idx = u_bytecode[pc++];
                push(u_constants[const_idx]);
                break;
            }
            case OP_PUSH_VAR: {
                int var_idx = u_bytecode[pc++];
                push(u_vars[var_idx]);
                break;
            }
            case OP_STORE: {
                int var_idx = u_bytecode[pc++];
                u_vars[var_idx] = pop();
                break;
            }
            case OP_DUP: {
                push(peek());
                break;
            }
            case OP_SWAP: {
                float a = pop();
                float b = pop();
                push(a);
                push(b);
                break;
            }
            case OP_POP: {
                pop();
                break;
            }
            
            // Arithmetic
            case OP_ADD: {
                float b = pop();
                float a = pop();
                push(a + b);
                break;
            }
            case OP_SUB: {
                float b = pop();
                float a = pop();
                push(a - b);
                break;
            }
            case OP_MUL: {
                float b = pop();
                float a = pop();
                push(a * b);
                break;
            }
            case OP_DIV: {
                float b = pop();
                float a = pop();
                push(a / b);
                break;
            }
            case OP_MOD: {
                float b = pop();
                float a = pop();
                push(mod(a, b));
                break;
            }
            case OP_POW: {
                float b = pop();
                float a = pop();
                push(pow(a, b));
                break;
            }
            case OP_NEG: {
                float a = pop();
                push(-a);
                break;
            }
            
            // Comparison
            case OP_GT: {
                float b = pop();
                float a = pop();
                push(float(a > b));
                break;
            }
            case OP_LT: {
                float b = pop();
                float a = pop();
                push(float(a < b));
                break;
            }
            case OP_GE: {
                float b = pop();
                float a = pop();
                push(float(a >= b));
                break;
            }
            case OP_LE: {
                float b = pop();
                float a = pop();
                push(float(a <= b));
                break;
            }
            case OP_EQ: {
                float b = pop();
                float a = pop();
                push(float(a == b));
                break;
            }
            case OP_NE: {
                float b = pop();
                float a = pop();
                push(float(a != b));
                break;
            }
            
            // Logical
            case OP_AND: {
                float b = pop();
                float a = pop();
                push(float(bool(a) && bool(b)));
                break;
            }
            case OP_OR: {
                float b = pop();
                float a = pop();
                push(float(bool(a) || bool(b)));
                break;
            }
            case OP_NOT: {
                float a = pop();
                push(float(!bool(a)));
                break;
            }
            
            // Ternary
            case OP_TERNARY: {
                float c = pop();
                float b = pop();
                float a = pop();
                push(a > 0.5 ? b : c);
                break;
            }
            
            // Math functions
            case OP_SIN: {
                float a = pop();
                push(sin(a));
                break;
            }
            case OP_COS: {
                float a = pop();
                push(cos(a));
                break;
            }
            case OP_TAN: {
                float a = pop();
                push(tan(a));
                break;
            }
            case OP_ASIN: {
                float a = pop();
                push(asin(a));
                break;
            }
            case OP_ACOS: {
                float a = pop();
                push(acos(a));
                break;
            }
            case OP_ATAN: {
                float a = pop();
                push(atan(a));
                break;
            }
            case OP_SQRT: {
                float a = pop();
                push(sqrt(a));
                break;
            }
            case OP_ABS: {
                float a = pop();
                push(abs(a));
                break;
            }
            case OP_FLOOR: {
                float a = pop();
                push(floor(a));
                break;
            }
            case OP_CEIL: {
                float a = pop();
                push(ceil(a));
                break;
            }
            case OP_ROUND: {
                float a = pop();
                push(round(a));
                break;
            }
            case OP_SIGN: {
                float a = pop();
                push(sign(a));
                break;
            }
            case OP_EXP: {
                float a = pop();
                push(exp(a));
                break;
            }
            case OP_LOG: {
                float a = pop();
                push(log(a));
                break;
            }
            case OP_LOG10: {
                float a = pop();
                push(log10(a));
                break;
            }
            case OP_MIN: {
                float b = pop();
                float a = pop();
                push(min(a, b));
                break;
            }
            case OP_MAX: {
                float b = pop();
                float a = pop();
                push(max(a, b));
                break;
            }
            case OP_RAND: {
                float a = pop();
                push(fract(sin(a * 12.9898) * 43758.5453));
                break;
            }
            
            // Control
            case OP_NOOP:
                break;
            case OP_HALT:
                return peek();
            
            default:
                // Unknown opcode - halt
                return peek();
        }
    }
    
    return peek();
}

// Initialize standard variables
void init_vars() {
    // Coordinates
    u_vars[0] = 0.0;  // x
    u_vars[1] = 0.0;  // y
    
    // Time
    u_vars[2] = u_time;
    u_vars[3] = u_frame;
    
    // Audio
    u_vars[4] = u_audio.x;  // bass
    u_vars[5] = u_audio.y;  // mid
    u_vars[6] = u_audio.x + u_audio.y;  // treb
    
    // Audio attenuation
    u_vars[7] = u_audio.x;  // bass_att
    u_vars[8] = u_audio.y;  // mid_att
    u_vars[9] = u_audio.x + u_audio.y;  // treb_att
    
    // Radial
    u_vars[10] = 0.0;  // rad (computed in shader)
    
    // Warp
    u_vars[11] = 1.0;  // zoom
    u_vars[12] = 0.0;  // rot
    
    // Displacement
    u_vars[13] = 0.0;  // dx
    u_vars[14] = 0.0;  // dy
    
    // Threshold
    u_vars[15] = 1.3;  // thresh
    u_vars[16] = 0.0;  // dx_r
    u_vars[17] = 0.0;  // dy_r
    
    // Aspect
    u_vars[18] = 1.0;  // aspectx
    u_vars[19] = 1.0;  // aspecty
    
    // FPS
    u_vars[20] = 60.0;  // fps
    
    // q1-q32 initialized to 0.0
    for (int i = 21; i < 53; i++) {
        u_vars[i] = 0.0;
    }
    
    // tg1-tg3
    u_vars[53] = 0.0;  // tg1
    u_vars[54] = 0.0;  // tg2
    u_vars[55] = 0.0;  // tg3
    
    // wave colors
    u_vars[56] = 0.0;  // wave_r
    u_vars[57] = 0.0;  // wave_g
    u_vars[58] = 0.0;  // wave_b
}

void main() {
    // Pass through texcoord
    v_texcoord = a_texcoord;
    
    // Initialize variables
    init_vars();
    
    // Set x, y from texcoord
    u_vars[0] = v_texcoord.x;
    u_vars[1] = v_texcoord.y;
    
    // Compute radial distance
    u_vars[10] = sqrt((u_vars[0] - 0.5) * (u_vars[0] - 0.5) + (u_vars[1] - 0.5) * (u_vars[1] - 0.5));
    
    // Execute per-frame bytecode (once per frame)
    // Note: In actual implementation, this would be done in compute shader or separate pass
    // For now, we execute it per-pixel for simplicity
    if (u_per_frame_pc >= 0) {
        run_vm(u_per_frame_pc);
    }
    
    // Execute per-pixel bytecode
    if (u_per_pixel_pc >= 0) {
        run_vm(u_per_pixel_pc);
    }
    
    // Get final warp values
    float dx = u_vars[13];
    float dy = u_vars[14];
    float zoom = u_vars[11];
    
    // Apply warp
    vec2 uv = v_texcoord + vec2(dx, dy);
    
    // Generate color from warped coordinates
    vec3 color = vec3(0.0);
    float warp_intensity = length(vec2(dx, dy)) * 10.0;
    color.r = sin(uv.x * 10.0 + u_time) * 0.5 + 0.5 + u_audio.x * warp_intensity;
    color.g = cos(uv.y * 10.0 + u_time) * 0.5 + 0.5 + u_audio.y * warp_intensity;
    color.b = sin(uv.x * uv.y * 20.0 + u_time) * 0.5 + 0.5 + (u_audio.x + u_audio.y) * warp_intensity;
    color *= zoom * (1.0 + u_audio.x);
    
    gl_Position = vec4(a_position, 0.0, 1.0);
}
