#version 330 core

in vec3 vNormal;
in vec2 vUV;

out vec4 FragColor;

uniform vec3 uColor;
uniform int uHasTexture;
uniform sampler2D uDiffuse;
uniform vec3 uLightDir;

void main()
{
    vec3 base = uColor;
    if (uHasTexture == 1) {
        base = texture(uDiffuse, vUV).rgb;
    }

    // Simple headlight: diffuse term plus a small ambient so back faces
    // are still visible.
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir);
    float ndotl = max(dot(N, L), 0.0);
    vec3 shaded = base * (0.25 + 0.75 * ndotl);

    FragColor = vec4(shaded, 1.0);
}
