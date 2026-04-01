#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D currentFrameTex;
uniform sampler2D lastFrameTex;
uniform int frameCount;
uniform int cameraMoving;

#define MAX_FRAMES 5000.0f

void main()
{
    vec3 current = texture(currentFrameTex, TexCoords).rgb;
    vec3 accumulated = texture(lastFrameTex, TexCoords).rgb;

    vec3 col;
    if (cameraMoving == 1) {
        col = mix(accumulated, current, 0.2); // 20% new, 80% old during motion
    } else {
        col = mix(current, accumulated, min(frameCount/MAX_FRAMES, 1.0));
    }
    FragColor = vec4(col, 1.0);
}
