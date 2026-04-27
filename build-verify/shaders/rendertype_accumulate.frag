#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D currentFrameTex;
uniform sampler2D lastFrameTex;
uniform int frameCount;
uniform int cameraMoving;

void main()
{
    vec3 current = texture(currentFrameTex, TexCoords).rgb;
    vec3 accumulated = texture(lastFrameTex, TexCoords).rgb;

    vec3 col;
    if (cameraMoving == 1) {
        col = mix(accumulated, current, 0.2); // 20% new, 80% old during motion
    } else {
        col = mix(accumulated, current, 1.0 / float(frameCount));
    }
    FragColor = vec4(col, 1.0);
}
