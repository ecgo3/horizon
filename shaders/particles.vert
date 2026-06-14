#version 430 core

layout(location = 0) in vec4 aPositionSize;
layout(location = 1) in vec4 aDiagnostics;

uniform mat4 uViewProjection;
uniform float uPointScale;
uniform float uTime;

out vec4 vDiagnostics;

void main() {
  vec4 clip = uViewProjection * vec4(aPositionSize.xyz, 1.0);
  gl_Position = clip;
  float perspective = max(0.001, clip.w);
  gl_PointSize = clamp(aPositionSize.w * uPointScale / perspective, 0.0, 18.0);
  vDiagnostics = aDiagnostics;
}
