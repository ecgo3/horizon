#version 430 core

in vec4 vDiagnostics;
out vec4 fragColor;

vec3 blackbodyRamp(float shift) {
  float s = clamp(log2(max(shift, 0.001)), -2.0, 2.0);
  vec3 red = vec3(1.0, 0.22, 0.07);
  vec3 white = vec3(1.0, 0.86, 0.55);
  vec3 blue = vec3(0.35, 0.62, 1.0);
  return s < 0.0 ? mix(white, red, -s * 0.55) : mix(white, blue, s * 0.45);
}

void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) {
    discard;
  }

  float redshift = vDiagnostics.x;
  float dilation = vDiagnostics.y;
  float curvature = vDiagnostics.z;
  float alpha = smoothstep(1.0, 0.0, r2) * mix(0.18, 0.92, dilation);
  vec3 color = blackbodyRamp(redshift);
  color += vec3(1.0, 0.42, 0.08) * curvature * 0.08;
  fragColor = vec4(color * (0.65 + 0.35 * redshift), alpha);
}
