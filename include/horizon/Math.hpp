#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace horizon {

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Mat4 {
  std::array<float, 16> m{};

  [[nodiscard]] const float* data() const {
    return m.data();
  }
};

inline Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(Vec3 a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}

inline Vec3 operator/(Vec3 a, double s) {
  return {a.x / s, a.y / s, a.z / s};
}

inline double dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double length(Vec3 v) {
  return std::sqrt(dot(v, v));
}

inline Vec3 normalize(Vec3 v) {
  const double len = length(v);
  return len > 0.0 ? v / len : Vec3{0.0, 0.0, 0.0};
}

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
  Mat4 out{};
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      float value = 0.0f;
      for (int k = 0; k < 4; ++k) {
        value += a.m[k * 4 + row] * b.m[col * 4 + k];
      }
      out.m[col * 4 + row] = value;
    }
  }
  return out;
}

inline Mat4 perspective(double verticalFovRadians, double aspect, double nearPlane, double farPlane) {
  const double f = 1.0 / std::tan(verticalFovRadians * 0.5);
  Mat4 out{};
  out.m[0] = static_cast<float>(f / aspect);
  out.m[5] = static_cast<float>(f);
  out.m[10] = static_cast<float>((farPlane + nearPlane) / (nearPlane - farPlane));
  out.m[11] = -1.0f;
  out.m[14] = static_cast<float>((2.0 * farPlane * nearPlane) / (nearPlane - farPlane));
  return out;
}

inline Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 upHint) {
  const Vec3 f = normalize(center - eye);
  const Vec3 s = normalize(cross(f, normalize(upHint)));
  const Vec3 u = cross(s, f);

  Mat4 out{};
  out.m[0] = static_cast<float>(s.x);
  out.m[4] = static_cast<float>(s.y);
  out.m[8] = static_cast<float>(s.z);
  out.m[1] = static_cast<float>(u.x);
  out.m[5] = static_cast<float>(u.y);
  out.m[9] = static_cast<float>(u.z);
  out.m[2] = static_cast<float>(-f.x);
  out.m[6] = static_cast<float>(-f.y);
  out.m[10] = static_cast<float>(-f.z);
  out.m[12] = static_cast<float>(-dot(s, eye));
  out.m[13] = static_cast<float>(-dot(u, eye));
  out.m[14] = static_cast<float>(dot(f, eye));
  out.m[15] = 1.0f;
  return out;
}

inline double radians(double degrees) {
  return degrees * 3.141592653589793238462643383279502884 / 180.0;
}

}  // namespace horizon
