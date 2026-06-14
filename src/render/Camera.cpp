#include "horizon/Camera.hpp"

#include <algorithm>
#include <cmath>

namespace horizon {

void ArcballCamera::resize(int width, int height) {
  aspect_ = height > 0 ? static_cast<double>(width) / static_cast<double>(height) : 1.0;
}

void ArcballCamera::orbit(double deltaX, double deltaY) {
  yaw_ += deltaX * 0.0065;
  pitch_ = std::clamp(pitch_ + deltaY * 0.0065, -1.45, 1.45);
}

void ArcballCamera::pan(double deltaX, double deltaY) {
  const double scale = distance_ * 0.0016;
  target_ = target_ - right() * (deltaX * scale) + up() * (deltaY * scale);
}

void ArcballCamera::zoom(double scrollDelta) {
  distance_ *= std::exp(-scrollDelta * 0.115);
  distance_ = std::clamp(distance_, 2.025, 1500.0);
  nearPlane_ = std::max(1.0e-4, distance_ * 1.0e-4);
}

Vec3 ArcballCamera::position() const {
  const double cp = std::cos(pitch_);
  const Vec3 offset{
      distance_ * cp * std::sin(yaw_),
      distance_ * std::sin(pitch_),
      distance_ * cp * std::cos(yaw_)};
  return target_ + offset;
}

Vec3 ArcballCamera::forward() const {
  return normalize(target_ - position());
}

Vec3 ArcballCamera::right() const {
  return normalize(cross(forward(), {0.0, 1.0, 0.0}));
}

Vec3 ArcballCamera::up() const {
  return normalize(cross(right(), forward()));
}

Mat4 ArcballCamera::viewMatrix() const {
  return lookAt(position(), target_, {0.0, 1.0, 0.0});
}

Mat4 ArcballCamera::projectionMatrix() const {
  return perspective(verticalFovRadians_, aspect_, nearPlane_, farPlane_);
}

Mat4 ArcballCamera::viewProjectionMatrix() const {
  return multiply(projectionMatrix(), viewMatrix());
}

}  // namespace horizon
