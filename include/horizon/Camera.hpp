#pragma once

#include "horizon/Math.hpp"

namespace horizon {

class ArcballCamera {
 public:
  void resize(int width, int height);
  void orbit(double deltaX, double deltaY);
  void pan(double deltaX, double deltaY);
  void zoom(double scrollDelta);

  [[nodiscard]] Mat4 viewMatrix() const;
  [[nodiscard]] Mat4 projectionMatrix() const;
  [[nodiscard]] Mat4 viewProjectionMatrix() const;

  [[nodiscard]] Vec3 position() const;
  [[nodiscard]] Vec3 target() const {
    return target_;
  }
  [[nodiscard]] Vec3 forward() const;
  [[nodiscard]] Vec3 right() const;
  [[nodiscard]] Vec3 up() const;
  [[nodiscard]] double aspect() const {
    return aspect_;
  }
  [[nodiscard]] double verticalFovRadians() const {
    return verticalFovRadians_;
  }

 private:
  Vec3 target_{0.0, 0.0, 0.0};
  double distance_ = 46.0;
  double yaw_ = 0.72;
  double pitch_ = 0.32;
  double aspect_ = 16.0 / 9.0;
  double verticalFovRadians_ = radians(58.0);
  double nearPlane_ = 0.015;
  double farPlane_ = 5000.0;
};

}  // namespace horizon
