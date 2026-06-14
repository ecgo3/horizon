#pragma once

#include <GL/glew.h>

#include <filesystem>
#include <string>

#include "horizon/Math.hpp"

namespace horizon {

class ShaderProgram {
 public:
  ShaderProgram() = default;
  ~ShaderProgram();

  ShaderProgram(const ShaderProgram&) = delete;
  ShaderProgram& operator=(const ShaderProgram&) = delete;

  ShaderProgram(ShaderProgram&& other) noexcept;
  ShaderProgram& operator=(ShaderProgram&& other) noexcept;

  void load(const std::filesystem::path& vertexPath, const std::filesystem::path& fragmentPath);
  void use() const;

  [[nodiscard]] GLuint id() const {
    return program_;
  }

  void setInt(const char* name, int value) const;
  void setFloat(const char* name, float value) const;
  void setDouble(const char* name, double value) const;
  void setVec3(const char* name, Vec3 value) const;
  void setMat4(const char* name, const Mat4& value) const;

 private:
  GLuint program_ = 0;
};

}  // namespace horizon
