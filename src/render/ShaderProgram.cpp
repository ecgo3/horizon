#include "horizon/ShaderProgram.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace horizon {

namespace {

std::string readTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open shader: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

GLuint compileShader(GLenum type, const std::string& source, const std::filesystem::path& path) {
  const GLuint shader = glCreateShader(type);
  const char* sourcePtr = source.c_str();
  glShaderSource(shader, 1, &sourcePtr, nullptr);
  glCompileShader(shader);

  GLint ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (ok == GL_TRUE) {
    return shader;
  }

  GLint length = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
  std::vector<char> log(static_cast<std::size_t>(length) + 1u);
  glGetShaderInfoLog(shader, length, nullptr, log.data());
  glDeleteShader(shader);
  throw std::runtime_error("Shader compile failed for " + path.string() + ":\n" + log.data());
}

}  // namespace

ShaderProgram::~ShaderProgram() {
  if (program_ != 0) {
    glDeleteProgram(program_);
  }
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept : program_(std::exchange(other.program_, 0)) {}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
  if (this != &other) {
    if (program_ != 0) {
      glDeleteProgram(program_);
    }
    program_ = std::exchange(other.program_, 0);
  }
  return *this;
}

void ShaderProgram::load(const std::filesystem::path& vertexPath, const std::filesystem::path& fragmentPath) {
  const std::string vertexSource = readTextFile(vertexPath);
  const std::string fragmentSource = readTextFile(fragmentPath);
  const GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexSource, vertexPath);
  const GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource, fragmentPath);

  const GLuint linked = glCreateProgram();
  glAttachShader(linked, vertex);
  glAttachShader(linked, fragment);
  glLinkProgram(linked);
  glDeleteShader(vertex);
  glDeleteShader(fragment);

  GLint ok = GL_FALSE;
  glGetProgramiv(linked, GL_LINK_STATUS, &ok);
  if (ok != GL_TRUE) {
    GLint length = 0;
    glGetProgramiv(linked, GL_INFO_LOG_LENGTH, &length);
    std::vector<char> log(static_cast<std::size_t>(length) + 1u);
    glGetProgramInfoLog(linked, length, nullptr, log.data());
    glDeleteProgram(linked);
    throw std::runtime_error("Shader link failed:\n" + std::string(log.data()));
  }

  if (program_ != 0) {
    glDeleteProgram(program_);
  }
  program_ = linked;
}

void ShaderProgram::use() const {
  glUseProgram(program_);
}

void ShaderProgram::setInt(const char* name, int value) const {
  glUniform1i(glGetUniformLocation(program_, name), value);
}

void ShaderProgram::setFloat(const char* name, float value) const {
  glUniform1f(glGetUniformLocation(program_, name), value);
}

void ShaderProgram::setDouble(const char* name, double value) const {
  glUniform1d(glGetUniformLocation(program_, name), value);
}

void ShaderProgram::setVec3(const char* name, Vec3 value) const {
  glUniform3d(glGetUniformLocation(program_, name), value.x, value.y, value.z);
}

void ShaderProgram::setMat4(const char* name, const Mat4& value) const {
  glUniformMatrix4fv(glGetUniformLocation(program_, name), 1, GL_FALSE, value.data());
}

}  // namespace horizon
