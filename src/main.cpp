#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "horizon/Camera.hpp"
#include "horizon/CsvExporter.hpp"
#include "horizon/CudaSimulation.hpp"
#include "horizon/ShaderProgram.hpp"
#include "horizon/SimulationTypes.hpp"

namespace {

struct InputState {
  horizon::ArcballCamera* camera = nullptr;
  bool leftDown = false;
  bool rightDown = false;
  bool paused = false;
  bool showParticles = true;
  bool snapshotRequested = false;
  int overlayMode = 0;
  double lastX = 0.0;
  double lastY = 0.0;
};

std::filesystem::path shaderDirectory(const char* executablePath) {
  const std::filesystem::path exe = std::filesystem::absolute(executablePath).parent_path();
  if (std::filesystem::exists(exe / "shaders" / "lensing.frag")) {
    return exe / "shaders";
  }
  if (std::filesystem::exists(std::filesystem::current_path() / "shaders" / "lensing.frag")) {
    return std::filesystem::current_path() / "shaders";
  }
  throw std::runtime_error("Unable to locate shader directory.");
}

void glfwErrorCallback(int code, const char* description) {
  std::cerr << "GLFW error " << code << ": " << description << '\n';
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input != nullptr && input->camera != nullptr) {
    input->camera->resize(width, height);
  }
  glViewport(0, 0, width, height);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input == nullptr) {
    return;
  }
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    input->leftDown = action == GLFW_PRESS;
  }
  if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
    input->rightDown = action == GLFW_PRESS;
  }
  glfwGetCursorPos(window, &input->lastX, &input->lastY);
}

void cursorCallback(GLFWwindow* window, double x, double y) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input == nullptr || input->camera == nullptr) {
    return;
  }

  const double dx = x - input->lastX;
  const double dy = y - input->lastY;
  input->lastX = x;
  input->lastY = y;

  if (input->leftDown) {
    input->camera->orbit(dx, dy);
  } else if (input->rightDown) {
    input->camera->pan(dx, -dy);
  }
}

void scrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input != nullptr && input->camera != nullptr) {
    input->camera->zoom(yoffset);
  }
}

void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
  auto* input = static_cast<InputState*>(glfwGetWindowUserPointer(window));
  if (input == nullptr || action != GLFW_PRESS) {
    return;
  }

  if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  } else if (key == GLFW_KEY_SPACE) {
    input->paused = !input->paused;
  } else if (key == GLFW_KEY_P) {
    input->showParticles = !input->showParticles;
  } else if (key == GLFW_KEY_E) {
    input->snapshotRequested = true;
  } else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_5) {
    input->overlayMode = key - GLFW_KEY_0;
  }
}

void createParticleVao(GLuint& vao, GLuint& vbo) {
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(sizeof(horizon::RenderVertex) * horizon::kParticleCount),
               nullptr,
               GL_DYNAMIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(
      0, 4, GL_FLOAT, GL_FALSE, sizeof(horizon::RenderVertex), reinterpret_cast<void*>(0));

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1,
                        4,
                        GL_FLOAT,
                        GL_FALSE,
                        sizeof(horizon::RenderVertex),
                        reinterpret_cast<void*>(sizeof(float) * 4u));
  glBindVertexArray(0);
}

std::string titleFromDiagnostics(const horizon::Diagnostics& diagnostics, double frameMs) {
  const double active = static_cast<double>(diagnostics.activeCount == 0u ? 1u : diagnostics.activeCount);
  std::ostringstream title;
  title << "Horizon Schwarzschild Engine | "
        << std::fixed << std::setprecision(2) << frameMs << " ms | active "
        << diagnostics.activeCount << " | <dE/E> " << std::scientific << std::setprecision(2)
        << diagnostics.sumEnergyDrift / active << " | max norm drift " << diagnostics.maxNormDrift
        << " | captures " << diagnostics.capturedCount;
  return title.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    glfwSetErrorCallback(glfwErrorCallback);
    if (glfwInit() != GLFW_TRUE) {
      throw std::runtime_error("Failed to initialize GLFW.");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1600, 960, "Horizon Schwarzschild Engine", nullptr, nullptr);
    if (window == nullptr) {
      glfwTerminate();
      throw std::runtime_error("Failed to create OpenGL window.");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glewExperimental = GL_TRUE;
    const GLenum glewStatus = glewInit();
    glGetError();
    if (glewStatus != GLEW_OK) {
      throw std::runtime_error(reinterpret_cast<const char*>(glewGetErrorString(glewStatus)));
    }

    horizon::ArcballCamera camera;
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    camera.resize(width, height);

    InputState input{};
    input.camera = &camera;
    glfwSetWindowUserPointer(window, &input);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    const std::filesystem::path shaders = shaderDirectory(argc > 0 ? argv[0] : "");
    horizon::ShaderProgram lensingShader;
    horizon::ShaderProgram particleShader;
    lensingShader.load(shaders / "lensing.vert", shaders / "lensing.frag");
    particleShader.load(shaders / "particles.vert", shaders / "particles.frag");

    GLuint particleVao = 0;
    GLuint particleVbo = 0;
    GLuint screenVao = 0;
    glGenVertexArrays(1, &screenVao);
    createParticleVao(particleVao, particleVbo);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    horizon::SimulationParams params{};
    params.particleCount = horizon::kParticleCount;
    params.diskInnerRadius = 6.12;
    params.diskOuterRadius = 84.0;
    params.resetOuterRadius = 72.0;
    params.resetWidth = 14.0;
    params.baseProperTimeStep = 0.018;
    params.maxProperTimeStep = 0.045;
    params.minProperTimeStep = 0.0002;
    params.curvatureStepScale = 0.04;

    horizon::CsvExporter exporter("exports");

    {
      horizon::CudaSimulation simulation;
      simulation.initialize(particleVbo, params);

      auto previous = std::chrono::steady_clock::now();
      auto start = previous;
      std::uint64_t frame = 0;

      while (glfwWindowShouldClose(window) != GLFW_TRUE) {
        const auto now = std::chrono::steady_clock::now();
        const double frameSeconds = std::chrono::duration<double>(now - previous).count();
        const double wallSeconds = std::chrono::duration<double>(now - start).count();
        previous = now;

        params.coordinateTime += input.paused ? 0.0 : frameSeconds;
        if (!input.paused) {
          simulation.step(params, static_cast<std::uint32_t>(frame));
        }

        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        lensingShader.use();
        lensingShader.setVec3("uCameraPosition", camera.position());
        lensingShader.setVec3("uCameraForward", camera.forward());
        lensingShader.setVec3("uCameraRight", camera.right());
        lensingShader.setVec3("uCameraUp", camera.up());
        lensingShader.setDouble("uMass", params.geometricMass);
        lensingShader.setDouble("uDiskInner", params.diskInnerRadius);
        lensingShader.setDouble("uDiskOuter", params.diskOuterRadius);
        lensingShader.setDouble("uFovY", camera.verticalFovRadians());
        lensingShader.setDouble("uAspect", camera.aspect());
        lensingShader.setFloat("uTime", static_cast<float>(params.coordinateTime));
        lensingShader.setInt("uOverlayMode", input.overlayMode);
        glBindVertexArray(screenVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);

        if (input.showParticles) {
          particleShader.use();
          particleShader.setMat4("uViewProjection", camera.viewProjectionMatrix());
          particleShader.setFloat("uPointScale", 1550.0f);
          particleShader.setFloat("uTime", static_cast<float>(params.coordinateTime));
          glBindVertexArray(particleVao);
          glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(horizon::kParticleCount));
          glBindVertexArray(0);
        }

        if (frame % 30u == 0u) {
          const horizon::Diagnostics diagnostics = simulation.diagnostics();
          glfwSetWindowTitle(window, titleFromDiagnostics(diagnostics, frameSeconds * 1000.0).c_str());
          exporter.appendDiagnostics(frame, wallSeconds, diagnostics);
        }

        if (input.snapshotRequested) {
          input.snapshotRequested = false;
          exporter.writeParticleSnapshot(frame, simulation.downloadParticles(20000));
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
        ++frame;
      }
    }

    glDeleteBuffers(1, &particleVbo);
    glDeleteVertexArrays(1, &particleVao);
    glDeleteVertexArrays(1, &screenVao);
    glfwDestroyWindow(window);
    glfwTerminate();
  } catch (const std::exception& error) {
    std::cerr << "Fatal error: " << error.what() << '\n';
    glfwTerminate();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
