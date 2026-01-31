#include "ChronosApp.h"
// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
// clang-format on
#include <iostream>

#include "BloomRenderer.h"
#include "Camera3D.h"
#include "Lenia3DCUDA.cuh"
#include "VolumeRenderer.h"

ChronosApp::ChronosApp() = default;
ChronosApp::~ChronosApp() = default;

bool ChronosApp::init(int width, int height, const std::string& title) {
  _windowWidth = width;
  _windowHeight = height;

  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);

  _window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
  if (!_window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(_window);
  glfwSwapInterval(1);  // VSync

  // Initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return false;
  }

  // OpenGL settings
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_MULTISAMPLE);

  // Initialize ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 8.0f;
  style.FrameRounding = 4.0f;
  style.Colors[ImGuiCol_WindowBg].w = 0.9f;

  ImGui_ImplGlfw_InitForOpenGL(_window, true);
  ImGui_ImplOpenGL3_Init("#version 450");

  // Initialize 3D Lenia simulation
  _simulation = std::make_unique<Lenia3DCUDA>();
  _simulation->init(_gridSize, _gridSize, _gridSize);

  // Initialize volume renderer
  _renderer = std::make_unique<VolumeRenderer>();
  _renderer->init();

  // Initialize camera
  _camera = std::make_unique<Camera3D>();
  _camera->setPosition(glm::vec3(0.0f, 0.0f, 2.0f));

  // Initialize bloom renderer
  _bloomRenderer = std::make_unique<BloomRenderer>();
  _bloomRenderer->init(_windowWidth, _windowHeight);

  // Create scene FBO for post-processing
  createSceneFBO(_windowWidth, _windowHeight);

  std::cout << "Lenia 3D CUDA initialized!" << std::endl;
  std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

  return true;
}

void ChronosApp::run() {
  while (!glfwWindowShouldClose(_window)) {
    float currentFrame = static_cast<float>(glfwGetTime());
    _deltaTime = currentFrame - _lastFrameTime;
    _lastFrameTime = currentFrame;

    glfwPollEvents();
    processInput();
    update(_deltaTime);
    render();
    renderUI();

    glfwSwapBuffers(_window);
  }
}

void ChronosApp::shutdown() {
  destroySceneFBO();
  _bloomRenderer->shutdown();
  _renderer->shutdown();
  _simulation->shutdown();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(_window);
  glfwTerminate();
}

void ChronosApp::processInput() {
  if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(_window, true);
  }

  // Pause toggle
  static bool spaceWasPressed = false;
  bool spacePressed = glfwGetKey(_window, GLFW_KEY_SPACE) == GLFW_PRESS;
  if (spacePressed && !spaceWasPressed) {
    _paused = !_paused;
  }
  spaceWasPressed = spacePressed;

  // Reset with R
  static bool rWasPressed = false;
  bool rPressed = glfwGetKey(_window, GLFW_KEY_R) == GLFW_PRESS;
  if (rPressed && !rWasPressed) {
    _simulation->reset();
  }
  rWasPressed = rPressed;

  // Camera movement with WASD
  float cameraSpeed = 2.0f * _deltaTime;
  if (glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS)
    _camera->moveForward(cameraSpeed);
  if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS)
    _camera->moveForward(-cameraSpeed);
  if (glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS)
    _camera->moveRight(-cameraSpeed);
  if (glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS)
    _camera->moveRight(cameraSpeed);
  if (glfwGetKey(_window, GLFW_KEY_Q) == GLFW_PRESS)
    _camera->moveUp(-cameraSpeed);
  if (glfwGetKey(_window, GLFW_KEY_E) == GLFW_PRESS)
    _camera->moveUp(cameraSpeed);

  // Mouse look with right-click
  double mouseX, mouseY;
  glfwGetCursorPos(_window, &mouseX, &mouseY);

  bool rightPressed =
      glfwGetMouseButton(_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

  if (rightPressed && !_mouseRightPressed) {
    _firstMouse = true;
  }
  _mouseRightPressed = rightPressed;

  if (_mouseRightPressed) {
    if (_firstMouse) {
      _lastMouseX = static_cast<float>(mouseX);
      _lastMouseY = static_cast<float>(mouseY);
      _firstMouse = false;
    }

    float xOffset = static_cast<float>(mouseX) - _lastMouseX;
    float yOffset = _lastMouseY - static_cast<float>(mouseY);
    _lastMouseX = static_cast<float>(mouseX);
    _lastMouseY = static_cast<float>(mouseY);

    _camera->rotate(xOffset * 0.1f, yOffset * 0.1f);
  }

  // Add blob on left click
  if (glfwGetMouseButton(_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
    float cx = _gridSize / 2.0f;
    float cy = _gridSize / 2.0f;
    float cz = _gridSize / 2.0f;
    _simulation->addBlob(cx, cy, cz, _gridSize / 8.0f);
  }
}

void ChronosApp::update(float deltaTime) {
  if (!_paused) {
    _simAccumulator += deltaTime * _simulationSpeed;
    float simStep = 1.0f / 30.0f;
    
    while (_simAccumulator >= simStep) {
      _simulation->update();
      _simAccumulator -= simStep;
    }
  }
}

void ChronosApp::render() {
  int width, height;
  glfwGetFramebufferSize(_window, &width, &height);
  float aspect = static_cast<float>(width) / static_cast<float>(height);

  if (width != _windowWidth || height != _windowHeight) {
    _windowWidth = width;
    _windowHeight = height;
    destroySceneFBO();
    createSceneFBO(width, height);
    _bloomRenderer->resize(width, height);
  }

  if (_enableBloom) {
    glBindFramebuffer(GL_FRAMEBUFFER, _sceneFbo);
    glViewport(0, 0, width, height);
  }

  glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const auto& grid = _simulation->getGrid();
  _renderer->updateTexture(grid, _simulation->getSizeX(), 
                           _simulation->getSizeY(), _simulation->getSizeZ());
  _renderer->render(_camera->getViewMatrix(),
                    _camera->getProjectionMatrix(aspect),
                    _camera->getPosition(),
                    _volumeDensity);

  if (_enableBloom) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    _bloomRenderer->apply(_sceneTexture, _bloomIntensity, _bloomThreshold);

    unsigned int bloomFbo = 0;
    glGenFramebuffers(1, &bloomFbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, bloomFbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, _bloomRenderer->getOutputTexture(), 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glDeleteFramebuffers(1, &bloomFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
}

void ChronosApp::renderUI() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(350, 450), ImGuiCond_FirstUseEver);

  ImGui::Begin("Lenia 3D", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

  ImGui::Text("EMERGENT LIFE IN 3D");
  ImGui::Separator();

  ImGui::Text("Simulation");
  if (ImGui::Button(_paused ? "Resume (Space)" : "Pause (Space)")) {
    _paused = !_paused;
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset (R)")) {
    _simulation->reset();
  }

  ImGui::SliderFloat("Speed", &_simulationSpeed, 0.1f, 5.0f, "%.1f");
  
  ImGui::Separator();
  ImGui::Text("Lenia Parameters");
  ImGui::SliderFloat("Kernel Radius", &_simulation->species.R, 4.0f, 16.0f, "%.0f");
  ImGui::SliderFloat("Time Scale", &_simulation->species.T, 1.0f, 20.0f, "%.0f");
  
  ImGui::Text("Growth Function:");
  ImGui::SliderFloat("Mu 1", &_simulation->species.mu[0], 0.05f, 0.5f, "%.3f");
  ImGui::SliderFloat("Mu 2", &_simulation->species.mu[1], 0.05f, 0.5f, "%.3f");
  ImGui::SliderFloat("Mu 3", &_simulation->species.mu[2], 0.05f, 0.5f, "%.3f");
  ImGui::SliderFloat("Sigma 1", &_simulation->species.sigma[0], 0.01f, 0.2f, "%.3f");
  ImGui::SliderFloat("Sigma 2", &_simulation->species.sigma[1], 0.01f, 0.2f, "%.3f");
  ImGui::SliderFloat("Sigma 3", &_simulation->species.sigma[2], 0.01f, 0.2f, "%.3f");

  ImGui::Separator();
  ImGui::Text("Rendering");
  ImGui::SliderFloat("Volume Density", &_volumeDensity, 0.5f, 5.0f, "%.1f");

  ImGui::Separator();
  ImGui::Text("Post-Processing");
  ImGui::Checkbox("Enable Bloom", &_enableBloom);
  if (_enableBloom) {
    ImGui::SliderFloat("Bloom Intensity", &_bloomIntensity, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Bloom Threshold", &_bloomThreshold, 0.0f, 1.0f, "%.2f");
  }

  ImGui::Separator();
  ImGui::Text("Statistics");
  ImGui::Text("Grid: %dx%dx%d", _simulation->getSizeX(), 
              _simulation->getSizeY(), _simulation->getSizeZ());
  ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

  ImGui::Separator();
  ImGui::Text("Controls");
  ImGui::BulletText("WASD - Move camera");
  ImGui::BulletText("Q/E - Up/Down");
  ImGui::BulletText("Right-click + drag - Look");
  ImGui::BulletText("Left-click - Add blob");
  ImGui::BulletText("Space - Pause/Resume");
  ImGui::BulletText("R - Reset simulation");

  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ChronosApp::createSceneFBO(int width, int height) {
  glGenTextures(1, &_sceneTexture);
  glBindTexture(GL_TEXTURE_2D, _sceneTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA,
               GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glGenRenderbuffers(1, &_sceneDepth);
  glBindRenderbuffer(GL_RENDERBUFFER, _sceneDepth);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);

  glGenFramebuffers(1, &_sceneFbo);
  glBindFramebuffer(GL_FRAMEBUFFER, _sceneFbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         _sceneTexture, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, _sceneDepth);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ChronosApp::destroySceneFBO() {
  if (_sceneFbo) {
    glDeleteFramebuffers(1, &_sceneFbo);
    _sceneFbo = 0;
  }
  if (_sceneTexture) {
    glDeleteTextures(1, &_sceneTexture);
    _sceneTexture = 0;
  }
  if (_sceneDepth) {
    glDeleteRenderbuffers(1, &_sceneDepth);
    _sceneDepth = 0;
  }
}
