#pragma once

#include <memory>
#include <string>

struct GLFWwindow;

class Lenia3DCUDA;
class VolumeRenderer;
class Camera3D;
class BloomRenderer;

class ChronosApp {
 public:
  ChronosApp();
  ~ChronosApp();

  bool init(int width, int height, const std::string& title);
  void run();
  void shutdown();

 private:
  void processInput();
  void update(float deltaTime);
  void render();
  void renderUI();

  // Window
  GLFWwindow* _window = nullptr;
  int _windowWidth = 1920;
  int _windowHeight = 1080;

  // Core components
  std::unique_ptr<Lenia3DCUDA> _simulation;
  std::unique_ptr<VolumeRenderer> _renderer;
  std::unique_ptr<Camera3D> _camera;
  std::unique_ptr<BloomRenderer> _bloomRenderer;

  // Scene FBO for post-processing
  unsigned int _sceneFbo = 0;
  unsigned int _sceneTexture = 0;
  unsigned int _sceneDepth = 0;

  // Simulation parameters
  int _gridSize = 64;  // 64x64x64 voxel grid
  float _volumeDensity = 2.0f;
  bool _paused = false;
  float _simulationSpeed = 1.0f;

  // Bloom settings
  bool _enableBloom = true;
  float _bloomIntensity = 1.2f;
  float _bloomThreshold = 0.15f;

  // Timing
  float _lastFrameTime = 0.0f;
  float _deltaTime = 0.0f;
  float _simAccumulator = 0.0f;

  // Mouse state for camera
  bool _firstMouse = true;
  float _lastMouseX = 0.0f;
  float _lastMouseY = 0.0f;
  bool _mouseRightPressed = false;

  void createSceneFBO(int width, int height);
  void destroySceneFBO();
};
