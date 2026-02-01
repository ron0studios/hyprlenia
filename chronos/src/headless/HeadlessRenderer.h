#ifndef CHRONOS_HEADLESS_RENDERER_H
#define CHRONOS_HEADLESS_RENDERER_H

#include <glad/glad.h>
#include <cstdint>
#include <vector>

// Forward declarations for EGL/GBM types
typedef void* EGLDisplay;
typedef void* EGLContext;
typedef void* EGLConfig;
struct gbm_device;

class HeadlessRenderer {
 public:
  HeadlessRenderer();
  ~HeadlessRenderer();

  // Initialize EGL context with GBM backend and create FBO
  bool init(int width, int height);

  // Read pixels from FBO into RGB24 buffer (flipped for video)
  void readPixels(std::vector<uint8_t>& buffer);

  // Get FBO for rendering
  GLuint fbo() const { return m_fbo; }

  // Get dimensions
  int width() const { return m_width; }
  int height() const { return m_height; }

  // Cleanup resources
  void cleanup();

 private:
  int m_width;
  int m_height;

  // DRM/GBM
  int m_drmFd;
  gbm_device* m_gbmDevice;

  // EGL
  EGLDisplay m_eglDisplay;
  EGLContext m_eglContext;
  EGLConfig m_eglConfig;

  // OpenGL FBO
  GLuint m_fbo;
  GLuint m_colorTexture;
  GLuint m_depthRenderbuffer;

  bool m_initialized;
};

#endif  // CHRONOS_HEADLESS_RENDERER_H
