#include "headless/HeadlessRenderer.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <gbm.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

// EGL extension function pointers
static PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = nullptr;

HeadlessRenderer::HeadlessRenderer()
    : m_width(0),
      m_height(0),
      m_drmFd(-1),
      m_gbmDevice(nullptr),
      m_eglDisplay(EGL_NO_DISPLAY),
      m_eglContext(EGL_NO_CONTEXT),
      m_eglConfig(nullptr),
      m_fbo(0),
      m_colorTexture(0),
      m_depthRenderbuffer(0),
      m_initialized(false) {}

HeadlessRenderer::~HeadlessRenderer() {
  cleanup();
}

bool HeadlessRenderer::init(int width, int height) {
  m_width = width;
  m_height = height;

  // Try to open DRM render node
  const char* renderNodes[] = {
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/card0",
    nullptr
  };

  for (int i = 0; renderNodes[i] != nullptr; i++) {
    m_drmFd = open(renderNodes[i], O_RDWR);
    if (m_drmFd >= 0) {
      std::cout << "Opened DRM device: " << renderNodes[i] << std::endl;
      break;
    }
  }

  if (m_drmFd < 0) {
    std::cerr << "ERROR: Failed to open any DRM render node" << std::endl;
    return false;
  }

  // Create GBM device
  m_gbmDevice = gbm_create_device(m_drmFd);
  if (!m_gbmDevice) {
    std::cerr << "ERROR: Failed to create GBM device" << std::endl;
    cleanup();
    return false;
  }

  // Get EGL extension function
  eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)
      eglGetProcAddress("eglGetPlatformDisplayEXT");

  if (!eglGetPlatformDisplayEXT) {
    std::cerr << "ERROR: eglGetPlatformDisplayEXT not available" << std::endl;
    cleanup();
    return false;
  }

  // Create EGL display from GBM device
  m_eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_MESA, m_gbmDevice, nullptr);
  if (m_eglDisplay == EGL_NO_DISPLAY) {
    std::cerr << "ERROR: Failed to get EGL display" << std::endl;
    cleanup();
    return false;
  }

  // Initialize EGL
  EGLint major, minor;
  if (!eglInitialize(m_eglDisplay, &major, &minor)) {
    std::cerr << "ERROR: Failed to initialize EGL" << std::endl;
    cleanup();
    return false;
  }
  std::cout << "EGL version: " << major << "." << minor << std::endl;

  // Choose EGL config - for surfaceless rendering we don't need a specific surface type
  EGLint configAttribs[] = {
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_RED_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_ALPHA_SIZE, 8,
    EGL_DEPTH_SIZE, 24,
    EGL_NONE
  };

  EGLint numConfigs;
  if (!eglChooseConfig(m_eglDisplay, configAttribs, &m_eglConfig, 1, &numConfigs) || numConfigs == 0) {
    // Try more relaxed config
    std::cerr << "WARNING: First config choice failed, trying relaxed config..." << std::endl;
    EGLint relaxedAttribs[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_NONE
    };
    if (!eglChooseConfig(m_eglDisplay, relaxedAttribs, &m_eglConfig, 1, &numConfigs) || numConfigs == 0) {
      std::cerr << "ERROR: Failed to choose EGL config" << std::endl;
      cleanup();
      return false;
    }
  }

  // Bind OpenGL API
  if (!eglBindAPI(EGL_OPENGL_API)) {
    std::cerr << "ERROR: Failed to bind OpenGL API" << std::endl;
    cleanup();
    return false;
  }

  // Create EGL context (OpenGL 4.5 Core)
  EGLint contextAttribs[] = {
    EGL_CONTEXT_MAJOR_VERSION, 4,
    EGL_CONTEXT_MINOR_VERSION, 5,
    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
    EGL_NONE
  };

  m_eglContext = eglCreateContext(m_eglDisplay, m_eglConfig, EGL_NO_CONTEXT, contextAttribs);
  if (m_eglContext == EGL_NO_CONTEXT) {
    std::cerr << "ERROR: Failed to create EGL context" << std::endl;
    cleanup();
    return false;
  }

  // Make context current (surfaceless)
  if (!eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, m_eglContext)) {
    std::cerr << "ERROR: Failed to make EGL context current" << std::endl;
    cleanup();
    return false;
  }

  // Initialize GLAD with EGL
  if (!gladLoadGLLoader((GLADloadproc)eglGetProcAddress)) {
    std::cerr << "ERROR: Failed to initialize GLAD" << std::endl;
    cleanup();
    return false;
  }

  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
  std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;

  // Create FBO for offscreen rendering
  glGenFramebuffers(1, &m_fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

  // Color attachment (texture)
  glGenTextures(1, &m_colorTexture);
  glBindTexture(GL_TEXTURE_2D, m_colorTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_colorTexture, 0);

  // Depth attachment (renderbuffer)
  glGenRenderbuffers(1, &m_depthRenderbuffer);
  glBindRenderbuffer(GL_RENDERBUFFER, m_depthRenderbuffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_width, m_height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthRenderbuffer);

  // Check FBO completeness
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr << "ERROR: Framebuffer incomplete: " << status << std::endl;
    cleanup();
    return false;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  m_initialized = true;
  std::cout << "HeadlessRenderer initialized: " << m_width << "x" << m_height << std::endl;
  return true;
}

void HeadlessRenderer::readPixels(std::vector<uint8_t>& buffer) {
  if (!m_initialized) return;

  buffer.resize(m_width * m_height * 3);

  // Assume FBO is already bound by caller
  glReadBuffer(GL_COLOR_ATTACHMENT0);

  // Read as RGBA then convert to RGB (FFmpeg expects RGB24)
  std::vector<uint8_t> rgba(m_width * m_height * 4);
  glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

  // Convert RGBA to RGB and flip vertically (OpenGL origin is bottom-left)
  for (int y = 0; y < m_height; y++) {
    int srcY = m_height - 1 - y;  // Flip
    for (int x = 0; x < m_width; x++) {
      int srcIdx = (srcY * m_width + x) * 4;
      int dstIdx = (y * m_width + x) * 3;
      buffer[dstIdx + 0] = rgba[srcIdx + 0];  // R
      buffer[dstIdx + 1] = rgba[srcIdx + 1];  // G
      buffer[dstIdx + 2] = rgba[srcIdx + 2];  // B
    }
  }
}

void HeadlessRenderer::cleanup() {
  if (m_fbo) {
    glDeleteFramebuffers(1, &m_fbo);
    m_fbo = 0;
  }
  if (m_colorTexture) {
    glDeleteTextures(1, &m_colorTexture);
    m_colorTexture = 0;
  }
  if (m_depthRenderbuffer) {
    glDeleteRenderbuffers(1, &m_depthRenderbuffer);
    m_depthRenderbuffer = 0;
  }

  if (m_eglContext != EGL_NO_CONTEXT) {
    eglDestroyContext(m_eglDisplay, m_eglContext);
    m_eglContext = EGL_NO_CONTEXT;
  }

  if (m_eglDisplay != EGL_NO_DISPLAY) {
    eglTerminate(m_eglDisplay);
    m_eglDisplay = EGL_NO_DISPLAY;
  }

  if (m_gbmDevice) {
    gbm_device_destroy(m_gbmDevice);
    m_gbmDevice = nullptr;
  }

  if (m_drmFd >= 0) {
    close(m_drmFd);
    m_drmFd = -1;
  }

  m_initialized = false;
}
