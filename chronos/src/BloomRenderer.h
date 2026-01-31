#pragma once

/**
 * BloomRenderer - Adds a bloom/glow post-processing effect.
 *
 * Uses a multi-pass approach:
 * 1. Extract bright areas (threshold)
 * 2. Blur the bright areas (two-pass Gaussian)
 * 3. Combine with original image
 */
class BloomRenderer {
 public:
  BloomRenderer();
  ~BloomRenderer();

  void init(int width, int height);
  void shutdown();
  void resize(int width, int height);

  /**
   * Apply bloom effect to the current framebuffer content.
   * @param sourceTexture The texture to apply bloom to
   * @param intensity Bloom intensity (0-2)
   * @param threshold Brightness threshold for bloom extraction
   */
  void apply(unsigned int sourceTexture, float intensity, float threshold);

  unsigned int getOutputTexture() const { return _outputTexture; }

 private:
  void createResources();
  void destroyResources();

  int _width = 0;
  int _height = 0;

  // Framebuffers and textures
  unsigned int _brightFbo = 0;
  unsigned int _brightTexture = 0;

  unsigned int _blurFbo[2] = {0, 0};
  unsigned int _blurTexture[2] = {0, 0};

  unsigned int _outputFbo = 0;
  unsigned int _outputTexture = 0;

  // Shaders
  unsigned int _extractShader = 0;
  unsigned int _blurShader = 0;
  unsigned int _combineShader = 0;

  // Fullscreen quad
  unsigned int _quadVao = 0;
  unsigned int _quadVbo = 0;
};
