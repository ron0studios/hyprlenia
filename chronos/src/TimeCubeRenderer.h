#pragma once

#include <glm/glm.hpp>

class ChronosHistoryBuffer;

/**
 * TimeCubeRenderer - Renders the history buffer as a 3D stack of time slices.
 *
 * Each frame texture is rendered as a semi-transparent quad positioned along
 * the Z-axis. Older frames are further back (negative Z), newer frames are
 * closer to the camera.
 */
class TimeCubeRenderer {
 public:
  TimeCubeRenderer();
  ~TimeCubeRenderer();

  void init();
  void shutdown();

  /**
   * Render the time cube visualization.
   *
   * @param historyBuffer The buffer containing frame textures
   * @param view Camera view matrix
   * @param projection Camera projection matrix
   * @param layerSpacing Distance between layers in Z
   * @param layerAlpha Base alpha for each layer
   * @param alphaThreshold Pixels with alpha below this are discarded
   * @param useHeatmapColors Apply time-based color gradient
   */
  void render(ChronosHistoryBuffer const& historyBuffer, glm::mat4 const& view,
              glm::mat4 const& projection, float layerSpacing, float layerAlpha,
              float alphaThreshold, bool useHeatmapColors);

 private:
  void createQuadMesh();

  unsigned int _shaderProgram = 0;
  unsigned int _vao = 0;
  unsigned int _vbo = 0;
  unsigned int _ebo = 0;

  // Uniform locations
  int _locModel = -1;
  int _locView = -1;
  int _locProjection = -1;
  int _locTexture = -1;
  int _locAlpha = -1;
  int _locAlphaThreshold = -1;
  int _locTimeColor = -1;
  int _locUseHeatmap = -1;
};
