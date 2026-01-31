#pragma once

#include <cstddef>
#include <deque>
#include <optional>

/**
 * ChronosHistoryBuffer - Stores a circular buffer of rendered frame textures.
 *
 * This is the core of the "Time-Cube" visualization: by keeping the last N
 * frames in GPU memory, we can stack them along the Z-axis to create a 3D
 * spacetime volume.
 */
class ChronosHistoryBuffer {
 public:
  static constexpr size_t DEFAULT_MAX_FRAMES = 128;
  static constexpr size_t MAX_FRAMES_LIMIT = 512;

  ChronosHistoryBuffer();
  ~ChronosHistoryBuffer();

  // Disable copy (OpenGL resources)
  ChronosHistoryBuffer(ChronosHistoryBuffer const&) = delete;
  ChronosHistoryBuffer& operator=(ChronosHistoryBuffer const&) = delete;

  /**
   * Initialize or reinitialize the buffer with a new size.
   * This will clear all existing history.
   */
  void resize(int width, int height);

  /**
   * Capture from a texture and store it as a new frame.
   */
  void captureFromTexture(unsigned int sourceTexture);

  /**
   * Capture from a framebuffer and store it as a new frame.
   */
  void captureFromFBO(unsigned int sourceFbo);

  /**
   * Get all stored frame textures, ordered from oldest to newest.
   */
  std::deque<unsigned int> const& getFrameTextures() const;

  /**
   * Get the texture for a specific frame index (0 = oldest).
   */
  std::optional<unsigned int> getFrameTexture(size_t index) const;

  /**
   * Get the number of frames currently stored.
   */
  size_t getFrameCount() const;

  /**
   * Get the maximum number of frames that can be stored.
   */
  size_t getMaxFrames() const;

  /**
   * Set the maximum number of frames to store.
   */
  void setMaxFrames(size_t maxFrames);

  /**
   * Check if the buffer is enabled and capturing.
   */
  bool isEnabled() const;

  /**
   * Enable or disable frame capture.
   */
  void setEnabled(bool enabled);

  /**
   * Clear all stored frames.
   */
  void clear();

  /**
   * Get texture dimensions.
   */
  int getWidth() const { return _width; }
  int getHeight() const { return _height; }

 private:
  struct FrameData {
    unsigned int fbo = 0;
    unsigned int texture = 0;
    bool initialized = false;
  };

  FrameData createFrameData();
  void destroyFrameData(FrameData& frame);

  std::deque<FrameData> _frames;
  std::deque<unsigned int> _textureIds;  // Cache for getFrameTextures()
  std::deque<FrameData> _freePool;

  size_t _maxFrames = DEFAULT_MAX_FRAMES;
  bool _enabled = false;
  int _width = 0;
  int _height = 0;
  bool _textureListDirty = true;
};
