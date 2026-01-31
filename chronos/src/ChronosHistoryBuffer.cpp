#include "ChronosHistoryBuffer.h"

#include <glad/glad.h>

#include <algorithm>

ChronosHistoryBuffer::ChronosHistoryBuffer() = default;

ChronosHistoryBuffer::~ChronosHistoryBuffer() {
  clear();
  for (auto& frame : _freePool) {
    destroyFrameData(frame);
  }
  _freePool.clear();
}

void ChronosHistoryBuffer::resize(int width, int height) {
  if (_width == width && _height == height) {
    return;
  }

  clear();
  for (auto& frame : _freePool) {
    destroyFrameData(frame);
  }
  _freePool.clear();

  _width = width;
  _height = height;
}

void ChronosHistoryBuffer::captureFromTexture(unsigned int sourceTexture) {
  if (!_enabled || _width == 0 || _height == 0) {
    return;
  }

  // Get a frame data struct
  FrameData frameData;
  if (!_freePool.empty()) {
    frameData = _freePool.back();
    _freePool.pop_back();
  } else {
    frameData = createFrameData();
  }

  // Create temporary FBO to read from source texture
  unsigned int tempFbo;
  glGenFramebuffers(1, &tempFbo);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, tempFbo);
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                         GL_TEXTURE_2D, sourceTexture, 0);

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameData.fbo);
  glBlitFramebuffer(0, 0, _width, _height, 0, 0, _width, _height,
                    GL_COLOR_BUFFER_BIT, GL_NEAREST);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &tempFbo);

  _frames.push_back(frameData);
  _textureListDirty = true;

  while (_frames.size() > _maxFrames) {
    _freePool.push_back(_frames.front());
    _frames.pop_front();
  }
}

void ChronosHistoryBuffer::captureFromFBO(unsigned int sourceFbo) {
  if (!_enabled || _width == 0 || _height == 0) {
    return;
  }

  FrameData frameData;
  if (!_freePool.empty()) {
    frameData = _freePool.back();
    _freePool.pop_back();
  } else {
    frameData = createFrameData();
  }

  glBindFramebuffer(GL_READ_FRAMEBUFFER, sourceFbo);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameData.fbo);
  glBlitFramebuffer(0, 0, _width, _height, 0, 0, _width, _height,
                    GL_COLOR_BUFFER_BIT, GL_NEAREST);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  _frames.push_back(frameData);
  _textureListDirty = true;

  while (_frames.size() > _maxFrames) {
    _freePool.push_back(_frames.front());
    _frames.pop_front();
  }
}

std::deque<unsigned int> const& ChronosHistoryBuffer::getFrameTextures() const {
  if (_textureListDirty) {
    auto* self = const_cast<ChronosHistoryBuffer*>(this);
    self->_textureIds.clear();
    for (auto const& frame : _frames) {
      self->_textureIds.push_back(frame.texture);
    }
    self->_textureListDirty = false;
  }
  return _textureIds;
}

std::optional<unsigned int> ChronosHistoryBuffer::getFrameTexture(
    size_t index) const {
  if (index >= _frames.size()) {
    return std::nullopt;
  }
  return _frames[index].texture;
}

size_t ChronosHistoryBuffer::getFrameCount() const { return _frames.size(); }

size_t ChronosHistoryBuffer::getMaxFrames() const { return _maxFrames; }

void ChronosHistoryBuffer::setMaxFrames(size_t maxFrames) {
  _maxFrames = std::clamp(maxFrames, size_t(1), MAX_FRAMES_LIMIT);

  while (_frames.size() > _maxFrames) {
    _freePool.push_back(_frames.front());
    _frames.pop_front();
  }
  _textureListDirty = true;
}

bool ChronosHistoryBuffer::isEnabled() const { return _enabled; }

void ChronosHistoryBuffer::setEnabled(bool enabled) { _enabled = enabled; }

void ChronosHistoryBuffer::clear() {
  for (auto& frame : _frames) {
    _freePool.push_back(frame);
  }
  _frames.clear();
  _textureListDirty = true;
}

ChronosHistoryBuffer::FrameData ChronosHistoryBuffer::createFrameData() {
  FrameData frame;

  glGenTextures(1, &frame.texture);
  glBindTexture(GL_TEXTURE_2D, frame.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _width, _height, 0, GL_RGBA,
               GL_FLOAT, nullptr);

  glGenFramebuffers(1, &frame.fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, frame.fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         frame.texture, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  frame.initialized = true;
  return frame;
}

void ChronosHistoryBuffer::destroyFrameData(FrameData& frame) {
  if (frame.initialized) {
    glDeleteFramebuffers(1, &frame.fbo);
    glDeleteTextures(1, &frame.texture);
    frame.initialized = false;
    frame.fbo = 0;
    frame.texture = 0;
  }
}
