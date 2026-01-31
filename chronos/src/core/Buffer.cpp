#include "Buffer.h"

#include <cstring>
#include <iostream>

Buffer::Buffer()
    : m_id(0),
      m_count(0),
      m_type(GL_SHADER_STORAGE_BUFFER),
      m_initialized(false) {}

Buffer::Buffer(int count, GLenum type)
    : m_id(0), m_count(count), m_type(type), m_initialized(false) {}

Buffer::~Buffer() { cleanup(); }

void Buffer::init() {
  if (m_initialized) return;

  glGenBuffers(1, &m_id);
  glBindBuffer(m_type, m_id);
  glBufferData(m_type, sizeof(float) * m_count, nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(m_type, 0);

  m_initialized = true;
}

void Buffer::cleanup() {
  if (m_initialized && m_id != 0) {
    glDeleteBuffers(1, &m_id);
    m_id = 0;
    m_initialized = false;
  }
}

void Buffer::setData(const std::vector<float>& data) {
  if (!m_initialized) {
    std::cerr << "Buffer not initialized!" << std::endl;
    return;
  }

  glBindBuffer(m_type, m_id);
  GLvoid* ptr = glMapBuffer(m_type, GL_WRITE_ONLY);
  if (ptr) {
    size_t copySize =
        std::min(data.size(), static_cast<size_t>(m_count)) * sizeof(float);
    memcpy(ptr, data.data(), copySize);
    glUnmapBuffer(m_type);
  }
  glBindBuffer(m_type, 0);
}

std::vector<float> Buffer::getData() const {
  std::vector<float> data(m_count);
  glBindBuffer(m_type, m_id);
  glGetBufferSubData(m_type, 0, m_count * sizeof(float), data.data());
  glBindBuffer(m_type, 0);
  return data;
}

void Buffer::bind(GLuint index) const { glBindBufferBase(m_type, index, m_id); }

void Buffer::unbind() const { glBindBuffer(m_type, 0); }
