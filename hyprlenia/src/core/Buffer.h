#ifndef CHRONOS_BUFFER_H
#define CHRONOS_BUFFER_H

#include <glad/glad.h>

#include <vector>

class Buffer {
 public:
  Buffer();
  Buffer(int count, GLenum type);
  ~Buffer();

  void init();
  void cleanup();

  void setData(const std::vector<float>& data);
  std::vector<float> getData() const;

  void bind(GLuint index) const;
  void unbind() const;

  GLuint getId() const { return m_id; }
  int getCount() const { return m_count; }

 private:
  GLuint m_id;
  int m_count;
  GLenum m_type;
  bool m_initialized;
};

#endif  // CHRONOS_BUFFER_H
