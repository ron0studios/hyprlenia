#include "ComputeShader.h"

#include <iostream>

ComputeShader::ComputeShader() : Shader(), m_path("") {}

ComputeShader::ComputeShader(const std::string& path)
    : Shader(), m_path(path) {}

void ComputeShader::init() {
  std::string source = readFile(m_path);
  if (source.empty()) {
    std::cerr << "ERROR: Failed to load compute shader: " << m_path
              << std::endl;
    return;
  }

  GLuint shader = compileShader(GL_COMPUTE_SHADER, source);

  m_id = glCreateProgram();
  glAttachShader(m_id, shader);
  glLinkProgram(m_id);
  checkLinkErrors(m_id);

  glDeleteShader(shader);
}

void ComputeShader::dispatch(GLuint x, GLuint y, GLuint z) const {
  glDispatchCompute(x, y, z);
}

void ComputeShader::wait() const { glMemoryBarrier(GL_ALL_BARRIER_BITS); }

void ComputeShader::bindBuffer(const std::string& name, const Buffer& buffer,
                               GLuint bindingPoint) const {
  GLint blockIndex =
      glGetProgramResourceIndex(m_id, GL_SHADER_STORAGE_BLOCK, name.c_str());
  if (blockIndex != GL_INVALID_INDEX) {
    glShaderStorageBlockBinding(m_id, blockIndex, bindingPoint);
  }
  buffer.bind(bindingPoint);
}
