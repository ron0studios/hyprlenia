#ifndef CHRONOS_COMPUTE_SHADER_H
#define CHRONOS_COMPUTE_SHADER_H

#include "Buffer.h"
#include "Shader.h"

class ComputeShader : public Shader {
 public:
  ComputeShader();
  explicit ComputeShader(const std::string& path);

  void init();
  void dispatch(GLuint x, GLuint y, GLuint z) const;
  void wait() const;

  void bindBuffer(const std::string& name, const Buffer& buffer,
                  GLuint bindingPoint) const;

 private:
  std::string m_path;
};

#endif  // CHRONOS_COMPUTE_SHADER_H
