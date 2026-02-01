#ifndef CHRONOS_RENDER_SHADER_H
#define CHRONOS_RENDER_SHADER_H

#include "Buffer.h"
#include "Shader.h"

class RenderShader : public Shader {
 public:
  RenderShader();
  RenderShader(const std::string& vertexPath, const std::string& fragmentPath);

  void init();
  void render() const;

  void bindBuffer(const std::string& name, const Buffer& buffer,
                  GLuint bindingPoint) const;

 private:
  std::string m_vertexPath;
  std::string m_fragmentPath;

  GLuint m_vao;
  GLuint m_vbo;
  GLuint m_ebo;
};

#endif  
