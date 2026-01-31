#ifndef CHRONOS_SHADER_H
#define CHRONOS_SHADER_H

#include <glad/glad.h>

#include <array>
#include <string>

class Shader {
 public:
  Shader();
  virtual ~Shader();

  GLuint getId() const { return m_id; }

  void use() const;

  // Uniform binding methods
  void setUniform(const std::string& name, int value) const;
  void setUniform(const std::string& name, float value) const;
  void setUniform(const std::string& name, bool value) const;
  void setUniform(const std::string& name,
                  const std::array<float, 3>& vec) const;
  void setUniform(const std::string& name,
                  const std::array<float, 4>& vec) const;
  void setUniform(const std::string& name, float* values, int count) const;

  GLint getUniformLocation(const std::string& name) const;

 protected:
  GLuint m_id;

  std::string readFile(const std::string& path);
  GLuint compileShader(GLenum type, const std::string& source);
  void checkCompileErrors(GLuint shader, const std::string& type);
  void checkLinkErrors(GLuint program);
};

#endif  // CHRONOS_SHADER_H
