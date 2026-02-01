#include "Shader.h"

#include <fstream>
#include <iostream>
#include <sstream>

Shader::Shader() : m_id(0) {}

Shader::~Shader() {
  if (m_id != 0) {
    glDeleteProgram(m_id);
  }
}

void Shader::use() const { glUseProgram(m_id); }

void Shader::setUniform(const std::string& name, int value) const {
  glUniform1i(glGetUniformLocation(m_id, name.c_str()), value);
}

void Shader::setUniform(const std::string& name, float value) const {
  glUniform1f(glGetUniformLocation(m_id, name.c_str()), value);
}

void Shader::setUniform(const std::string& name, bool value) const {
  glUniform1i(glGetUniformLocation(m_id, name.c_str()),
              static_cast<int>(value));
}

void Shader::setUniform(const std::string& name,
                        const std::array<float, 3>& vec) const {
  glUniform3f(glGetUniformLocation(m_id, name.c_str()), vec[0], vec[1], vec[2]);
}

void Shader::setUniform(const std::string& name,
                        const std::array<float, 4>& vec) const {
  glUniform4f(glGetUniformLocation(m_id, name.c_str()), vec[0], vec[1], vec[2],
              vec[3]);
}

void Shader::setUniform(const std::string& name, float* values,
                        int count) const {
  glUniform1fv(glGetUniformLocation(m_id, name.c_str()), count, values);
}

void Shader::setUniform(const std::string& name, float x, float y) const {
  glUniform2f(glGetUniformLocation(m_id, name.c_str()), x, y);
}

void Shader::setUniform(const std::string& name, float x, float y, float z) const {
  glUniform3f(glGetUniformLocation(m_id, name.c_str()), x, y, z);
}

void Shader::setUniformMat4(const std::string& name, const float* matrix) const {
  glUniformMatrix4fv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, matrix);
}

GLint Shader::getUniformLocation(const std::string& name) const {
  return glGetUniformLocation(m_id, name.c_str());
}

std::string Shader::readFile(const std::string& path) {
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  try {
    file.open(path);
    std::stringstream stream;
    stream << file.rdbuf();
    file.close();
    return stream.str();
  } catch (const std::ifstream::failure& e) {
    std::cerr << "ERROR: Failed to read shader file: " << path << std::endl;
    return "";
  }
}

GLuint Shader::compileShader(GLenum type, const std::string& source) {
  GLuint shader = glCreateShader(type);
  const char* src = source.c_str();
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);

  std::string typeName;
  switch (type) {
    case GL_VERTEX_SHADER:
      typeName = "VERTEX";
      break;
    case GL_FRAGMENT_SHADER:
      typeName = "FRAGMENT";
      break;
    case GL_COMPUTE_SHADER:
      typeName = "COMPUTE";
      break;
    default:
      typeName = "UNKNOWN";
      break;
  }

  checkCompileErrors(shader, typeName);
  return shader;
}

void Shader::checkCompileErrors(GLuint shader, const std::string& type) {
  int success;
  char infoLog[1024];

  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
    std::cerr << "ERROR: Shader compilation failed (" << type << "):\n"
              << infoLog << std::endl;
  }
}

void Shader::checkLinkErrors(GLuint program) {
  int success;
  char infoLog[1024];

  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(program, 1024, nullptr, infoLog);
    std::cerr << "ERROR: Program linking failed:\n" << infoLog << std::endl;
  }
}
