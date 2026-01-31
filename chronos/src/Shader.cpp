#include "Shader.h"

#include <glad/glad.h>

#include <fstream>
#include <iostream>
#include <sstream>

Shader::~Shader() {
  if (_program) {
    glDeleteProgram(_program);
  }
}

bool Shader::loadFromSource(const std::string& vertexSource,
                            const std::string& fragmentSource) {
  // Compile vertex shader
  unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
  const char* vSrc = vertexSource.c_str();
  glShaderSource(vertexShader, 1, &vSrc, nullptr);
  glCompileShader(vertexShader);

  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
    std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
    return false;
  }

  // Compile fragment shader
  unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  const char* fSrc = fragmentSource.c_str();
  glShaderSource(fragmentShader, 1, &fSrc, nullptr);
  glCompileShader(fragmentShader);

  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
    std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
    glDeleteShader(vertexShader);
    return false;
  }

  // Link program
  _program = glCreateProgram();
  glAttachShader(_program, vertexShader);
  glAttachShader(_program, fragmentShader);
  glLinkProgram(_program);

  glGetProgramiv(_program, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(_program, 512, nullptr, infoLog);
    std::cerr << "Shader program linking failed: " << infoLog << std::endl;
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return false;
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return true;
}

bool Shader::loadFromFiles(const std::string& vertexPath,
                           const std::string& fragmentPath) {
  std::ifstream vFile(vertexPath);
  std::ifstream fFile(fragmentPath);

  if (!vFile.is_open()) {
    std::cerr << "Failed to open vertex shader: " << vertexPath << std::endl;
    return false;
  }
  if (!fFile.is_open()) {
    std::cerr << "Failed to open fragment shader: " << fragmentPath
              << std::endl;
    return false;
  }

  std::stringstream vStream, fStream;
  vStream << vFile.rdbuf();
  fStream << fFile.rdbuf();

  return loadFromSource(vStream.str(), fStream.str());
}

void Shader::use() const { glUseProgram(_program); }

void Shader::setInt(const std::string& name, int value) const {
  glUniform1i(glGetUniformLocation(_program, name.c_str()), value);
}

void Shader::setFloat(const std::string& name, float value) const {
  glUniform1f(glGetUniformLocation(_program, name.c_str()), value);
}

void Shader::setVec2(const std::string& name, float x, float y) const {
  glUniform2f(glGetUniformLocation(_program, name.c_str()), x, y);
}

void Shader::setVec3(const std::string& name, float x, float y, float z) const {
  glUniform3f(glGetUniformLocation(_program, name.c_str()), x, y, z);
}

void Shader::setMat4(const std::string& name, const float* value) const {
  glUniformMatrix4fv(glGetUniformLocation(_program, name.c_str()), 1, GL_FALSE,
                     value);
}
