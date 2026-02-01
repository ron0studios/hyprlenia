#include "RenderShader.h"

#include <iostream>

RenderShader::RenderShader()
    : Shader(),
      m_vertexPath(""),
      m_fragmentPath(""),
      m_vao(0),
      m_vbo(0),
      m_ebo(0) {}

RenderShader::RenderShader(const std::string& vertexPath,
                           const std::string& fragmentPath)
    : Shader(),
      m_vertexPath(vertexPath),
      m_fragmentPath(fragmentPath),
      m_vao(0),
      m_vbo(0),
      m_ebo(0) {}

void RenderShader::init() {
  // Read and compile shaders
  std::string vertexSource = readFile(m_vertexPath);
  std::string fragmentSource = readFile(m_fragmentPath);

  if (vertexSource.empty() || fragmentSource.empty()) {
    std::cerr << "ERROR: Failed to load render shaders" << std::endl;
    return;
  }

  GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
  GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

  m_id = glCreateProgram();
  glAttachShader(m_id, vertexShader);
  glAttachShader(m_id, fragmentShader);
  glLinkProgram(m_id);
  checkLinkErrors(m_id);

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // Setup fullscreen quad
  float vertices[] = {// positions        // texcoords
                      1.0f, 1.0f,  0.0f, 1.0f,  1.0f,  1.0f, -1.0f,
                      0.0f, 1.0f,  0.0f, -1.0f, -1.0f, 0.0f, 0.0f,
                      0.0f, -1.0f, 1.0f, 0.0f,  0.0f,  1.0f};

  unsigned int indices[] = {0, 1, 3, 1, 2, 3};

  glGenVertexArrays(1, &m_vao);
  glGenBuffers(1, &m_vbo);
  glGenBuffers(1, &m_ebo);

  glBindVertexArray(m_vao);

  glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // Position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // TexCoord attribute
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                        (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
}

void RenderShader::render() const {
  glBindVertexArray(m_vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);
}

void RenderShader::bindBuffer(const std::string& name, const Buffer& buffer,
                              GLuint bindingPoint) const {
  GLint blockIndex =
      glGetProgramResourceIndex(m_id, GL_SHADER_STORAGE_BLOCK, name.c_str());
  if (blockIndex != GL_INVALID_INDEX) {
    glShaderStorageBlockBinding(m_id, blockIndex, bindingPoint);
  }
  buffer.bind(bindingPoint);
}
