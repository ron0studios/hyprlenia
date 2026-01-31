#include "BloomRenderer.h"

#include <glad/glad.h>

#include <iostream>

namespace {
const char* quadVertexShader = R"(
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* extractFragmentShader = R"(
#version 450 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D sourceTexture;
uniform float threshold;

void main() {
    vec4 color = texture(sourceTexture, TexCoord);
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    
    if (brightness > threshold) {
        FragColor = color * (brightness - threshold) / (1.0 - threshold);
    } else {
        FragColor = vec4(0.0);
    }
}
)";

const char* blurFragmentShader = R"(
#version 450 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D sourceTexture;
uniform vec2 direction;
uniform vec2 resolution;

// Gaussian blur weights for 9-tap filter
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 texelSize = 1.0 / resolution;
    vec3 result = texture(sourceTexture, TexCoord).rgb * weights[0];
    
    for (int i = 1; i < 5; ++i) {
        vec2 offset = direction * texelSize * float(i) * 2.0;
        result += texture(sourceTexture, TexCoord + offset).rgb * weights[i];
        result += texture(sourceTexture, TexCoord - offset).rgb * weights[i];
    }
    
    FragColor = vec4(result, 1.0);
}
)";

const char* combineFragmentShader = R"(
#version 450 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D sceneTexture;
uniform sampler2D bloomTexture;
uniform float intensity;

void main() {
    vec3 scene = texture(sceneTexture, TexCoord).rgb;
    vec3 bloom = texture(bloomTexture, TexCoord).rgb;
    
    // Additive blending with intensity control
    vec3 result = scene + bloom * intensity;
    
    // Tone mapping (simple Reinhard)
    result = result / (result + vec3(1.0));
    
    // Gamma correction
    result = pow(result, vec3(1.0 / 2.2));
    
    FragColor = vec4(result, 1.0);
}
)";

unsigned int compileShader(const char* vertexSrc, const char* fragmentSrc) {
  unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vertexSrc, nullptr);
  glCompileShader(vs);

  int success;
  char infoLog[512];
  glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vs, 512, nullptr, infoLog);
    std::cerr << "Bloom vertex shader error: " << infoLog << std::endl;
  }

  unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fragmentSrc, nullptr);
  glCompileShader(fs);

  glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fs, 512, nullptr, infoLog);
    std::cerr << "Bloom fragment shader error: " << infoLog << std::endl;
  }

  unsigned int program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  glDeleteShader(vs);
  glDeleteShader(fs);

  return program;
}
}  // namespace

BloomRenderer::BloomRenderer() = default;

BloomRenderer::~BloomRenderer() { shutdown(); }

void BloomRenderer::init(int width, int height) {
  _width = width;
  _height = height;

  // Compile shaders
  _extractShader = compileShader(quadVertexShader, extractFragmentShader);
  _blurShader = compileShader(quadVertexShader, blurFragmentShader);
  _combineShader = compileShader(quadVertexShader, combineFragmentShader);

  // Create fullscreen quad
  float quadVertices[] = {
      -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  -1.0f, 1.0f, 0.0f,
      1.0f,  1.0f,  1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f,
      1.0f,  1.0f,  1.0f, 1.0f, -1.0f, 1.0f,  0.0f, 1.0f,
  };

  glGenVertexArrays(1, &_quadVao);
  glGenBuffers(1, &_quadVbo);

  glBindVertexArray(_quadVao);
  glBindBuffer(GL_ARRAY_BUFFER, _quadVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);

  createResources();
}

void BloomRenderer::shutdown() {
  destroyResources();

  if (_extractShader) glDeleteProgram(_extractShader);
  if (_blurShader) glDeleteProgram(_blurShader);
  if (_combineShader) glDeleteProgram(_combineShader);
  if (_quadVao) glDeleteVertexArrays(1, &_quadVao);
  if (_quadVbo) glDeleteBuffers(1, &_quadVbo);

  _extractShader = 0;
  _blurShader = 0;
  _combineShader = 0;
  _quadVao = 0;
  _quadVbo = 0;
}

void BloomRenderer::resize(int width, int height) {
  if (_width == width && _height == height) return;

  _width = width;
  _height = height;
  destroyResources();
  createResources();
}

void BloomRenderer::createResources() {
  // Bloom textures at half resolution for performance
  int bloomWidth = _width / 2;
  int bloomHeight = _height / 2;

  // Bright extraction texture
  glGenTextures(1, &_brightTexture);
  glBindTexture(GL_TEXTURE_2D, _brightTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, bloomWidth, bloomHeight, 0, GL_RGBA,
               GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glGenFramebuffers(1, &_brightFbo);
  glBindFramebuffer(GL_FRAMEBUFFER, _brightFbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         _brightTexture, 0);

  // Blur ping-pong textures
  for (int i = 0; i < 2; ++i) {
    glGenTextures(1, &_blurTexture[i]);
    glBindTexture(GL_TEXTURE_2D, _blurTexture[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, bloomWidth, bloomHeight, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &_blurFbo[i]);
    glBindFramebuffer(GL_FRAMEBUFFER, _blurFbo[i]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           _blurTexture[i], 0);
  }

  // Output texture (full resolution)
  glGenTextures(1, &_outputTexture);
  glBindTexture(GL_TEXTURE_2D, _outputTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _width, _height, 0, GL_RGBA,
               GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glGenFramebuffers(1, &_outputFbo);
  glBindFramebuffer(GL_FRAMEBUFFER, _outputFbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         _outputTexture, 0);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BloomRenderer::destroyResources() {
  if (_brightFbo) glDeleteFramebuffers(1, &_brightFbo);
  if (_brightTexture) glDeleteTextures(1, &_brightTexture);

  for (int i = 0; i < 2; ++i) {
    if (_blurFbo[i]) glDeleteFramebuffers(1, &_blurFbo[i]);
    if (_blurTexture[i]) glDeleteTextures(1, &_blurTexture[i]);
  }

  if (_outputFbo) glDeleteFramebuffers(1, &_outputFbo);
  if (_outputTexture) glDeleteTextures(1, &_outputTexture);

  _brightFbo = 0;
  _brightTexture = 0;
  _blurFbo[0] = _blurFbo[1] = 0;
  _blurTexture[0] = _blurTexture[1] = 0;
  _outputFbo = 0;
  _outputTexture = 0;
}

void BloomRenderer::apply(unsigned int sourceTexture, float intensity,
                          float threshold) {
  int bloomWidth = _width / 2;
  int bloomHeight = _height / 2;

  glDisable(GL_DEPTH_TEST);

  // Step 1: Extract bright areas
  glBindFramebuffer(GL_FRAMEBUFFER, _brightFbo);
  glViewport(0, 0, bloomWidth, bloomHeight);
  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(_extractShader);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, sourceTexture);
  glUniform1i(glGetUniformLocation(_extractShader, "sourceTexture"), 0);
  glUniform1f(glGetUniformLocation(_extractShader, "threshold"), threshold);

  glBindVertexArray(_quadVao);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  // Step 2: Blur (multiple passes for stronger effect)
  glUseProgram(_blurShader);
  glUniform2f(glGetUniformLocation(_blurShader, "resolution"),
              static_cast<float>(bloomWidth), static_cast<float>(bloomHeight));

  bool horizontal = true;
  unsigned int currentTexture = _brightTexture;

  for (int pass = 0; pass < 6; ++pass) {
    glBindFramebuffer(GL_FRAMEBUFFER, _blurFbo[horizontal ? 0 : 1]);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, currentTexture);
    glUniform1i(glGetUniformLocation(_blurShader, "sourceTexture"), 0);
    glUniform2f(glGetUniformLocation(_blurShader, "direction"),
                horizontal ? 1.0f : 0.0f, horizontal ? 0.0f : 1.0f);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    currentTexture = _blurTexture[horizontal ? 0 : 1];
    horizontal = !horizontal;
  }

  // Step 3: Combine with original
  glBindFramebuffer(GL_FRAMEBUFFER, _outputFbo);
  glViewport(0, 0, _width, _height);
  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(_combineShader);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, sourceTexture);
  glUniform1i(glGetUniformLocation(_combineShader, "sceneTexture"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, currentTexture);
  glUniform1i(glGetUniformLocation(_combineShader, "bloomTexture"), 1);

  glUniform1f(glGetUniformLocation(_combineShader, "intensity"), intensity);

  glDrawArrays(GL_TRIANGLES, 0, 6);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glEnable(GL_DEPTH_TEST);
}
