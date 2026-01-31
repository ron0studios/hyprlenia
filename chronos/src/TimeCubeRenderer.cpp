#include "TimeCubeRenderer.h"

#include <glad/glad.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "ChronosHistoryBuffer.h"

namespace {
const char* vertexShaderSource = R"(
#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* fragmentShaderSource = R"(
#version 450 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D frameTexture;
uniform float alpha;
uniform float alphaThreshold;
uniform vec3 timeColor;
uniform bool useHeatmap;

// Heatmap: cold (blue) -> hot (red/yellow)
vec3 heatmap(float t) {
    // t: 0 = old (cold), 1 = new (hot)
    vec3 cold = vec3(0.1, 0.2, 0.8);   // Blue
    vec3 mid = vec3(0.2, 0.8, 0.3);    // Green
    vec3 hot = vec3(1.0, 0.3, 0.1);    // Red-orange
    
    if (t < 0.5) {
        return mix(cold, mid, t * 2.0);
    } else {
        return mix(mid, hot, (t - 0.5) * 2.0);
    }
}

void main()
{
    vec4 texColor = texture(frameTexture, TexCoord);
    
    // Compute luminance for alpha thresholding (make dark areas transparent)
    float luminance = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));
    
    if (luminance < alphaThreshold) {
        discard;
    }
    
    vec3 finalColor = texColor.rgb;
    
    if (useHeatmap) {
        // Blend with time-based heatmap color
        vec3 heatColor = timeColor;
        finalColor = mix(texColor.rgb, texColor.rgb * heatColor, 0.5);
    }
    
    // Add glow effect for bright areas
    float glow = smoothstep(0.5, 1.0, luminance) * 0.3;
    finalColor += vec3(glow);
    
    FragColor = vec4(finalColor, alpha * texColor.a);
}
)";
}  // namespace

TimeCubeRenderer::TimeCubeRenderer() = default;

TimeCubeRenderer::~TimeCubeRenderer() { shutdown(); }

void TimeCubeRenderer::init() {
  // Compile shaders
  unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
  glCompileShader(vertexShader);

  unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
  glCompileShader(fragmentShader);

  _shaderProgram = glCreateProgram();
  glAttachShader(_shaderProgram, vertexShader);
  glAttachShader(_shaderProgram, fragmentShader);
  glLinkProgram(_shaderProgram);

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // Get uniform locations
  _locModel = glGetUniformLocation(_shaderProgram, "model");
  _locView = glGetUniformLocation(_shaderProgram, "view");
  _locProjection = glGetUniformLocation(_shaderProgram, "projection");
  _locTexture = glGetUniformLocation(_shaderProgram, "frameTexture");
  _locAlpha = glGetUniformLocation(_shaderProgram, "alpha");
  _locAlphaThreshold = glGetUniformLocation(_shaderProgram, "alphaThreshold");
  _locTimeColor = glGetUniformLocation(_shaderProgram, "timeColor");
  _locUseHeatmap = glGetUniformLocation(_shaderProgram, "useHeatmap");

  createQuadMesh();
}

void TimeCubeRenderer::shutdown() {
  if (_shaderProgram) {
    glDeleteProgram(_shaderProgram);
    _shaderProgram = 0;
  }
  if (_vao) {
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;
  }
  if (_vbo) {
    glDeleteBuffers(1, &_vbo);
    _vbo = 0;
  }
  if (_ebo) {
    glDeleteBuffers(1, &_ebo);
    _ebo = 0;
  }
}

void TimeCubeRenderer::createQuadMesh() {
  // Quad vertices: position (x, y, z) + texcoord (u, v)
  float vertices[] = {// positions          // texture coords
                      -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f,
                      0.0f,  1.0f,  0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
                      1.0f,  -1.0f, 1.0f, 0.0f, 0.0f, 1.0f};

  unsigned int indices[] = {0, 1, 2, 2, 3, 0};

  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);
  glGenBuffers(1, &_ebo);

  glBindVertexArray(_vao);

  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // Position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // Texture coord attribute
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                        (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
}

void TimeCubeRenderer::render(ChronosHistoryBuffer const& historyBuffer,
                              glm::mat4 const& view,
                              glm::mat4 const& projection, float layerSpacing,
                              float layerAlpha, float alphaThreshold,
                              bool useHeatmapColors) {
  auto const& textures = historyBuffer.getFrameTextures();
  if (textures.empty()) {
    return;
  }

  glUseProgram(_shaderProgram);
  glUniformMatrix4fv(_locView, 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(_locProjection, 1, GL_FALSE, glm::value_ptr(projection));
  glUniform1i(_locTexture, 0);
  glUniform1f(_locAlpha, layerAlpha);
  glUniform1f(_locAlphaThreshold, alphaThreshold);
  glUniform1i(_locUseHeatmap, useHeatmapColors ? 1 : 0);

  glBindVertexArray(_vao);
  glActiveTexture(GL_TEXTURE0);

  // Enable blending for transparency
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Disable depth write but keep depth test for proper ordering
  glDepthMask(GL_FALSE);

  size_t numFrames = textures.size();
  float totalDepth = static_cast<float>(numFrames) * layerSpacing;
  float startZ = -totalDepth / 2.0f;

  // Render from back to front (oldest to newest)
  for (size_t i = 0; i < numFrames; ++i) {
    float z = startZ + static_cast<float>(i) * layerSpacing;
    float t = static_cast<float>(i) / static_cast<float>(numFrames - 1);

    // Heatmap color: cold (blue) for old, hot (red) for new
    glm::vec3 heatColor;
    if (t < 0.5f) {
      heatColor = glm::mix(glm::vec3(0.1f, 0.2f, 0.8f),
                           glm::vec3(0.2f, 0.8f, 0.3f), t * 2.0f);
    } else {
      heatColor = glm::mix(glm::vec3(0.2f, 0.8f, 0.3f),
                           glm::vec3(1.0f, 0.3f, 0.1f), (t - 0.5f) * 2.0f);
    }

    glUniform3fv(_locTimeColor, 1, glm::value_ptr(heatColor));

    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, z));
    glUniformMatrix4fv(_locModel, 1, GL_FALSE, glm::value_ptr(model));

    glBindTexture(GL_TEXTURE_2D, textures[i]);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  }

  glDepthMask(GL_TRUE);
  glBindVertexArray(0);
  glUseProgram(0);
}
