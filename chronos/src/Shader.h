#pragma once

#include <string>

/**
 * Shader - Simple shader utility class.
 */
class Shader {
 public:
  Shader() = default;
  ~Shader();

  bool loadFromSource(const std::string& vertexSource,
                      const std::string& fragmentSource);
  bool loadFromFiles(const std::string& vertexPath,
                     const std::string& fragmentPath);

  void use() const;
  unsigned int getProgram() const { return _program; }

  // Uniform setters
  void setInt(const std::string& name, int value) const;
  void setFloat(const std::string& name, float value) const;
  void setVec2(const std::string& name, float x, float y) const;
  void setVec3(const std::string& name, float x, float y, float z) const;
  void setMat4(const std::string& name, const float* value) const;

 private:
  unsigned int _program = 0;
};
