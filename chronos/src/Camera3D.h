#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/**
 * Camera3D - A simple 3D camera with FPS-style controls.
 */
class Camera3D {
 public:
  Camera3D();

  void setPosition(glm::vec3 const& position);
  glm::vec3 getPosition() const { return _position; }

  void moveForward(float amount);
  void moveRight(float amount);
  void moveUp(float amount);

  void rotate(float yawDelta, float pitchDelta);

  glm::mat4 getViewMatrix() const;
  glm::mat4 getProjectionMatrix(float aspectRatio) const;

  void setFOV(float fov) { _fov = fov; }
  float getFOV() const { return _fov; }

  void setNearPlane(float near) { _nearPlane = near; }
  void setFarPlane(float far) { _farPlane = far; }

 private:
  void updateVectors();

  glm::vec3 _position = glm::vec3(0.0f, 0.0f, 3.0f);
  glm::vec3 _front = glm::vec3(0.0f, 0.0f, -1.0f);
  glm::vec3 _up = glm::vec3(0.0f, 1.0f, 0.0f);
  glm::vec3 _right = glm::vec3(1.0f, 0.0f, 0.0f);
  glm::vec3 _worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

  float _yaw = -90.0f;
  float _pitch = 0.0f;
  float _fov = 60.0f;
  float _nearPlane = 0.1f;
  float _farPlane = 100.0f;
};
