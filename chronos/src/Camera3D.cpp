#include "Camera3D.h"

#include <algorithm>
#include <cmath>

Camera3D::Camera3D() { updateVectors(); }

void Camera3D::setPosition(glm::vec3 const& position) { _position = position; }

void Camera3D::moveForward(float amount) { _position += _front * amount; }

void Camera3D::moveRight(float amount) { _position += _right * amount; }

void Camera3D::moveUp(float amount) { _position += _worldUp * amount; }

void Camera3D::rotate(float yawDelta, float pitchDelta) {
  _yaw += yawDelta;
  _pitch += pitchDelta;

  // Clamp pitch to avoid gimbal lock
  _pitch = std::clamp(_pitch, -89.0f, 89.0f);

  updateVectors();
}

glm::mat4 Camera3D::getViewMatrix() const {
  return glm::lookAt(_position, _position + _front, _up);
}

glm::mat4 Camera3D::getProjectionMatrix(float aspectRatio) const {
  return glm::perspective(glm::radians(_fov), aspectRatio, _nearPlane,
                          _farPlane);
}

void Camera3D::updateVectors() {
  glm::vec3 front;
  front.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
  front.y = sin(glm::radians(_pitch));
  front.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));
  _front = glm::normalize(front);

  _right = glm::normalize(glm::cross(_front, _worldUp));
  _up = glm::normalize(glm::cross(_right, _front));
}
