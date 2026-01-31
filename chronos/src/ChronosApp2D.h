#pragma once

#include <string>
#include <memory>

struct GLFWwindow;

class FlowLenia2D;
class BloomEffect;

class ChronosApp2D {
public:
    ChronosApp2D();
    ~ChronosApp2D();
    
    bool init(int width, int height, const std::string& title);
    void run();
    void shutdown();
    
private:
    void processInput();
    void update(float deltaTime);
    void render();
    void renderUI();
    
    void createDisplayShader();
    void createQuad();
    void createRenderTexture();
    
    // Window
    GLFWwindow* _window = nullptr;
    int _windowWidth = 1280;
    int _windowHeight = 720;
    
    // Flow Lenia simulation
    std::unique_ptr<FlowLenia2D> _lenia;
    std::unique_ptr<BloomEffect> _bloom;
    int _simWidth = 512;
    int _simHeight = 512;
    
    // Display shader and quad
    unsigned int _displayProgram = 0;
    unsigned int _quadVAO = 0;
    unsigned int _quadVBO = 0;
    
    // Render texture for bloom input
    unsigned int _renderTexture = 0;
    unsigned int _renderFBO = 0;
    
    // Simulation state
    bool _paused = false;
    float _simulationSpeed = 1.0f;
    int _stepsPerFrame = 2;
    
    // Visual settings - reduced glow for clearer creatures
    float _bloomIntensity = 0.25f;
    float _bloomThreshold = 0.35f;
    float _glowPower = 1.8f;
    
    // Obstacle drawing
    bool _drawingObstacle = false;
    float _obstacleRadius = 20.0f;
    
    // Timing
    float _lastFrameTime = 0.0f;
    float _deltaTime = 0.0f;
    float _totalTime = 0.0f;
};
