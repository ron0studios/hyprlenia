#include "ChronosApp2D.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "FlowLenia2D.h"
#include "BloomEffect.h"

ChronosApp2D::ChronosApp2D() = default;
ChronosApp2D::~ChronosApp2D() = default;

static std::string loadFile(const std::string& path) {
    std::vector<std::string> paths = {path, "../" + path, "shaders/" + path, "../shaders/" + path};
    for (const auto& p : paths) {
        std::ifstream file(p);
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }
    }
    return "";
}

void ChronosApp2D::createDisplayShader() {
    std::string vertSrc = loadFile("display_vert.glsl");
    std::string fragSrc = loadFile("display_frag.glsl");
    
    if (vertSrc.empty() || fragSrc.empty()) {
        std::cerr << "Failed to load display shaders" << std::endl;
        return;
    }
    
    const char* vSrc = vertSrc.c_str();
    const char* fSrc = fragSrc.c_str();
    
    unsigned int vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vSrc, nullptr);
    glCompileShader(vertShader);
    
    int success;
    char log[512];
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertShader, 512, nullptr, log);
        std::cerr << "Vertex shader error: " << log << std::endl;
    }
    
    unsigned int fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fSrc, nullptr);
    glCompileShader(fragShader);
    
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragShader, 512, nullptr, log);
        std::cerr << "Fragment shader error: " << log << std::endl;
    }
    
    _displayProgram = glCreateProgram();
    glAttachShader(_displayProgram, vertShader);
    glAttachShader(_displayProgram, fragShader);
    glLinkProgram(_displayProgram);
    
    glGetProgramiv(_displayProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(_displayProgram, 512, nullptr, log);
        std::cerr << "Shader program link error: " << log << std::endl;
    }
    
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
}

void ChronosApp2D::createQuad() {
    float vertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    
    glGenVertexArrays(1, &_quadVAO);
    glGenBuffers(1, &_quadVBO);
    
    glBindVertexArray(_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, _quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void ChronosApp2D::createRenderTexture() {
    // Create FBO and texture for rendering to (used for bloom input)
    glGenFramebuffers(1, &_renderFBO);
    glGenTextures(1, &_renderTexture);
    
    glBindTexture(GL_TEXTURE_2D, _renderTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _simWidth, _simHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glBindFramebuffer(GL_FRAMEBUFFER, _renderFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _renderTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool ChronosApp2D::init(int width, int height, const std::string& title) {
    _windowWidth = width;
    _windowHeight = height;
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    _window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);
    
    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }
    
    std::cout << "OpenGL " << GLAD_VERSION_MAJOR(version) << "." << GLAD_VERSION_MINOR(version) << std::endl;
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 10.0f;
    style.FrameRounding = 5.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.05f, 0.05f, 0.1f, 0.9f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.1f, 0.2f, 0.3f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.15f, 0.3f, 0.45f, 1.0f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.4f, 0.5f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.3f, 0.5f, 0.6f, 1.0f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.3f, 0.7f, 0.6f, 1.0f);
    
    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 430");
    
    // Create resources
    createDisplayShader();
    createQuad();
    createRenderTexture();
    
    // Initialize Flow Lenia
    _lenia = std::make_unique<FlowLenia2D>();
    _lenia->init(_simWidth, _simHeight);
    
    // Initialize bloom
    _bloom = std::make_unique<BloomEffect>();
    _bloom->init(_simWidth, _simHeight);
    
    std::cout << "Flow Lenia 2D initialized!" << std::endl;
    
    return true;
}

void ChronosApp2D::run() {
    while (!glfwWindowShouldClose(_window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        _deltaTime = currentFrame - _lastFrameTime;
        _lastFrameTime = currentFrame;
        _totalTime = currentFrame;
        
        glfwPollEvents();
        processInput();
        update(_deltaTime);
        render();
        renderUI();
        
        glfwSwapBuffers(_window);
    }
}

void ChronosApp2D::shutdown() {
    _bloom->shutdown();
    _lenia->shutdown();
    
    if (_displayProgram) glDeleteProgram(_displayProgram);
    if (_quadVAO) glDeleteVertexArrays(1, &_quadVAO);
    if (_quadVBO) glDeleteBuffers(1, &_quadVBO);
    if (_renderFBO) glDeleteFramebuffers(1, &_renderFBO);
    if (_renderTexture) glDeleteTextures(1, &_renderTexture);
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void ChronosApp2D::processInput() {
    if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(_window, true);
    }
    
    // Pause toggle
    static bool spaceWasPressed = false;
    bool spacePressed = glfwGetKey(_window, GLFW_KEY_SPACE) == GLFW_PRESS;
    if (spacePressed && !spaceWasPressed) {
        _paused = !_paused;
    }
    spaceWasPressed = spacePressed;
    
    // Reset
    static bool rWasPressed = false;
    bool rPressed = glfwGetKey(_window, GLFW_KEY_R) == GLFW_PRESS;
    if (rPressed && !rWasPressed) {
        _lenia->reset();
    }
    rWasPressed = rPressed;
    
    // Clear obstacles
    static bool cWasPressed = false;
    bool cPressed = glfwGetKey(_window, GLFW_KEY_C) == GLFW_PRESS;
    if (cPressed && !cWasPressed) {
        _lenia->clearObstacles();
    }
    cWasPressed = cPressed;
    
    // Check if shift is held for obstacle mode
    _drawingObstacle = glfwGetKey(_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                       glfwGetKey(_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    
    // Mouse interaction
    double mx, my;
    glfwGetCursorPos(_window, &mx, &my);
    float x = static_cast<float>(mx) / _windowWidth * _simWidth;
    float y = static_cast<float>(_windowHeight - my) / _windowHeight * _simHeight;
    
    if (glfwGetMouseButton(_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        if (_drawingObstacle) {
            _lenia->addObstacle(x, y, _obstacleRadius);
        } else {
            _lenia->addBlob(x, y, _lenia->R * 2.5f);
        }
    }
}

void ChronosApp2D::update(float deltaTime) {
    if (!_paused) {
        for (int i = 0; i < _stepsPerFrame; i++) {
            _lenia->update();
        }
    }
    
    // Update bloom threshold
    _bloom->threshold = _bloomThreshold;
}

void ChronosApp2D::render() {
    int width, height;
    glfwGetFramebufferSize(_window, &width, &height);
    
    if (width != _windowWidth || height != _windowHeight) {
        _windowWidth = width;
        _windowHeight = height;
    }
    
    // Process bloom on the lenia texture
    _bloom->process(_lenia->getTexture());
    
    // Render to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.02f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glUseProgram(_displayProgram);
    
    // Bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _lenia->getTexture());
    glUniform1i(glGetUniformLocation(_displayProgram, "leniaTexture"), 0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _bloom->getBloomTexture());
    glUniform1i(glGetUniformLocation(_displayProgram, "bloomTexture"), 1);
    
    // Set uniforms
    glUniform1f(glGetUniformLocation(_displayProgram, "time"), _totalTime);
    glUniform1f(glGetUniformLocation(_displayProgram, "bloomIntensity"), _bloomIntensity);
    glUniform1f(glGetUniformLocation(_displayProgram, "glowPower"), _glowPower);
    
    glBindVertexArray(_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void ChronosApp2D::renderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 480), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Flow Lenia Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // FPS display with color
    float fps = 1.0f / _deltaTime;
    ImVec4 fpsColor = fps > 50 ? ImVec4(0.3f, 1.0f, 0.5f, 1.0f) : 
                      fps > 30 ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f) : 
                                 ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
    ImGui::TextColored(fpsColor, "FPS: %.1f", fps);
    
    ImGui::Separator();
    
    // Playback controls
    ImGui::Text("Playback");
    if (ImGui::Button(_paused ? "▶ Resume" : "⏸ Pause", ImVec2(100, 30))) {
        _paused = !_paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("↺ Reset", ImVec2(100, 30))) {
        _lenia->reset();
    }
    
    ImGui::Separator();
    
    // Simulation parameters
    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Kernel Radius", &_lenia->R, 6.0f, 25.0f);
        ImGui::SliderFloat("Time Resolution", &_lenia->T, 2.0f, 20.0f);
        ImGui::SliderFloat("Base Noise", &_lenia->baseNoise, 0.1f, 1.0f);
        ImGui::SliderInt("Steps/Frame", &_stepsPerFrame, 1, 8);
    }
    
    // Visual settings
    if (ImGui::CollapsingHeader("Visuals", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Bloom Intensity", &_bloomIntensity, 0.0f, 2.0f);
        ImGui::SliderFloat("Bloom Threshold", &_bloomThreshold, 0.0f, 0.8f);
        ImGui::SliderFloat("Glow Power", &_glowPower, 1.0f, 5.0f);
        ImGui::SliderInt("Blur Passes", &_bloom->blurPasses, 1, 8);
    }
    
    // Obstacles
    if (ImGui::CollapsingHeader("Obstacles")) {
        ImGui::SliderFloat("Obstacle Size", &_obstacleRadius, 5.0f, 50.0f);
        if (ImGui::Button("Clear Obstacles", ImVec2(-1, 0))) {
            _lenia->clearObstacles();
        }
        ImGui::TextWrapped("Hold SHIFT + Left-click to draw obstacles");
    }
    
    ImGui::Separator();
    
    // Controls help
    ImGui::Text("Controls:");
    ImGui::BulletText("Left-click: Add organism");
    ImGui::BulletText("SHIFT + Left-click: Add obstacle");
    ImGui::BulletText("Space: Pause/Resume");
    ImGui::BulletText("R: Reset simulation");
    ImGui::BulletText("C: Clear obstacles");
    
    ImGui::End();
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
