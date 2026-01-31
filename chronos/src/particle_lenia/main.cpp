/*
 * CHRONOS - Particle Lenia with Evolution
 * 
 * An advanced cellular automata simulation featuring:
 * - Particle-based Lenia (continuous game of life)
 * - Multiple species with different parameters
 * - Evolution: particles can reproduce, mutate, and die
 * - Survival mechanics: energy, predation, competition
 * 
 * Based on: https://google-research.github.io/self-organising-systems/particle-lenia/
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "core/Buffer.h"
#include "core/ComputeShader.h"
#include "core/RenderShader.h"

#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cmath>

// Window dimensions
int WINDOW_WIDTH = 1200;
int WINDOW_HEIGHT = 900;

// Simulation parameters
struct SimulationParams {
    // World dimensions
    float worldWidth = 40.0f;
    float worldHeight = 40.0f;
    
    // Particle count
    int numParticles = 500;
    int maxParticles = 2000;
    
    // Kernel parameters - controls sensing/interaction range
    float w_k = 0.022f;      // Kernel weight
    float mu_k = 4.0f;       // Kernel peak distance
    float sigma_k2 = 1.0f;   // Kernel width squared
    
    // Growth parameters - the "Lenia magic"
    float mu_g = 0.6f;       // Optimal density (growth center)
    float sigma_g2 = 0.0225f; // Growth width squared
    
    // Repulsion parameters
    float c_rep = 1.0f;      // Repulsion strength
    
    // Time integration
    float dt = 0.1f;         // Time step
    float h = 0.01f;         // Gradient calculation distance
    
    // Evolution parameters
    bool evolutionEnabled = true;
    float birthRate = 0.001f;       // Chance to reproduce per step
    float deathRate = 0.0005f;      // Base death rate
    float mutationRate = 0.1f;      // Mutation strength
    float energyDecay = 0.001f;     // Energy loss per step
    float energyFromGrowth = 0.01f; // Energy gained from good growth
    
    // View parameters
    float translateX = 0.0f;
    float translateY = 0.0f;
    float zoom = 1.0f;
    
    // Rendering
    int stepsPerFrame = 5;
    bool showFields = true;
    int fieldType = 3; // 0=none, 1=U, 2=R, 3=G, 4=E
};

// Particle structure (must match shader)
struct Particle {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Health/energy [0, 1]
    float species;        // Species ID (affects color)
    float age;            // Age in simulation steps
    float dna[5];         // Genetic parameters (mu_k, sigma_k2, mu_g, sigma_g2, c_rep variations)
};

class ParticleLeniaSimulation {
public:
    SimulationParams params;
    
    Buffer particleBufferA;
    Buffer particleBufferB;
    bool useBufferA = true;
    
    ComputeShader stepShader;
    RenderShader displayShader;
    
    std::mt19937 rng;
    
    // Stats
    int aliveCount = 0;
    float avgEnergy = 0.0f;
    float avgAge = 0.0f;
    
    void init() {
        // Initialize RNG
        rng = std::mt19937(std::random_device{}());
        
        // Calculate buffer size: each particle has 12 floats
        int particleFloats = 12;
        int bufferSize = params.maxParticles * particleFloats;
        
        particleBufferA = Buffer(bufferSize, GL_SHADER_STORAGE_BUFFER);
        particleBufferB = Buffer(bufferSize, GL_SHADER_STORAGE_BUFFER);
        
        particleBufferA.init();
        particleBufferB.init();
        
        // Initialize particles
        resetParticles();
        
        // Load shaders
        stepShader = ComputeShader("shaders/particle_lenia_step.comp");
        stepShader.init();
        
        displayShader = RenderShader("shaders/passthrough.vert", "shaders/particle_lenia_display.frag");
        displayShader.init();
    }
    
    void resetParticles() {
        std::uniform_real_distribution<float> posDist(-params.worldWidth * 0.4f, params.worldWidth * 0.4f);
        std::uniform_real_distribution<float> speciesDist(0.0f, 3.0f);
        std::uniform_real_distribution<float> dnaDist(-0.2f, 0.2f);
        
        std::vector<float> data;
        data.reserve(params.maxParticles * 12);
        
        for (int i = 0; i < params.maxParticles; i++) {
            if (i < params.numParticles) {
                // Position
                data.push_back(posDist(rng));
                data.push_back(posDist(rng));
                // Velocity
                data.push_back(0.0f);
                data.push_back(0.0f);
                // Energy
                data.push_back(1.0f);
                // Species
                data.push_back(speciesDist(rng));
                // Age
                data.push_back(0.0f);
                // DNA (5 values)
                for (int d = 0; d < 5; d++) {
                    data.push_back(dnaDist(rng));
                }
            } else {
                // Dead/inactive particle
                data.push_back(0.0f);  // x
                data.push_back(0.0f);  // y
                data.push_back(0.0f);  // vx
                data.push_back(0.0f);  // vy
                data.push_back(0.0f);  // energy = 0 means dead
                data.push_back(0.0f);  // species
                data.push_back(0.0f);  // age
                for (int d = 0; d < 5; d++) {
                    data.push_back(0.0f);
                }
            }
        }
        
        particleBufferA.setData(data);
        particleBufferB.setData(data);
        aliveCount = params.numParticles;
    }
    
    void step() {
        Buffer& readBuffer = useBufferA ? particleBufferA : particleBufferB;
        Buffer& writeBuffer = useBufferA ? particleBufferB : particleBufferA;
        
        stepShader.use();
        
        // Bind buffers
        stepShader.bindBuffer("ParticlesIn", readBuffer, 0);
        stepShader.bindBuffer("ParticlesOut", writeBuffer, 1);
        
        // Bind uniforms
        stepShader.setUniform("u_NumParticles", params.maxParticles);
        stepShader.setUniform("u_AliveCount", aliveCount);
        stepShader.setUniform("u_WorldWidth", params.worldWidth);
        stepShader.setUniform("u_WorldHeight", params.worldHeight);
        stepShader.setUniform("u_Wk", params.w_k);
        stepShader.setUniform("u_MuK", params.mu_k);
        stepShader.setUniform("u_SigmaK2", params.sigma_k2);
        stepShader.setUniform("u_MuG", params.mu_g);
        stepShader.setUniform("u_SigmaG2", params.sigma_g2);
        stepShader.setUniform("u_Crep", params.c_rep);
        stepShader.setUniform("u_Dt", params.dt);
        stepShader.setUniform("u_H", params.h);
        stepShader.setUniform("u_EvolutionEnabled", params.evolutionEnabled);
        stepShader.setUniform("u_BirthRate", params.birthRate);
        stepShader.setUniform("u_DeathRate", params.deathRate);
        stepShader.setUniform("u_MutationRate", params.mutationRate);
        stepShader.setUniform("u_EnergyDecay", params.energyDecay);
        stepShader.setUniform("u_EnergyFromGrowth", params.energyFromGrowth);
        
        // Random seed for evolution
        static int frame = 0;
        stepShader.setUniform("u_RandomSeed", frame++);
        
        // Dispatch compute shader
        int workGroups = (params.maxParticles + 255) / 256;
        stepShader.dispatch(workGroups, 1, 1);
        stepShader.wait();
        
        useBufferA = !useBufferA;
    }
    
    void display(int windowWidth, int windowHeight) {
        Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
        
        displayShader.use();
        displayShader.bindBuffer("Particles", activeBuffer, 0);
        
        displayShader.setUniform("u_NumParticles", params.maxParticles);
        displayShader.setUniform("u_WorldWidth", params.worldWidth);
        displayShader.setUniform("u_WorldHeight", params.worldHeight);
        displayShader.setUniform("u_TranslateX", params.translateX);
        displayShader.setUniform("u_TranslateY", params.translateY);
        displayShader.setUniform("u_Zoom", params.zoom);
        displayShader.setUniform("u_WindowWidth", static_cast<float>(windowWidth));
        displayShader.setUniform("u_WindowHeight", static_cast<float>(windowHeight));
        displayShader.setUniform("u_Wk", params.w_k);
        displayShader.setUniform("u_MuK", params.mu_k);
        displayShader.setUniform("u_SigmaK2", params.sigma_k2);
        displayShader.setUniform("u_MuG", params.mu_g);
        displayShader.setUniform("u_SigmaG2", params.sigma_g2);
        displayShader.setUniform("u_ShowFields", params.showFields);
        displayShader.setUniform("u_FieldType", params.fieldType);
        
        displayShader.render();
    }
    
    void updateStats() {
        Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
        std::vector<float> data = activeBuffer.getData();
        
        aliveCount = 0;
        float totalEnergy = 0.0f;
        float totalAge = 0.0f;
        
        for (int i = 0; i < params.maxParticles; i++) {
            int base = i * 12;
            float energy = data[base + 4];
            float age = data[base + 6];
            
            if (energy > 0.01f) {
                aliveCount++;
                totalEnergy += energy;
                totalAge += age;
            }
        }
        
        avgEnergy = aliveCount > 0 ? totalEnergy / aliveCount : 0.0f;
        avgAge = aliveCount > 0 ? totalAge / aliveCount : 0.0f;
    }
    
    void addParticle(float x, float y) {
        Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
        std::vector<float> data = activeBuffer.getData();
        
        // Find a dead slot
        for (int i = 0; i < params.maxParticles; i++) {
            int base = i * 12;
            if (data[base + 4] < 0.01f) { // Dead particle
                std::uniform_real_distribution<float> speciesDist(0.0f, 3.0f);
                std::uniform_real_distribution<float> dnaDist(-0.2f, 0.2f);
                
                data[base + 0] = x;  // x
                data[base + 1] = y;  // y
                data[base + 2] = 0.0f;  // vx
                data[base + 3] = 0.0f;  // vy
                data[base + 4] = 1.0f;  // energy
                data[base + 5] = speciesDist(rng);  // species
                data[base + 6] = 0.0f;  // age
                for (int d = 0; d < 5; d++) {
                    data[base + 7 + d] = dnaDist(rng);
                }
                
                activeBuffer.setData(data);
                // Also update the other buffer
                Buffer& otherBuffer = useBufferA ? particleBufferB : particleBufferA;
                otherBuffer.setData(data);
                
                aliveCount++;
                break;
            }
        }
    }
};

// Global simulation
ParticleLeniaSimulation simulation;
bool paused = false;

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    WINDOW_WIDTH = width;
    WINDOW_HEIGHT = height;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

ImVec2 screenToWorld(float screenX, float screenY) {
    float worldX = (screenX / WINDOW_WIDTH - 0.5f) * 2.0f * simulation.params.worldWidth / simulation.params.zoom + simulation.params.translateX;
    float worldY = ((1.0f - screenY / WINDOW_HEIGHT) - 0.5f) * 2.0f * simulation.params.worldHeight / simulation.params.zoom + simulation.params.translateY;
    return ImVec2(worldX, worldY);
}

void renderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 700), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Particle Lenia - Evolution");
    
    // Controls
    ImGui::Text("Press 'A' + hover to add particles");
    ImGui::Text("Middle mouse to pan");
    ImGui::Text("Scroll to zoom");
    
    ImGui::Separator();
    
    // Playback
    if (ImGui::Button(paused ? "Play" : "Pause")) {
        paused = !paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        simulation.resetParticles();
    }
    
    ImGui::SliderInt("Steps/Frame", &simulation.params.stepsPerFrame, 1, 50);
    
    ImGui::Separator();
    
    // Statistics
    ImGui::Text("Statistics:");
    ImGui::Text("  Alive: %d / %d", simulation.aliveCount, simulation.params.maxParticles);
    ImGui::Text("  Avg Energy: %.3f", simulation.avgEnergy);
    ImGui::Text("  Avg Age: %.1f", simulation.avgAge);
    
    ImGui::Separator();
    
    // World Settings
    if (ImGui::CollapsingHeader("World", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("World Width", &simulation.params.worldWidth, 10.0f, 100.0f);
        ImGui::SliderFloat("World Height", &simulation.params.worldHeight, 10.0f, 100.0f);
        
        int numParticles = simulation.params.numParticles;
        if (ImGui::SliderInt("Initial Particles", &numParticles, 10, 1000)) {
            simulation.params.numParticles = numParticles;
        }
    }
    
    // Kernel Settings
    if (ImGui::CollapsingHeader("Kernel (Sensing)")) {
        ImGui::SliderFloat("w_k (weight)", &simulation.params.w_k, 0.001f, 0.1f);
        ImGui::SliderFloat("mu_k (peak dist)", &simulation.params.mu_k, 0.5f, 20.0f);
        ImGui::SliderFloat("sigma_k^2 (width)", &simulation.params.sigma_k2, 0.1f, 10.0f);
    }
    
    // Growth Settings
    if (ImGui::CollapsingHeader("Growth (Lenia)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("mu_g (optimal density)", &simulation.params.mu_g, 0.0f, 2.0f);
        ImGui::SliderFloat("sigma_g^2 (tolerance)", &simulation.params.sigma_g2, 0.001f, 0.5f);
    }
    
    // Repulsion
    if (ImGui::CollapsingHeader("Repulsion")) {
        ImGui::SliderFloat("c_rep (strength)", &simulation.params.c_rep, 0.0f, 5.0f);
    }
    
    // Time Settings
    if (ImGui::CollapsingHeader("Time")) {
        ImGui::SliderFloat("dt (time step)", &simulation.params.dt, 0.01f, 0.5f);
        ImGui::SliderFloat("h (gradient step)", &simulation.params.h, 0.001f, 0.1f);
    }
    
    // Evolution Settings
    if (ImGui::CollapsingHeader("Evolution", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Evolution", &simulation.params.evolutionEnabled);
        
        if (simulation.params.evolutionEnabled) {
            ImGui::SliderFloat("Birth Rate", &simulation.params.birthRate, 0.0f, 0.01f, "%.5f");
            ImGui::SliderFloat("Death Rate", &simulation.params.deathRate, 0.0f, 0.01f, "%.5f");
            ImGui::SliderFloat("Mutation Rate", &simulation.params.mutationRate, 0.0f, 0.5f);
            ImGui::SliderFloat("Energy Decay", &simulation.params.energyDecay, 0.0f, 0.01f, "%.5f");
            ImGui::SliderFloat("Energy from Growth", &simulation.params.energyFromGrowth, 0.0f, 0.1f);
        }
    }
    
    // Display Settings
    if (ImGui::CollapsingHeader("Display")) {
        ImGui::Checkbox("Show Fields", &simulation.params.showFields);
        const char* fieldTypes[] = {"None", "U (Density)", "R (Repulsion)", "G (Growth)", "E (Energy)"};
        ImGui::Combo("Field Type", &simulation.params.fieldType, fieldTypes, 5);
        ImGui::SliderFloat("Zoom", &simulation.params.zoom, 0.1f, 5.0f);
    }
    
    ImGui::Separator();
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("ms/step: %.3f", (1000.0f / ImGui::GetIO().Framerate) / simulation.params.stepsPerFrame);
    
    ImGui::End();
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Chronos - Particle Lenia Evolution", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSwapInterval(1); // VSync
    
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
    
    // Initialize simulation
    simulation.init();
    
    // Pan tracking
    static ImVec2 panStart;
    static float panStartX, panStartY;
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        glfwPollEvents();
        
        // Handle input
        if (!io.WantCaptureMouse) {
            // Add particles with 'A' key
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                double mx, my;
                glfwGetCursorPos(window, &mx, &my);
                ImVec2 worldPos = screenToWorld(mx, my);
                simulation.addParticle(worldPos.x, worldPos.y);
            }
            
            // Pan with middle mouse
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                panStart = ImGui::GetMousePos();
                panStartX = simulation.params.translateX;
                panStartY = simulation.params.translateY;
            }
            if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                ImVec2 pos = ImGui::GetMousePos();
                float dx = (pos.x - panStart.x) / WINDOW_WIDTH * simulation.params.worldWidth * 2.0f / simulation.params.zoom;
                float dy = (pos.y - panStart.y) / WINDOW_HEIGHT * simulation.params.worldHeight * 2.0f / simulation.params.zoom;
                simulation.params.translateX = panStartX - dx;
                simulation.params.translateY = panStartY + dy;
            }
            
            // Zoom with scroll
            float scroll = io.MouseWheel;
            if (scroll != 0) {
                simulation.params.zoom *= (1.0f + scroll * 0.1f);
                simulation.params.zoom = std::max(0.1f, std::min(10.0f, simulation.params.zoom));
            }
        }
        
        // Update simulation
        if (!paused) {
            for (int i = 0; i < simulation.params.stepsPerFrame; i++) {
                simulation.step();
            }
            
            // Update stats every 10 frames
            static int frameCount = 0;
            if (++frameCount % 10 == 0) {
                simulation.updateStats();
            }
        }
        
        // Render
        glClearColor(0.0f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        simulation.display(WINDOW_WIDTH, WINDOW_HEIGHT);
        
        renderUI();
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
