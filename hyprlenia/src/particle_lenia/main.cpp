 

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <omp.h>

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>
#include <sstream>

#include "core/Buffer.h"
#include "core/ComputeShader.h"
#include "core/RenderShader.h"


int WINDOW_WIDTH = 1200;
int WINDOW_HEIGHT = 900;


struct SimulationParams {
  
  float worldWidth = 40.0f;
  float worldHeight = 40.0f;
  float worldDepth = 40.0f;

  
  int numParticles = 500;
  int maxParticles = 2000;

  
  float w_k = 0.022f;     
  float mu_k = 4.0f;      
  float sigma_k2 = 1.0f;  

  
  float mu_g = 0.6f;         
  float sigma_g2 = 0.0225f;  

  
  float c_rep = 1.0f;  

  
  float dt = 0.1f;  
  float h = 0.01f;  

  
  bool evolutionEnabled = false;
  float birthRate = 0.001f;        
  float deathRate = 0.0f;          
  float mutationRate = 0.1f;       
  float energyDecay = 0.0f;        
  float energyFromGrowth = 0.01f;  

  
  float translateX = 0.0f;
  float translateY = 0.0f;
  float translateZ = 0.0f;
  float zoom = 1.0f;

  
  int stepsPerFrame = 5;
  bool showFields = true;
  int fieldType = 3;  

  
  bool foodEnabled = true;
  float foodSpawnRate =
      0.002f;  
  float foodDecayRate = 0.001f;        
  float foodMaxAmount = 1.0f;          
  float foodConsumptionRadius = 2.0f;  
  bool showFood = true;                

  
  bool view3D = true;            
  float cameraAngle = 45.0f;     
  float cameraRotation = 0.0f;   
  float cameraDistance = 60.0f;  
  float heightScale = 10.0f;     
  float glowIntensity = 1.5f;    
  bool showWireframe = false;    
  float ambientLight = 0.5f;     
  float particleSize = 20.0f;    

  
  int interactionMode =
      0;  
  float brushRadius = 5.0f;
  float forceStrength = 0.5f;

  
  int goalMode = 0;  
  float goalStrength = 0.1f;
  char goalImagePath[256] = "goal.bmp";

  
  bool showGoal = false;

  
  bool sonificationEnabled = false;
  float audioVolume = 0.3f;
  float minFrequency = 80.0f;    
  float maxFrequency = 800.0f;   
  int maxVoices = 32;            
};


struct Particle {
  float x, y, z;     
  float vx, vy, vz;  
  float energy;      
  float species;     
  float age;         
  float dna[5];  
                 
  float potential;   
};

constexpr int PARTICLE_FLOATS = 15;  









struct AudioVoice {
  float frequency = 220.0f;   
  float amplitude = 0.0f;     
  float phase = 0.0f;         
  float targetFreq = 220.0f;  
  float targetAmp = 0.0f;     
};

struct AudioState {
  ma_device device;
  bool initialized = false;
  bool enabled = false;

  static constexpr int MAX_VOICES = 64;
  AudioVoice voices[MAX_VOICES];
  int numVoices = 32;

  float masterVolume = 0.3f;
  float minFreq = 80.0f;
  float maxFreq = 800.0f;

  std::mutex voiceMutex;
  std::atomic<bool> running{false};
};

AudioState g_audio;


void audioCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
  (void)pInput;

  float* output = static_cast<float*>(pOutput);

  if (!g_audio.running || !g_audio.enabled) {
    memset(output, 0, frameCount * sizeof(float));
    return;
  }

  const float sampleRate = static_cast<float>(pDevice->sampleRate);
  const float smoothing = 0.995f;  

  for (ma_uint32 i = 0; i < frameCount; i++) {
    float sample = 0.0f;

    
    for (int v = 0; v < g_audio.numVoices; v++) {
      AudioVoice& voice = g_audio.voices[v];

      
      voice.frequency = voice.frequency * smoothing + voice.targetFreq * (1.0f - smoothing);
      voice.amplitude = voice.amplitude * smoothing + voice.targetAmp * (1.0f - smoothing);

      if (voice.amplitude > 0.001f) {
        
        float sine = std::sin(voice.phase * 2.0f * 3.14159265f);
        sample += sine * voice.amplitude;

        
        voice.phase += voice.frequency / sampleRate;
        if (voice.phase >= 1.0f) voice.phase -= 1.0f;
      }
    }

    
    sample *= g_audio.masterVolume / static_cast<float>(std::max(1, g_audio.numVoices / 4));
    sample = std::tanh(sample);  

    output[i] = sample;
  }
}

void initAudio() {
  ma_device_config config = ma_device_config_init(ma_device_type_playback);
  config.playback.format = ma_format_f32;
  config.playback.channels = 1;
  config.sampleRate = 44100;
  config.dataCallback = audioCallback;

  if (ma_device_init(nullptr, &config, &g_audio.device) != MA_SUCCESS) {
    std::cerr << "Failed to initialize audio device" << std::endl;
    return;
  }

  g_audio.initialized = true;
  g_audio.running = true;

  if (ma_device_start(&g_audio.device) != MA_SUCCESS) {
    std::cerr << "Failed to start audio device" << std::endl;
    ma_device_uninit(&g_audio.device);
    g_audio.initialized = false;
    return;
  }

  std::cout << "Audio initialized: " << g_audio.device.sampleRate << " Hz" << std::endl;
}

void shutdownAudio() {
  if (g_audio.initialized) {
    g_audio.running = false;
    ma_device_uninit(&g_audio.device);
    g_audio.initialized = false;
  }
}


void updateAudioFromParticles(const std::vector<float>& particleData, int maxParticles,
                               float minFreq, float maxFreq, float volume) {
  if (!g_audio.initialized) return;

  g_audio.minFreq = minFreq;
  g_audio.maxFreq = maxFreq;
  g_audio.masterVolume = volume;

  
  struct ParticleScore {
    int index;
    float score;
  };
  std::vector<ParticleScore> scores;
  scores.reserve(maxParticles);

  for (int i = 0; i < maxParticles; i++) {
    int base = i * PARTICLE_FLOATS;
    float energy = particleData[base + 6];
    if (energy < 0.01f) continue;

    
    float vx = particleData[base + 3];
    float vy = particleData[base + 4];
    float vz = particleData[base + 5];
    float speed = std::sqrt(vx*vx + vy*vy + vz*vz);

    
    float potential = particleData[base + 14];

    
    float score = energy * (1.0f + speed * 0.5f + potential * 0.3f);
    scores.push_back({i, score});
  }

  
  std::sort(scores.begin(), scores.end(),
            [](const ParticleScore& a, const ParticleScore& b) { return a.score > b.score; });

  
  int numVoices = std::min(g_audio.numVoices, static_cast<int>(scores.size()));

  for (int v = 0; v < AudioState::MAX_VOICES; v++) {
    if (v < numVoices) {
      int idx = scores[v].index;
      int base = idx * PARTICLE_FLOATS;

      float energy = particleData[base + 6];
      float vx = particleData[base + 3];
      float vy = particleData[base + 4];
      float vz = particleData[base + 5];
      float speed = std::sqrt(vx*vx + vy*vy + vz*vz);
      float potential = particleData[base + 14];

      
      
      float t = std::clamp(potential / 2.0f, 0.0f, 1.0f);
      float freq = minFreq * std::pow(maxFreq / minFreq, t);

      
      float amp = std::clamp(speed * 2.0f, 0.0f, 1.0f) * energy;

      g_audio.voices[v].targetFreq = freq;
      g_audio.voices[v].targetAmp = amp;
    } else {
      
      g_audio.voices[v].targetAmp = 0.0f;
    }
  }
}



class ParticleLeniaSimulation {
 public:
  SimulationParams params;

  Buffer particleBufferA;
  Buffer particleBufferB;
  bool useBufferA = true;

  ComputeShader stepShader;
  RenderShader displayShader;

  
  ComputeShader heightmapShader;
  RenderShader terrainShader;
  RenderShader particle3DShader;
  GLuint heightmapTexture = 0;
  GLuint terrainVAO = 0;
  GLuint terrainVBO = 0;
  GLuint terrainEBO = 0;
  GLuint particleVAO = 0;  
  int terrainGridSize = 128;  
  int terrainIndexCount = 0;

  
  ComputeShader foodUpdateShader;
  GLuint foodTexture = 0;
  int foodGridSize = 128;  

  
  GLuint goalTexture = 0;
  int goalGridSize = 512;

  std::mt19937 rng;

  
  int aliveCount = 0;
  float avgEnergy = 0.0f;
  float avgAge = 0.0f;
  
  
  std::vector<float> historyAlive;
  std::vector<float> historyEnergy;
  const size_t historyMaxSize = 300;

  void init() {
    
    rng = std::mt19937(std::random_device{}());

    
    int bufferSize = params.maxParticles * PARTICLE_FLOATS;

    particleBufferA = Buffer(bufferSize, GL_SHADER_STORAGE_BUFFER);
    particleBufferB = Buffer(bufferSize, GL_SHADER_STORAGE_BUFFER);

    particleBufferA.init();
    particleBufferB.init();

    
    resetParticles();

    
    stepShader = ComputeShader("shaders/particle_lenia_step.comp");
    stepShader.init();

    displayShader = RenderShader("shaders/passthrough.vert",
                                 "shaders/particle_lenia_display.frag");
    displayShader.init();

    
    init3D();

    
    initFood();

    
    initGoal();
  }

  void initGoal() {
    glGenTextures(1, &goalTexture);
    glBindTexture(GL_TEXTURE_2D, goalTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, goalGridSize, goalGridSize, 0,
                 GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    updateGoalTexture();
  }

  bool loadBMP(const char* filename, std::vector<float>& outData, int size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      std::cout << "Failed to open image: " << filename << std::endl;
      return false;
    }
    unsigned char header[54];
    if (!file.read(reinterpret_cast<char*>(header), 54)) return false;
    if (header[0] != 'B' || header[1] != 'M') return false;
    int width = *(int*)&header[18];
    int height = *(int*)&header[22];
    int imageSize = *(int*)&header[34];
    if (imageSize == 0) imageSize = width * height * 3;
    int dataPos = *(int*)&header[10];
    if (dataPos == 0) dataPos = 54;
    std::vector<unsigned char> img(imageSize);
    file.seekg(dataPos);
    file.read(reinterpret_cast<char*>(img.data()), imageSize);
    file.close();

    outData.resize(size * size);

#pragma omp parallel for collapse(2)
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        int srcX = x * width / size;
        int srcY = (size - 1 - y) * height / size;  
        if (srcX >= width) srcX = width - 1;
        if (srcY >= height) srcY = height - 1;
        int idx = (srcY * width + srcX) * 3;
        if (idx < imageSize - 2) {
          float val =
              (img[idx + 2] + img[idx + 1] + img[idx]) / (3.0f * 255.0f);
          outData[y * size + x] = val;
        }
      }
    }
    return true;
  }

  void updateGoalTexture() {
    std::vector<float> data(goalGridSize * goalGridSize, 0.0f);

    if (params.goalMode == 1) {  
      float cx = goalGridSize / 2.0f;
      float cy = goalGridSize / 2.0f;
      float r = goalGridSize * 0.3f;
      float thickness = goalGridSize * 0.05f;

#pragma omp parallel for collapse(2)
      for (int y = 0; y < goalGridSize; y++) {
        for (int x = 0; x < goalGridSize; x++) {
          float dx = x - cx;
          float dy = y - cy;
          float dist = sqrt(dx * dx + dy * dy);
          float val = exp(-pow(dist - r, 2) / (2.0f * thickness * thickness));
          data[y * goalGridSize + x] = val;
        }
      }
    } else if (params.goalMode == 2) {  
      float margin = goalGridSize * 0.2f;
#pragma omp parallel for collapse(2)
      for (int y = 0; y < goalGridSize; y++) {
        for (int x = 0; x < goalGridSize; x++) {
          if (x > margin && x < goalGridSize - margin && y > margin &&
              y < goalGridSize - margin) {
            float dx =
                std::min(std::min(x - margin, goalGridSize - margin - x),
                         std::min(y - margin, goalGridSize - margin - y));

            if (dx < 20.0f) data[y * goalGridSize + x] = 1.0f;
          }
        }
      }
    } else if (params.goalMode == 3) {  
      
      int w = goalGridSize;
      auto drawRect = [&](int x, int y, int rw, int rh) {
        for (int iy = y; iy < y + rh; iy++) {
          for (int ix = x; ix < x + rw; ix++) {
            if (ix >= 0 && ix < w && iy >= 0 && iy < w)
              data[iy * w + ix] = 1.0f;
          }
        }
      };

      int s = w / 10;  
      int thick = s / 2;
      
      drawRect(2 * s, 3 * s, thick, 4 * s);
      drawRect(4 * s, 3 * s, thick, 4 * s);
      drawRect(2 * s, 5 * s, 2 * s + thick, thick);
      
      drawRect(6 * s, 3 * s, thick, 4 * s);
    } else if (params.goalMode == 4) {  
      if (!loadBMP(params.goalImagePath, data, goalGridSize)) {

#pragma omp parallel for collapse(2)
        for (int y = 0; y < goalGridSize; y++) {
          for (int x = 0; x < goalGridSize; x++) {
            if (abs(x - y) < 20 || abs(x - (goalGridSize - y)) < 20)
              data[y * goalGridSize + x] = 1.0f;
          }
        }
      }
    }

    glBindTexture(GL_TEXTURE_2D, goalTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, goalGridSize, goalGridSize, GL_RED,
                    GL_FLOAT, data.data());
  }

  void saveScene(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to save scene: " << filename << std::endl;
        return;
    }
    
    out << "worldWidth=" << params.worldWidth << "\n";
    out << "worldHeight=" << params.worldHeight << "\n";
    out << "worldDepth=" << params.worldDepth << "\n";
    out << "numParticles=" << params.numParticles << "\n";
    out << "maxParticles=" << params.maxParticles << "\n";
    out << "w_k=" << params.w_k << "\n";
    out << "mu_k=" << params.mu_k << "\n";
    out << "sigma_k2=" << params.sigma_k2 << "\n";
    out << "mu_g=" << params.mu_g << "\n";
    out << "sigma_g2=" << params.sigma_g2 << "\n";
    out << "c_rep=" << params.c_rep << "\n";
    out << "dt=" << params.dt << "\n";
    out << "h=" << params.h << "\n";
    out << "evolutionEnabled=" << params.evolutionEnabled << "\n";
    out << "birthRate=" << params.birthRate << "\n";
    out << "deathRate=" << params.deathRate << "\n";
    out << "mutationRate=" << params.mutationRate << "\n";
    out << "energyDecay=" << params.energyDecay << "\n";
    out << "energyFromGrowth=" << params.energyFromGrowth << "\n";
    out << "translateX=" << params.translateX << "\n";
    out << "translateY=" << params.translateY << "\n";
    out << "translateZ=" << params.translateZ << "\n";
    out << "zoom=" << params.zoom << "\n";
    out << "stepsPerFrame=" << params.stepsPerFrame << "\n";
    out << "showFields=" << params.showFields << "\n";
    out << "fieldType=" << params.fieldType << "\n";
    out << "foodEnabled=" << params.foodEnabled << "\n";
    out << "foodSpawnRate=" << params.foodSpawnRate << "\n";
    out << "foodDecayRate=" << params.foodDecayRate << "\n";
    out << "foodMaxAmount=" << params.foodMaxAmount << "\n";
    out << "foodConsumptionRadius=" << params.foodConsumptionRadius << "\n";
    out << "showFood=" << params.showFood << "\n";
    out << "view3D=" << params.view3D << "\n";
    out << "cameraAngle=" << params.cameraAngle << "\n";
    out << "cameraRotation=" << params.cameraRotation << "\n";
    out << "cameraDistance=" << params.cameraDistance << "\n";
    out << "heightScale=" << params.heightScale << "\n";
    out << "glowIntensity=" << params.glowIntensity << "\n";
    out << "showWireframe=" << params.showWireframe << "\n";
    out << "ambientLight=" << params.ambientLight << "\n";
    out << "particleSize=" << params.particleSize << "\n";
    out << "interactionMode=" << params.interactionMode << "\n";
    out << "brushRadius=" << params.brushRadius << "\n";
    out << "forceStrength=" << params.forceStrength << "\n";
    out << "goalMode=" << params.goalMode << "\n";
    out << "goalStrength=" << params.goalStrength << "\n";
    out << "showGoal=" << params.showGoal << "\n";
    out << "goalImagePath=" << params.goalImagePath << "\n";
    
    std::cout << "Scene saved to " << filename << std::endl;
  }

  void loadScene(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Failed to load scene: " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;
        
        std::string key = line.substr(0, eqPos);
        std::string val = line.substr(eqPos + 1);
        
        try {
            if (key == "worldWidth") params.worldWidth = std::stof(val);
            else if (key == "worldHeight") params.worldHeight = std::stof(val);
            else if (key == "worldDepth") params.worldDepth = std::stof(val);
            else if (key == "numParticles") params.numParticles = std::stoi(val);
            else if (key == "maxParticles") params.maxParticles = std::stoi(val);
            else if (key == "w_k") params.w_k = std::stof(val);
            else if (key == "mu_k") params.mu_k = std::stof(val);
            else if (key == "sigma_k2") params.sigma_k2 = std::stof(val);
            else if (key == "mu_g") params.mu_g = std::stof(val);
            else if (key == "sigma_g2") params.sigma_g2 = std::stof(val);
            else if (key == "c_rep") params.c_rep = std::stof(val);
            else if (key == "dt") params.dt = std::stof(val);
            else if (key == "h") params.h = std::stof(val);
            else if (key == "evolutionEnabled") params.evolutionEnabled = std::stoi(val);
            else if (key == "birthRate") params.birthRate = std::stof(val);
            else if (key == "deathRate") params.deathRate = std::stof(val);
            else if (key == "mutationRate") params.mutationRate = std::stof(val);
            else if (key == "energyDecay") params.energyDecay = std::stof(val);
            else if (key == "energyFromGrowth") params.energyFromGrowth = std::stof(val);
            else if (key == "translateX") params.translateX = std::stof(val);
            else if (key == "translateY") params.translateY = std::stof(val);
            else if (key == "translateZ") params.translateZ = std::stof(val);
            else if (key == "zoom") params.zoom = std::stof(val);
            else if (key == "stepsPerFrame") params.stepsPerFrame = std::stoi(val);
            else if (key == "showFields") params.showFields = std::stoi(val);
            else if (key == "fieldType") params.fieldType = std::stoi(val);
            else if (key == "foodEnabled") params.foodEnabled = std::stoi(val);
            else if (key == "foodSpawnRate") params.foodSpawnRate = std::stof(val);
            else if (key == "foodDecayRate") params.foodDecayRate = std::stof(val);
            else if (key == "foodMaxAmount") params.foodMaxAmount = std::stof(val);
            else if (key == "foodConsumptionRadius") params.foodConsumptionRadius = std::stof(val);
            else if (key == "showFood") params.showFood = std::stoi(val);
            else if (key == "view3D") params.view3D = std::stoi(val);
            else if (key == "cameraAngle") params.cameraAngle = std::stof(val);
            else if (key == "cameraRotation") params.cameraRotation = std::stof(val);
            else if (key == "cameraDistance") params.cameraDistance = std::stof(val);
            else if (key == "heightScale") params.heightScale = std::stof(val);
            else if (key == "glowIntensity") params.glowIntensity = std::stof(val);
            else if (key == "showWireframe") params.showWireframe = std::stoi(val);
            else if (key == "ambientLight") params.ambientLight = std::stof(val);
            else if (key == "particleSize") params.particleSize = std::stof(val);
            else if (key == "interactionMode") params.interactionMode = std::stoi(val);
            else if (key == "brushRadius") params.brushRadius = std::stof(val);
            else if (key == "forceStrength") params.forceStrength = std::stof(val);
            else if (key == "goalMode") params.goalMode = std::stoi(val);
            else if (key == "goalStrength") params.goalStrength = std::stof(val);
            else if (key == "showGoal") params.showGoal = std::stoi(val);
            else if (key == "goalImagePath") {
                if (val.length() < 256) strncpy(params.goalImagePath, val.c_str(), 255);
            }
        } catch (...) {
        }
    }
    std::cout << "Scene loaded from " << filename << std::endl;
    
    init();
  }

  void initFood() {
    
    foodUpdateShader = ComputeShader("shaders/food_update.comp");
    foodUpdateShader.init();

    
    glGenTextures(1, &foodTexture);
    glBindTexture(GL_TEXTURE_2D, foodTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, foodGridSize, foodGridSize, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    
    std::vector<float> foodData(foodGridSize * foodGridSize * 4, 0.0f);
    int totalCells = foodGridSize * foodGridSize;

#pragma omp parallel
    {
      std::mt19937 localRng(std::random_device{}() + omp_get_thread_num());
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);

#pragma omp for
      for (int i = 0; i < totalCells; i++) {
        if (dist(localRng) < 0.1f) {  
          foodData[i * 4 + 0] = dist(localRng) * 0.5f;  
          foodData[i * 4 + 1] = 1.0f;                   
        }
      }
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, foodGridSize, foodGridSize, GL_RGBA,
                    GL_FLOAT, foodData.data());
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  void init3D() {
    
    heightmapShader = ComputeShader("shaders/terrain_heightmap.comp");
    heightmapShader.init();

    terrainShader =
        RenderShader("shaders/terrain.vert", "shaders/terrain.frag");
    terrainShader.init();

    particle3DShader =
        RenderShader("shaders/particle3d.vert", "shaders/particle3d.frag");
    particle3DShader.init();

    
    glGenTextures(1, &heightmapTexture);
    glBindTexture(GL_TEXTURE_2D, heightmapTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, terrainGridSize, terrainGridSize,
                 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    for (int y = 0; y < terrainGridSize; y++) {
      for (int x = 0; x < terrainGridSize; x++) {
        float u = static_cast<float>(x) / (terrainGridSize - 1);
        float v = static_cast<float>(y) / (terrainGridSize - 1);
        vertices.push_back(u);
        vertices.push_back(v);
      }
    }

    for (int y = 0; y < terrainGridSize - 1; y++) {
      for (int x = 0; x < terrainGridSize - 1; x++) {
        int topLeft = y * terrainGridSize + x;
        int topRight = topLeft + 1;
        int bottomLeft = (y + 1) * terrainGridSize + x;
        int bottomRight = bottomLeft + 1;

        indices.push_back(topLeft);
        indices.push_back(bottomLeft);
        indices.push_back(topRight);
        indices.push_back(topRight);
        indices.push_back(bottomLeft);
        indices.push_back(bottomRight);
      }
    }

    terrainIndexCount = static_cast<int>(indices.size());

    glGenVertexArrays(1, &terrainVAO);
    glGenBuffers(1, &terrainVBO);
    glGenBuffers(1, &terrainEBO);

    glBindVertexArray(terrainVAO);

    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                 vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                 indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    
    glGenVertexArrays(1, &particleVAO);
  }

  void resetParticles() {
    
    std::vector<float> data(params.maxParticles * PARTICLE_FLOATS);


#pragma omp parallel
    {
      
      std::mt19937 localRng(std::random_device{}() + omp_get_thread_num());
      std::uniform_real_distribution<float> posDistX(-params.worldWidth / 2.0f,
                                                     params.worldWidth / 2.0f);
      std::uniform_real_distribution<float> posDistY(-params.worldHeight / 2.0f,
                                                     params.worldHeight / 2.0f);
      std::uniform_real_distribution<float> posDistZ(-params.worldDepth / 2.0f,
                                                     params.worldDepth / 2.0f);
      std::uniform_real_distribution<float> speciesDist(0.0f, 3.0f);
      std::uniform_real_distribution<float> dnaDist(-0.2f, 0.2f);

#pragma omp for
      for (int i = 0; i < params.maxParticles; i++) {
        int base = i * PARTICLE_FLOATS;
        if (i < params.numParticles) {
          
          data[base + 0] = posDistX(localRng);
          data[base + 1] = posDistY(localRng);
          data[base + 2] = posDistZ(localRng);
          
          data[base + 3] = 0.0f;
          data[base + 4] = 0.0f;
          data[base + 5] = 0.0f;
          
          data[base + 6] = 1.0f;
          
          data[base + 7] = speciesDist(localRng);
          
          data[base + 8] = 0.0f;
          
          for (int d = 0; d < 5; d++) {
            data[base + 9 + d] = dnaDist(localRng);
          }
        } else {
          
          for (int j = 0; j < PARTICLE_FLOATS; j++) {
            data[base + j] = 0.0f;
          }
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

    
    if (params.foodEnabled) {
      foodUpdateShader.use();

      
      glBindImageTexture(0, foodTexture, 0, GL_FALSE, 0, GL_READ_WRITE,
                         GL_RGBA16F);

      
      static int foodFrame = 0;
      foodUpdateShader.setUniform("u_FoodGridSize", foodGridSize);
      foodUpdateShader.setUniform("u_FoodSpawnRate", params.foodSpawnRate);
      foodUpdateShader.setUniform("u_FoodDecayRate", params.foodDecayRate);
      foodUpdateShader.setUniform("u_FoodMaxAmount", params.foodMaxAmount);
      foodUpdateShader.setUniform("u_RandomSeed", foodFrame++);

      
      int foodWorkGroupsX = (foodGridSize + 15) / 16;
      int foodWorkGroupsY = (foodGridSize + 15) / 16;
      foodUpdateShader.dispatch(foodWorkGroupsX, foodWorkGroupsY, 1);
      foodUpdateShader.wait();
    }

    
    stepShader.use();

    
    stepShader.bindBuffer("ParticlesIn", readBuffer, 0);
    stepShader.bindBuffer("ParticlesOut", writeBuffer, 1);

    
    if (params.foodEnabled) {
      glBindImageTexture(0, foodTexture, 0, GL_FALSE, 0, GL_READ_WRITE,
                         GL_RGBA16F);
    }

    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, goalTexture);
    stepShader.setUniform("u_GoalTexture", 1);
    stepShader.setUniform("u_GoalMode", params.goalMode);
    stepShader.setUniform("u_GoalStrength", params.goalStrength);

    
    stepShader.setUniform("u_NumParticles", params.maxParticles);
    stepShader.setUniform("u_AliveCount", aliveCount);
    stepShader.setUniform("u_WorldWidth", params.worldWidth);
    stepShader.setUniform("u_WorldHeight", params.worldHeight);
    stepShader.setUniform("u_WorldDepth", params.worldDepth);
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

    
    stepShader.setUniform("u_FoodGridSize", foodGridSize);
    stepShader.setUniform("u_FoodConsumptionRadius",
                          params.foodConsumptionRadius);

    
    static int frame = 0;
    stepShader.setUniform("u_RandomSeed", frame++);

    
    int workGroups = (params.maxParticles + 127) / 128;
    stepShader.dispatch(workGroups, 1, 1);
    stepShader.wait();

    useBufferA = !useBufferA;
  }

  void display(int windowWidth, int windowHeight) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;

    displayShader.use();
    displayShader.bindBuffer("Particles", activeBuffer, 0);

    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, foodTexture);
    displayShader.setUniform("u_FoodTexture", 0);

    displayShader.setUniform("u_NumParticles", params.maxParticles);
    displayShader.setUniform("u_WorldWidth", params.worldWidth);
    displayShader.setUniform("u_WorldHeight", params.worldHeight);
    displayShader.setUniform("u_TranslateX", params.translateX);
    displayShader.setUniform("u_TranslateY", params.translateY);
    displayShader.setUniform("u_Zoom", params.zoom);
    displayShader.setUniform("u_WindowWidth", static_cast<float>(windowWidth));
    displayShader.setUniform("u_WindowHeight",
                             static_cast<float>(windowHeight));
    displayShader.setUniform("u_Wk", params.w_k);
    displayShader.setUniform("u_MuK", params.mu_k);
    displayShader.setUniform("u_SigmaK2", params.sigma_k2);
    displayShader.setUniform("u_MuG", params.mu_g);
    displayShader.setUniform("u_SigmaG2", params.sigma_g2);
    displayShader.setUniform("u_ShowFields", params.showFields);
    displayShader.setUniform("u_FieldType", params.fieldType);
    displayShader.setUniform("u_ShowFood", params.showFood);
    displayShader.setUniform("u_FoodGridSize", foodGridSize);

    displayShader.render();
  }

  void display3D(int windowWidth, int windowHeight) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;

    
    float aspect =
        static_cast<float>(windowWidth) / static_cast<float>(windowHeight);

    
    float camRadius = params.cameraDistance;
    float camAngleRad = params.cameraAngle * 3.14159f / 180.0f;
    float camRotRad = params.cameraRotation * 3.14159f / 180.0f;

    float camX = camRadius * std::cos(camRotRad) * std::cos(camAngleRad);
    float camY = camRadius * std::sin(camAngleRad);
    float camZ = camRadius * std::sin(camRotRad) * std::cos(camAngleRad);

    
    float eye[3] = {camX, camY, camZ};
    float target[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};

    
    float fwd[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
    float fwdLen =
        std::sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]);
    fwd[0] /= fwdLen;
    fwd[1] /= fwdLen;
    fwd[2] /= fwdLen;

    float right[3] = {fwd[1] * up[2] - fwd[2] * up[1],
                      fwd[2] * up[0] - fwd[0] * up[2],
                      fwd[0] * up[1] - fwd[1] * up[0]};
    float rightLen = std::sqrt(right[0] * right[0] + right[1] * right[1] +
                               right[2] * right[2]);
    right[0] /= rightLen;
    right[1] /= rightLen;
    right[2] /= rightLen;

    float upVec[3] = {right[1] * fwd[2] - right[2] * fwd[1],
                      right[2] * fwd[0] - right[0] * fwd[2],
                      right[0] * fwd[1] - right[1] * fwd[0]};

    
    float view[16] = {
        right[0],
        upVec[0],
        -fwd[0],
        0.0f,
        right[1],
        upVec[1],
        -fwd[1],
        0.0f,
        right[2],
        upVec[2],
        -fwd[2],
        0.0f,
        -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]),
        -(upVec[0] * eye[0] + upVec[1] * eye[1] + upVec[2] * eye[2]),
        (fwd[0] * eye[0] + fwd[1] * eye[1] + fwd[2] * eye[2]),
        1.0f};

    
    float fov = 60.0f * 3.14159f / 180.0f;
    float nearPlane = 0.1f;
    float farPlane = 500.0f;
    float tanHalfFov = std::tan(fov / 2.0f);

    float proj[16] = {1.0f / (aspect * tanHalfFov),
                      0.0f,
                      0.0f,
                      0.0f,
                      0.0f,
                      1.0f / tanHalfFov,
                      0.0f,
                      0.0f,
                      0.0f,
                      0.0f,
                      -(farPlane + nearPlane) / (farPlane - nearPlane),
                      -1.0f,
                      0.0f,
                      0.0f,
                      -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane),
                      0.0f};

    
    float viewProj[16];
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        viewProj[i * 4 + j] = 0.0f;
        for (int k = 0; k < 4; k++) {
          viewProj[i * 4 + j] += proj[k * 4 + j] * view[i * 4 + k];
        }
      }
    }

    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClear(GL_DEPTH_BUFFER_BIT);

    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);  
    glEnable(GL_PROGRAM_POINT_SIZE);

    particle3DShader.use();
    particle3DShader.bindBuffer("Particles", activeBuffer, 0);

    particle3DShader.setUniformMat4("u_ViewProjection", viewProj);
    particle3DShader.setUniform("u_NumParticles", params.maxParticles);
    particle3DShader.setUniform("u_WorldWidth", params.worldWidth);
    particle3DShader.setUniform("u_WorldHeight", params.worldHeight);
    particle3DShader.setUniform("u_WorldDepth", params.worldDepth);
    particle3DShader.setUniform("u_ParticleSize", params.particleSize);
    particle3DShader.setUniform("u_TranslateX", params.translateX);
    particle3DShader.setUniform("u_TranslateY", params.translateY);
    particle3DShader.setUniform("u_TranslateZ", params.translateZ);
    particle3DShader.setUniform("u_Zoom", params.zoom);
    particle3DShader.setUniform("u_CameraPos", camX, camY, camZ);

    
    glBindVertexArray(particleVAO);  
    glDrawArrays(GL_POINTS, 0, params.maxParticles);
    glBindVertexArray(0);

    glDisable(GL_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
  }

  void updateStats() {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
    std::vector<float> data = activeBuffer.getData();

    int localAliveCount = 0;
    float totalEnergy = 0.0f;
    float totalAge = 0.0f;

#pragma omp parallel for reduction(+ : localAliveCount, totalEnergy, totalAge)
    for (int i = 0; i < params.maxParticles; i++) {
      int base = i * PARTICLE_FLOATS;
      float energy = data[base + 6];  
      float age = data[base + 8];     

      if (energy > 0.01f) {
        localAliveCount++;
        totalEnergy += energy;
        totalAge += age;
      }
    }

    aliveCount = localAliveCount;
    avgEnergy = aliveCount > 0 ? totalEnergy / aliveCount : 0.0f;
    avgAge = aliveCount > 0 ? totalAge / aliveCount : 0.0f;
    
    
    historyAlive.push_back((float)aliveCount);
    if (historyAlive.size() > historyMaxSize) historyAlive.erase(historyAlive.begin());
    
    historyEnergy.push_back(avgEnergy);
    if (historyEnergy.size() > historyMaxSize) historyEnergy.erase(historyEnergy.begin());
  }

  void addParticle(float x, float y, float z) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
    std::vector<float> data = activeBuffer.getData();

    
    for (int i = 0; i < params.maxParticles; i++) {
      int base = i * PARTICLE_FLOATS;
      if (data[base + 6] < 0.01f) {  
        std::uniform_real_distribution<float> speciesDist(0.0f, 3.0f);
        std::uniform_real_distribution<float> dnaDist(-0.2f, 0.2f);

        data[base + 0] = x;                 
        data[base + 1] = y;                 
        data[base + 2] = z;                 
        data[base + 3] = 0.0f;              
        data[base + 4] = 0.0f;              
        data[base + 5] = 0.0f;              
        data[base + 6] = 1.0f;              
        data[base + 7] = speciesDist(rng);  
        data[base + 8] = 0.0f;              
        for (int d = 0; d < 5; d++) {
          data[base + 9 + d] = dnaDist(rng);
        }

        activeBuffer.setData(data);
        
        Buffer& otherBuffer = useBufferA ? particleBufferB : particleBufferA;
        otherBuffer.setData(data);

        aliveCount++;
        break;
      }
    }
  }

  void spawnOrbium(float x, float y, float z) {
    
    int count = 40;
    float radius = 3.0f;

    for (int i = 0; i < count; i++) {
      std::uniform_real_distribution<float> dist(-radius, radius);
      addParticle(x + dist(rng), y + dist(rng), z + dist(rng));
    }
  }

  void applyForce(float x, float y, float z, float strength, float radius) {
    Buffer& activeBuffer = useBufferA ? particleBufferA : particleBufferB;
    std::vector<float> data = activeBuffer.getData();

    for (int i = 0; i < params.maxParticles; i++) {
      int base = i * PARTICLE_FLOATS;
      if (data[base + 6] > 0.01f) {
        float dx = data[base + 0] - x;
        float dy = data[base + 1] - y;
        float dz = data[base + 2] - z;
        float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 < radius * radius) {
          float dist = sqrt(dist2);
          float force = strength * (1.0f - dist / radius);
          if (dist > 0.001f) {
            data[base + 3] += (dx / dist) * force;
            data[base + 4] += (dy / dist) * force;
            data[base + 5] += (dz / dist) * force;
          }
        }
      }
    }
    activeBuffer.setData(data);
    Buffer& otherBuffer = useBufferA ? particleBufferB : particleBufferA;
    otherBuffer.setData(data);
  }
};


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

  
  float rotSpeed = 2.0f;
  float angleSpeed = 1.0f;

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    simulation.params.cameraAngle =
        std::min(89.0f, simulation.params.cameraAngle + angleSpeed);
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    simulation.params.cameraAngle =
        std::max(5.0f, simulation.params.cameraAngle - angleSpeed);
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    simulation.params.cameraRotation =
        std::fmod(simulation.params.cameraRotation - rotSpeed + 360.0f, 360.0f);
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    simulation.params.cameraRotation =
        std::fmod(simulation.params.cameraRotation + rotSpeed, 360.0f);
  }

  
  float zoomSpeed = 1.0f;
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
    simulation.params.cameraDistance =
        std::max(10.0f, simulation.params.cameraDistance - zoomSpeed);
  }
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
    simulation.params.cameraDistance =
        std::min(200.0f, simulation.params.cameraDistance + zoomSpeed);
  }
}

ImVec2 screenToWorld(float screenX, float screenY) {
  float windowAspect =
      static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);
  float worldAspect =
      simulation.params.worldWidth / simulation.params.worldHeight;

  
  float uvX = (screenX / WINDOW_WIDTH - 0.5f) * 2.0f;
  float uvY = ((1.0f - screenY / WINDOW_HEIGHT) - 0.5f) * 2.0f;

  
  if (windowAspect > worldAspect) {
    uvX *= windowAspect / worldAspect;
  } else {
    uvY *= worldAspect / windowAspect;
  }

  
  float worldX =
      uvX * (simulation.params.worldWidth * 0.5f) / simulation.params.zoom +
      simulation.params.translateX;
  float worldY =
      uvY * (simulation.params.worldHeight * 0.5f) / simulation.params.zoom +
      simulation.params.translateY;
  return ImVec2(worldX, worldY);
}


void drawChronosIcon(ImVec2 pos, float size) {
  ImDrawList* draw = ImGui::GetWindowDrawList();

  
  float s = size / 100.0f;

  
  ImU32 blue   = IM_COL32(52, 152, 219, 255);   
  ImU32 orange = IM_COL32(230, 126, 34, 255);   
  ImU32 green  = IM_COL32(46, 204, 113, 255);   
  ImU32 gray   = IM_COL32(85, 85, 85, 255);     

  
  ImVec2 c1(pos.x + 30*s, pos.y + 30*s);  
  ImVec2 c2(pos.x + 70*s, pos.y + 30*s);  
  ImVec2 c3(pos.x + 50*s, pos.y + 70*s);  
  float r = 12*s;  

  
  draw->AddCircleFilled(c1, r, blue, 24);
  draw->AddCircleFilled(c2, r, orange, 24);
  draw->AddCircleFilled(c3, r, green, 24);

  
  float lineWidth = 2.5f * s;

  
  ImVec2 a1_start(pos.x + 42*s, pos.y + 25*s);
  ImVec2 a1_end(pos.x + 58*s, pos.y + 25*s);
  draw->AddLine(a1_start, a1_end, gray, lineWidth);
  
  draw->AddTriangleFilled(
    ImVec2(a1_end.x + 4*s, a1_end.y),
    ImVec2(a1_end.x - 3*s, a1_end.y - 4*s),
    ImVec2(a1_end.x - 3*s, a1_end.y + 4*s),
    gray
  );

  
  ImVec2 a2_start(pos.x + 72*s, pos.y + 44*s);
  ImVec2 a2_end(pos.x + 62*s, pos.y + 60*s);
  draw->AddLine(a2_start, a2_end, gray, lineWidth);
  
  draw->AddTriangleFilled(
    ImVec2(a2_end.x - 2*s, a2_end.y + 5*s),
    ImVec2(a2_end.x + 5*s, a2_end.y - 2*s),
    ImVec2(a2_end.x - 5*s, a2_end.y - 2*s),
    gray
  );

  
  ImVec2 a3_start(pos.x + 40*s, pos.y + 60*s);
  ImVec2 a3_end(pos.x + 32*s, pos.y + 44*s);
  draw->AddLine(a3_start, a3_end, gray, lineWidth);
  
  draw->AddTriangleFilled(
    ImVec2(a3_end.x - 2*s, a3_end.y - 5*s),
    ImVec2(a3_end.x - 5*s, a3_end.y + 2*s),
    ImVec2(a3_end.x + 5*s, a3_end.y + 2*s),
    gray
  );
}

void renderUI() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  
  ImGuiStyle& style = ImGui::GetStyle();
  style.Colors[ImGuiCol_Text] = ImVec4(0.90f, 0.90f, 0.90f, 1.00f);
  style.Colors[ImGuiCol_WindowBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
  style.Colors[ImGuiCol_Header] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
  style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
  style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
  style.Colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
  style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
  style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.45f, 0.45f, 0.45f, 1.00f);
  style.Colors[ImGuiCol_TitleBg] = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
  style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
  style.Colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
  style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
  style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
  style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
  style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.70f, 0.70f, 0.70f, 1.00f);
  style.Colors[ImGuiCol_CheckMark] = ImVec4(0.70f, 0.70f, 0.70f, 1.00f);
  
  ImGuiIO& io = ImGui::GetIO();
  float topBarHeight = 60.0f;

  
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, topBarHeight));
  ImGui::Begin("TopBar", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);

  
  float iconSize = 50.0f;
  ImVec2 iconPos = ImGui::GetCursorScreenPos();
  iconPos.y += (topBarHeight - iconSize) * 0.5f - 5.0f;  
  drawChronosIcon(iconPos, iconSize);
  ImGui::Dummy(ImVec2(iconSize + 10, 0));  
  ImGui::SameLine();

  
  if (paused) {
      if (ImGui::Button(" PLAY ", ImVec2(0, 30))) paused = !paused;
  } else {
      if (ImGui::Button(" PAUSE ", ImVec2(0, 30))) paused = !paused;
  }
  ImGui::SameLine();
  
  
  if (ImGui::Button("Restart", ImVec2(0, 30))) {
    simulation.resetParticles();
  }
  ImGui::SameLine();
  
  
  static char sceneFilename[128] = "scene.txt";
  ImGui::PushItemWidth(150);
  ImGui::InputText("##file", sceneFilename, 128);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  if (ImGui::Button("Load", ImVec2(0, 30))) simulation.loadScene(sceneFilename);
  ImGui::SameLine();
  if (ImGui::Button("Save", ImVec2(0, 30))) simulation.saveScene(sceneFilename);
  
  ImGui::SameLine(); ImGui::Text(" | "); ImGui::SameLine();
  
  
  ImGui::Text("Sim Speed:");
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderInt("##speed", &simulation.params.stepsPerFrame, 1, 50, "%d/frame");
  ImGui::PopItemWidth();
  
  ImGui::SameLine(); ImGui::Text(" | "); ImGui::SameLine();
  
  
  ImGui::Text("Particles: %d", simulation.aliveCount);
  ImGui::SameLine();
  ImGui::Text("FPS: %.1f", io.Framerate);
  
  ImGui::End();

  
  ImGui::SetNextWindowPos(ImVec2(0, topBarHeight));
  ImGui::SetNextWindowSize(ImVec2(350, io.DisplaySize.y - topBarHeight));
  ImGui::Begin("Sidebar", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

  
  if (ImGui::CollapsingHeader("Environment Settings",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::TextDisabled("World Dimensions (3D Cube)");
    ImGui::PushItemWidth(120);
    ImGui::DragFloat("Arena Width", &simulation.params.worldWidth, 0.5f, 10.0f,
                     100.0f);
    ImGui::DragFloat("Arena Height", &simulation.params.worldHeight, 0.5f,
                     10.0f, 100.0f);
    ImGui::DragFloat("Arena Depth", &simulation.params.worldDepth, 0.5f, 10.0f,
                     100.0f);
    ImGui::PopItemWidth();

    int numParticles = simulation.params.numParticles;
    if (ImGui::DragInt("Spawn Count", &numParticles, 5, 10, 1000)) {
      simulation.params.numParticles = numParticles;
    }
  }

  
  if (ImGui::CollapsingHeader("Interaction Tools", ImGuiTreeNodeFlags_DefaultOpen)) {
      const char* modes[] = { "Navigation Only", "Paint Particles", "Repel Force", "Attract Force", "Spawn Orbium", "Spawn Cancer" };
      ImGui::Combo("Tool", &simulation.params.interactionMode, modes, 6);
      
      if (simulation.params.interactionMode > 0) {
          ImGui::Indent();
          ImGui::DragFloat("Brush Radius", &simulation.params.brushRadius, 0.5f, 1.0f, 50.0f);
          if (simulation.params.interactionMode == 2 || simulation.params.interactionMode == 3) {
             ImGui::DragFloat("Force Strength", &simulation.params.forceStrength, 0.01f, 0.0f, 5.0f);
          }
          if (simulation.params.interactionMode == 5) {
             ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Cancer: Predatory Cells");
          }
          ImGui::Unindent();
          ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.0f, 1.0f), "Hold Left Click to use tool");
      }
  }

  
  if (ImGui::CollapsingHeader("Physics Parameters")) {
    ImGui::TextDisabled("Perception Kernel");
    ImGui::DragFloat("Sensitivity (w_k)", &simulation.params.w_k, 0.001f, 0.001f,
                     0.1f, "%.4f");
    ImGui::DragFloat("Optimal Range (mu_k)", &simulation.params.mu_k, 0.1f, 0.5f,
                     20.0f);
    ImGui::DragFloat("Variance (sigma_k)", &simulation.params.sigma_k2, 0.05f,
                     0.1f, 10.0f);

    ImGui::Spacing();
    ImGui::TextDisabled("Forces");
    ImGui::DragFloat("Repulsion (c_rep)", &simulation.params.c_rep, 0.1f, 0.0f,
                     5.0f);
  }

  
  if (ImGui::CollapsingHeader("Growth Dynamics")) {
    ImGui::DragFloat("Target Density (mu_g)", &simulation.params.mu_g, 0.01f, 0.0f,
                     2.0f);
    ImGui::DragFloat("Tolerance (sigma_g)", &simulation.params.sigma_g2, 0.001f,
                     0.001f, 0.5f, "%.4f");
  }

  
  if (ImGui::CollapsingHeader("Time & Space")) {
    ImGui::DragFloat("Delta Time (dt)", &simulation.params.dt, 0.01f, 0.01f,
                     0.5f);
    ImGui::DragFloat("Space Step (h)", &simulation.params.h, 0.001f, 0.001f,
                     0.1f, "%.4f");
  }

  
  if (ImGui::CollapsingHeader("Evolution")) {
    ImGui::Checkbox("Enable Evolution", &simulation.params.evolutionEnabled);

    if (simulation.params.evolutionEnabled) {
      ImGui::Spacing();
      ImGui::TextDisabled("Population");
      ImGui::DragFloat("Birth Rate", &simulation.params.birthRate,
                       0.0001f, 0.0f, 0.01f, "%.5f");
      ImGui::DragFloat("Death Rate", &simulation.params.deathRate,
                       0.0001f, 0.0f, 0.01f, "%.5f");

      ImGui::Spacing();
      ImGui::TextDisabled("Genetics");
      ImGui::DragFloat("Mutation Rate", &simulation.params.mutationRate,
                       0.01f, 0.0f, 0.5f);

      ImGui::Spacing();
      ImGui::TextDisabled("Metabolism");
      ImGui::DragFloat("Energy Decay", &simulation.params.energyDecay,
                       0.0001f, 0.0f, 0.01f, "%.5f");
      ImGui::DragFloat("Energy Gain", &simulation.params.energyFromGrowth,
                       0.001f, 0.0f, 0.1f);
    }
  }

  
  if (ImGui::CollapsingHeader("Food System")) {
    ImGui::Checkbox("Enable Food", &simulation.params.foodEnabled);

    if (simulation.params.foodEnabled) {
      ImGui::Checkbox("Show Food Grid", &simulation.params.showFood);
      ImGui::DragFloat("Spawn Rate", &simulation.params.foodSpawnRate, 0.0001f,
                       0.0f, 0.01f, "%.4f");
      ImGui::DragFloat("Decay Rate", &simulation.params.foodDecayRate, 0.0001f,
                       0.0f, 0.01f, "%.4f");
      ImGui::DragFloat("Max Food", &simulation.params.foodMaxAmount, 0.1f,
                       0.1f, 5.0f);
    }
  }

  
  if (ImGui::CollapsingHeader("Goal/Target")) {
    bool changed = false;
    const char* goalModes[] = {"None", "Circle", "Box", "Text 'HI'",
                               "Image (BMP)"};
    if (ImGui::Combo("Pattern", &simulation.params.goalMode, goalModes,
                     5)) {
      changed = true;
    }

    if (simulation.params.goalMode == 4) {
      ImGui::InputText("BMP File", simulation.params.goalImagePath, 256);
      if (ImGui::Button("Reload Image")) {
        changed = true;
      }
    }

    ImGui::DragFloat("Attraction", &simulation.params.goalStrength,
                     0.01f, 0.0f, 2.0f);

    if (changed) {
      simulation.updateGoalTexture();
    }
  }

  
  if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("3D Render", &simulation.params.view3D);

    if (simulation.params.view3D) {
      ImGui::Indent();
      ImGui::DragFloat("Camera Dist", &simulation.params.cameraDistance,
                       1.0f, 10.0f, 200.0f);
      ImGui::DragFloat("Particle Size", &simulation.params.particleSize, 1.0f,
                       1.0f, 50.0f);
      ImGui::DragFloat("Glow", &simulation.params.glowIntensity, 0.1f,
                       0.0f, 3.0f);
      ImGui::Unindent();
    } else {
      ImGui::Checkbox("Fields Overlay", &simulation.params.showFields);
      const char* fieldModes[] = {"Off", "Density", "Separation",
                                  "Growth", "Energy"};
      ImGui::Combo("Field Type", &simulation.params.fieldType, fieldModes, 5);
    }

    ImGui::DragFloat("Zoom", &simulation.params.zoom, 0.05f, 0.1f, 5.0f);
  }

  
  if (ImGui::CollapsingHeader("Sonification")) {
    ImGui::Checkbox("Enable Audio", &simulation.params.sonificationEnabled);

    if (simulation.params.sonificationEnabled) {
      ImGui::Indent();

      if (g_audio.initialized) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Audio: Active");
      } else {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Audio: Not Available");
      }

      ImGui::Spacing();
      ImGui::TextDisabled("Volume & Voices");
      ImGui::DragFloat("Master Volume", &simulation.params.audioVolume, 0.01f, 0.0f, 1.0f);
      ImGui::DragInt("Voice Count", &simulation.params.maxVoices, 1, 1, 64);

      ImGui::Spacing();
      ImGui::TextDisabled("Frequency Range");
      ImGui::DragFloat("Min Frequency", &simulation.params.minFrequency, 5.0f, 20.0f, 500.0f, "%.0f Hz");
      ImGui::DragFloat("Max Frequency", &simulation.params.maxFrequency, 10.0f, 200.0f, 2000.0f, "%.0f Hz");

      ImGui::Spacing();
      ImGui::TextDisabled("Mapping: Potential -> Frequency, Speed -> Volume");

      ImGui::Unindent();
    }
  }

  ImGui::Separator();
  ImGui::TextDisabled("Controls: WASD=Cam | Q/E=Zoom");
  ImGui::TextDisabled("Mouse: Left Click to Interact");

  
  float avail = ImGui::GetContentRegionAvail().y;
  if (avail > 160) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + avail - 160);
  
  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "LIVE ANALYTICS");
  
  if (!simulation.historyAlive.empty()) {
      char overlay[32];
      
      sprintf(overlay, "Pop: %d", (int)simulation.historyAlive.back());
      ImGui::PlotLines("", simulation.historyAlive.data(), (int)simulation.historyAlive.size(), 0, overlay, 0.0f, (float)simulation.params.maxParticles, ImVec2(ImGui::GetContentRegionAvail().x, 60));
      
      sprintf(overlay, "Avg Energy: %.2f", simulation.historyEnergy.back());
      ImGui::PlotLines("", simulation.historyEnergy.data(), (int)simulation.historyEnergy.size(), 0, overlay, 0.0f, 1.0f, ImVec2(ImGui::GetContentRegionAvail().x, 60));
  } else {
      ImGui::TextDisabled("Collecting data...");
  }

  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main() {
  
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);  

  GLFWwindow* window =
      glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                       "Chronos - Particle Lenia Evolution", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwSwapInterval(1);  

  
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450");

  
  simulation.init();
  simulation.init3D();

  
  initAudio();

  
  static ImVec2 panStart;
  static float panStartX, panStartY;

  
  while (!glfwWindowShouldClose(window)) {
    processInput(window);
    glfwPollEvents();

    
    if (!io.WantCaptureMouse) {
      
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
        panStart = ImGui::GetMousePos();
        panStartX = simulation.params.translateX;
        panStartY = simulation.params.translateY;
      }
      if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        ImVec2 pos = ImGui::GetMousePos();
      
      
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          ImVec2 mousePos = ImGui::GetMousePos();
          ImVec2 worldPos = screenToWorld(mousePos.x, mousePos.y);
          
          if (simulation.params.interactionMode == 1) { 
              
              for(int k=0; k<5; k++) {
                  std::uniform_real_distribution<float> dist(-simulation.params.brushRadius, simulation.params.brushRadius);
                  float rx = dist(simulation.rng);
                  float ry = dist(simulation.rng);
                  
                  if (rx*rx + ry*ry <= simulation.params.brushRadius*simulation.params.brushRadius) {
                      
                      float rz = dist(simulation.rng) * 0.1f; 
                      simulation.addParticle(worldPos.x + rx, worldPos.y + ry, rz);
                  }
              }
          }
          else if (simulation.params.interactionMode == 2) { 
              simulation.applyForce(worldPos.x, worldPos.y, 0.0f, simulation.params.forceStrength, simulation.params.brushRadius);
          }
          else if (simulation.params.interactionMode == 3) { 
              simulation.applyForce(worldPos.x, worldPos.y, 0.0f, -simulation.params.forceStrength, simulation.params.brushRadius);
          }
          else if (simulation.params.interactionMode == 4) { 
              
          }
      }
      
      
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
           ImVec2 mousePos = ImGui::GetMousePos();
           ImVec2 worldPos = screenToWorld(mousePos.x, mousePos.y);
           
           if (simulation.params.interactionMode == 4) { 
               simulation.spawnOrbium(worldPos.x, worldPos.y, 0.0f);
           }
      }
        float dx = (pos.x - panStart.x) / WINDOW_WIDTH *
                   simulation.params.worldWidth * 2.0f / simulation.params.zoom;
        float dy = (pos.y - panStart.y) / WINDOW_HEIGHT *
                   simulation.params.worldHeight * 2.0f /
                   simulation.params.zoom;
        simulation.params.translateX = panStartX - dx;
        simulation.params.translateY = panStartY + dy;
      }

      
      float scroll = io.MouseWheel;
      if (scroll != 0) {
        simulation.params.zoom *= (1.0f + scroll * 0.1f);
        simulation.params.zoom =
            std::max(0.1f, std::min(10.0f, simulation.params.zoom));
      }
      
      
      if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          ImVec2 mousePos = ImGui::GetMousePos();
          ImVec2 worldPos = screenToWorld(mousePos.x, mousePos.y);
          
          if (simulation.params.interactionMode == 1) { 
              for(int k=0; k<5; k++) {
                  std::uniform_real_distribution<float> dist(-simulation.params.brushRadius, simulation.params.brushRadius);
                  float rx = dist(simulation.rng);
                  float ry = dist(simulation.rng);
                  if (rx*rx + ry*ry <= simulation.params.brushRadius*simulation.params.brushRadius) {
                      float rz = dist(simulation.rng) * 0.1f;
                      simulation.addParticle(worldPos.x + rx, worldPos.y + ry, rz);
                  }
              }
          }
          else if (simulation.params.interactionMode == 2) { 
              simulation.applyForce(worldPos.x, worldPos.y, 0.0f, simulation.params.forceStrength, simulation.params.brushRadius);
          }
          else if (simulation.params.interactionMode == 3) { 
              simulation.applyForce(worldPos.x, worldPos.y, 0.0f, -simulation.params.forceStrength, simulation.params.brushRadius);
          }
      }
      
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
           ImVec2 mousePos = ImGui::GetMousePos();
           ImVec2 worldPos = screenToWorld(mousePos.x, mousePos.y);
           if (simulation.params.interactionMode == 4) { 
               simulation.spawnOrbium(worldPos.x, worldPos.y, 0.0f);
           }
      }
    }

    
    if (!paused) {
      for (int i = 0; i < simulation.params.stepsPerFrame; i++) {
        simulation.step();
      }

      
      static int frameCount = 0;
      if (++frameCount % 10 == 0) {
        simulation.updateStats();

        
        if (simulation.params.sonificationEnabled && g_audio.initialized) {
          g_audio.enabled = true;
          g_audio.numVoices = simulation.params.maxVoices;
          Buffer& activeBuffer = simulation.useBufferA ? simulation.particleBufferA : simulation.particleBufferB;
          std::vector<float> data = activeBuffer.getData();
          updateAudioFromParticles(data, simulation.params.maxParticles,
                                    simulation.params.minFrequency,
                                    simulation.params.maxFrequency,
                                    simulation.params.audioVolume);
        } else {
          g_audio.enabled = false;
        }
      }
    }

    
    if (simulation.params.view3D) {
      
      glClearColor(0.01f, 0.03f, 0.06f, 1.0f);
    } else {
      glClearColor(0.0f, 0.02f, 0.05f, 1.0f);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (simulation.params.view3D) {
      simulation.display3D(WINDOW_WIDTH, WINDOW_HEIGHT);
    } else {
      simulation.display(WINDOW_WIDTH, WINDOW_HEIGHT);
    }

    renderUI();

    glfwSwapBuffers(window);
  }

  
  shutdownAudio();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
