#include "ChronosApp.h"

int main() {
  ChronosApp app;

  if (!app.init(1920, 1080, "Lenia 3D - CUDA Emergent Life")) {
    return -1;
  }

  app.run();
  app.shutdown();

  return 0;
}

