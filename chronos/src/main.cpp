#include "ChronosApp2D.h"

int main() {
  ChronosApp2D app;

  if (!app.init(1280, 720, "Flow Lenia - Emergent Life")) {
    return -1;
  }

  app.run();
  app.shutdown();

  return 0;
}
