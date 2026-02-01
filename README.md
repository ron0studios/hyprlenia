# hyprlenia

A hyper-efficient sandbox for studying 3D emergent behaviours

# about

> This project was made for Marshall Wace's track: "Emergent Behaviour
> Design a system that evolves or adapts through interactions with users, agents or other systems to produce something greater than the sum of its parts"

This is a [particle Lenia](https://google-research.github.io/self-organising-systems/particle-lenia/) simulation used to showcase emergence of realistic patterns such as biological cells. It has also been shown to be applicable in fields such as physics and chemistry. This is due to the nature of Lenia: changing constraints is trivial. Hence, this allows us to generate images below:

<img src="img/main.png" alt="main" width="500"/>
<img src="img/ichack.gif" alt="ichack_logo" width="400"/>
<img src="img/circle.gif" alt="circle_gif" width="500"/>

These constraints can be dynamically applied and varied to achieve very interesting and visually appealing results

# compilation

```bash
cd hyprlenia && cmake -B build .
cmake --build build
```

use with a gpu, or use with caution.

# credits

[opengl](https://www.opengl.org/)
[imgui](https://github.com/ocornut/imgui)
[leniabreeder](https://leniabreeder.github.io/)
[various research papers](https://arxiv.org/pdf/2505.15998)
