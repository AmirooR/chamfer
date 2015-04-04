# Chamfer
Compute Chamfer distance between two edge images and then applies gradient
descent optimization to deform the contour points to move toward edges

**Compile:**

```bash
g++ *.cpp `pkg-config --cflags --libs opencv`
```

or use **make** command
