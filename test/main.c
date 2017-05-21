

#include <stdio.h>

int main() {

  const int WIDTH = 600;
  const int HEIGHT = 400;
  const int CHANNELS = 3;


  char output[HEIGHT * WIDTH * CHANNELS];

  for (int i = 0; i < WIDTH; ++i) {
    for (int j = 0; j < HEIGHT; ++j) {
      output[CHANNELS * WIDTH * j + i * CHANNELS] = (char) 255;
      output[CHANNELS * WIDTH * j + i * CHANNELS + 1] = (char) 255;
      output[CHANNELS * WIDTH * j + i * CHANNELS + 2] = (char) 0;
    }
  }

  //for (int i = 0; i < WIDTH; ++i) {
  //  for (int j = 0; j < HEIGHT; ++j) {
  //    for (int c = 0; c < CHANNELS; ++c) {
  //      printf("%s ", output[CHANNELS * WIDTH * j + i * CHANNELS + c]);
  //    }
  //  }
  //  printf("\n");
  //}

  
  FILE *fp = fopen("omg.ppm", "wb");
  fprintf(fp, "P6\n#asdf\n600 400\n255\n");
  fwrite(output, HEIGHT * WIDTH * CHANNELS, 1, fp);

  return 0;
}
