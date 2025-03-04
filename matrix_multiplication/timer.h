#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <chrono>

enum PrintColor { NONE, GREEN, DGREEN, CYAN };

typedef struct {
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
} Timer;

static void startTime(Timer* timer) {
    timer->startTime = std::chrono::high_resolution_clock::now();
}

static void stopTime(Timer* timer) {
    timer->endTime = std::chrono::high_resolution_clock::now();
}

static void printElapsedTime(Timer timer, const char* s, enum PrintColor color = NONE) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(timer.endTime - timer.startTime);
    float t = duration.count() / 1000.0f;
    
    switch(color) {
        case GREEN:  printf("\033[1;32m"); break;
        case DGREEN: printf("\033[0;32m"); break;
        case CYAN :  printf("\033[1;36m"); break;
    }
    printf("%s: %f ms\n", s, t);
    if(color != NONE) {
        printf("\033[0m");
    }
}

#endif
