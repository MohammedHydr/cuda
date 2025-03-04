#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

enum PrintColor { NONE, GREEN, DGREEN, CYAN };

typedef struct {
#ifdef _WIN32
    LARGE_INTEGER startTime;
    LARGE_INTEGER endTime;
    LARGE_INTEGER frequency;
#else
    struct timeval startTime;
    struct timeval endTime;
#endif
} Timer;

static void startTime(Timer* timer) {
#ifdef _WIN32
    QueryPerformanceFrequency(&timer->frequency);
    QueryPerformanceCounter(&timer->startTime);
#else
    gettimeofday(&(timer->startTime), NULL);
#endif
}

static void stopTime(Timer* timer) {
#ifdef _WIN32
    QueryPerformanceCounter(&timer->endTime);
#else
    gettimeofday(&(timer->endTime), NULL);
#endif
}

static void printElapsedTime(Timer timer, const char* s, enum PrintColor color = NONE) {
    float t;
#ifdef _WIN32
    t = (float)((timer.endTime.QuadPart - timer.startTime.QuadPart) * 1000.0 / timer.frequency.QuadPart);
#else
    t = ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6)) * 1e3;
#endif

#ifndef _WIN32
    switch(color) {
        case GREEN:  printf("\033[1;32m"); break;
        case DGREEN: printf("\033[0;32m"); break;
        case CYAN :  printf("\033[1;36m"); break;
    }
#endif

    printf("%s: %f ms\n", s, t);

#ifndef _WIN32
    if(color != NONE) {
        printf("\033[0m");
    }
#endif
}

#endif
