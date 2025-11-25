# Master Baiter
The best futuristic robot to help you daily. Master the bait with Master Baiter.

## Overview
Master Baiter is an intelligent autonomous robot built on the TurtleBot3 platform using ROS2. Combining AI-powered speech recognition and computer vision, this robot can navigate environments using voice commands and autonomously approach recognized individuals.

## Features
- **Voice-Controlled Navigation**: Navigate your robot using natural voice commands like "Left", "Right", "Forward", "Stop"
- **Person Recognition**: Built-in camera system with AI-powered facial/person recognition
- **Autonomous Approach**: Automatically navigates toward recognized individuals
- **AI Integration**: Leverages machine learning for speech and vision processing
- **ROS2 Architecture**: Built on modern ROS2 framework for robust robotics development

## Hardware Requirements
- TurtleBot3 (Burger)
- Camera module (compatible with ROS2)
- Microphone for voice input
- LiDAR sensor (included with TurtleBot3)

## Software Requirements
- Ubuntu 22.04 (Jammy Jellyfish)
- ROS2 Humble Hawksbill
- Python 3.10+

## System Architecture
```
┌─────────────────┐
│  Voice Input    │
│  (Microphone)   │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│Speech Recognition│
│    (AI Model)    │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Command Parser  │────▶│  Vision System  │
│                 │     │   (Camera)      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │Person Recognition│
         │              │    (AI Model)    │
         │              └────────┬─────────┘
         │                       │
         ▼                       ▼
┌──────────────────────────────────┐
│      Navigation Controller       │
│         (ROS2 Nav2)              │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│        TurtleBot3 Base           │
└──────────────────────────────────┘
```

## The Master Baiter team

Elian Delmas
Antoine Desmartin
Logan Lucas
Valentin Quiot

---
*This README has been written using an LLM for a better structure*
