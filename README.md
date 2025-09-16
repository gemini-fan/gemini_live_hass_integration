# Gemini Live WebRTC React Native App with Signalling
Are you frustrated with all those paid api keys or plan? What the hell PipeCat by Daily? (Restrict my plan to 168 hours per month?) I want my super intelligence AI to be slaved 24/7 and no bill!
> Reference News
> https://developers.googleblog.com/en/gemini-2-0-level-up-your-apps-with-real-time-multimodal-interactions/

<img src="./public/gemini-webrtc.png" />

### Why Choosing Gemini Live Api Plus Webrtc?
- **Fast Response Time < 0.5s** : Almost identical to talking or response to people in real
- **Integrated VAD**: Voice Activity Detection on gemini socket fast and accurate
- **Interrupt detection**: The response stream can be interrupted through VAD 
- **Free to use** : The live model can has at most 3 concurrent session lasting forever
- **Echo Cancellation**: The speaker and microphone will not interfere
- **Automatic Gain Control (AGC)**: Adjust the volume and output (from client side)
- **Noise Suppression**: Reduce background noise
- **Tool Use**: Both synchronous and asynchronous function calling is available
- **[NEW] Open Wake Word**: Can wake up anytime or switch into active standby mode

```

                                 ┌─────────────────────────────────────────┐     
                                 │                                         │     
                                 │ Server                                  │     
                                 │                                         │     
                                 │                                         │     
                                 │                 ┌────────────────────┐  │     
                                 │                 │                    │  │     
                                 │                 │  Custom            │  │     
                                 │                 │  Pipeline          │  │     
                                 │                 │                    │  │     
                                 │                 │                    │  │     
┌──────────────────────────┐     │                 │  Audio Processing  │  │     
│                          │     │                 │         ▼          │  │     
│       React Client       │     │   ┌─────────────│   Gemini Flash    ─┼──┼────►
│    ┌───────────────┐     │     │   │             │   Transcription   ◄┼──┼─────
│    │ WebRTC (Clean)│ ────┼────────►│   WebRTC    │         ▼          │  │     
│    │   Transport   │ ◄───┼─────────│  Transport  │  Gemini Multimodal─┼──┼────►
│    └───────────────┘     │     │   │             │     Live API      ◄┼──┼─────
│                          │     │   └─────────────│         ▼          │  │     
└──────────────────────────┘     │                 │   Gemini Flash    ─┼──┼────►
                                 │                 │   Transcription   ◄┼──┼─────
                                 │                 │         ▼          │  │     
                                 │                 │   Conversation     │  │     
                                 │                 │     Context        │  │     
                                 │                 │    Management      │  │     
                                 │                 │         ▼          │  │     
                                 │                 │   RTVI Events      │  │     
                                 │                 │                    │  │     
                                 │                 └────────────────────┘  │     
                                 │                                         │     
                                 └─────────────────────────────────────────┘  
```

---

## Roadmap
- [x] Establish client <--> gemini client <--> gemini websocket connection for 24/7
- [x] Standalone docker installation
- [x] Home Assistant Integration

## How To Use?
This project is compose of 3 parts (android client, signalling server & home assistant integration). 
You may follow the guide to setup 3 of them separately.

### Server Setup
Go to https://github.com/gemini-fan/gemini_signalling_server.git

### Client App Setup
Go to https://github.com/gemini-fan/react_native_webrtc_client.git

### Home Assistant Setup
#### Manual installation
1. Go to HA custom_components directory and clone the repo 
```
git clone https://github.com/gemini-fan/gemini_live_hass_integration.git
```
2. Restart home assistant instance
3. Search Gemini Live Conversation at the integration section
4. Enter the signalling server url and gemini api key
5. Call your gemini via caller id 666666 (default) from your client app
6. Happy chatting :D





