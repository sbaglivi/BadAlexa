# Alexa

## What and why

I own a device from Amazon with Alexa. Up until just a couple years ago, it felt almost incredible to have a device that responded to voice commands for such an affordable price.  
So much so, that I always suspected or feared that the device might listen in more than it needed to. Was my fear justified? I don't know.  
Point is, that with the technologies available today it feels quite reasonable to build a personalized Alexa. 
In doing so, we benefit from guaranteed privacy (we can make it work offline!) and access to data or tools that the real one does not have.  
So as too often programmers do, I asked myself: Why not?

## How

I wanted the result of this project to be able to run on accessible hardware (WIP). I therefore decided to not take the shortcut of using a voice capable LLM through an API.
The plan is pretty straightforward: use one of the readily available Text -> Text decoder models that can be ran locally. Sandwhich in between a ML model responsible of transcribing voice,
and one resposible for vocalizing the response from the LLM.
We want the model to always be ready to response, we don't want to, for example, press a button before asking it something. 
There are 2 key pieces that will come in handy:
- a VAD (Voice Activity Detector) that will tell us whether at the current time someone is speaking (we don't want to respond to noise)
- a wake word: the role normally played by "Alexa". It's a word that tells the model: "Hey, I'm talking to you!"

## Technologies

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): the ML model responsible for transcribing the commands we speak. Faster implementation of [whisper](https://github.com/openai/whisper) from OpenAI
- [piper-tts](https://github.com/OHF-Voice/piper1-gpl): ML model responsible for transcription
- [openwakeword](https://github.com/dscripka/openWakeWord): another ML model, responsible for recognizing whether in a small window of audio we invoked the wake word we selected (e.g. Alexa)
- [llama3.1:8b](https://ollama.com/library/llama3.1:8b): the LLM model I chose as the brain of the operation. Ollama and llama are not included in the repo, but they're what I'm communicating with through HTTPX on port 11434.
- [sounddevice](https://python-sounddevice.readthedocs.io/en/0.5.1/): the Python library I chose to record audio.

## Current state and TODOs:

Works on my machine™. Jokes aside, this is missing a getting started section, which will be added.  
Right now the source code relies on the user having Ollama installed and running and on having downloaded an openwakeword model (in my case "hey_mycroft_v0.1.onnx") and a vocalizer (piper) with a chosen voice (in my case "en_US-amy-medium.onnx").

TODO:
- create a guide to get it to run
- add barge in feature (allow user to interrupt the model response)
- add debug feature that logs conversation turns and flow locally