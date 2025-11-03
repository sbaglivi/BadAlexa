# BadAlexa

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

## Usage

**NB: in its current state, the app requires a local LLM available through Ollama. Also, right now if you choose a different LLM, piper voice or wakeword, you need to modify the references in the code, this will be solved**
- install Ollama and a model of choice (I'm constrained by resources and I'm currently trying `openchat:7b`)
- install Python (I'm currently using v3.12)
- (not necessary but stronly recommended) create a virtual env and activate it `python3 -m venv venv && source venv/bin/activate`
- install dependencies `pip3 install -r requirements.txt`
- download a voice for piper `python3 -m piper.download_voices en_US-amy-medium`. If you prefer a different voice you can just modify the name. You can sample them [here](https://rhasspy.github.io/piper-samples/)
- cross your fingers and run `python3 main.py`.  

The model will be idle until it detects the default wake word `Hey mycroft`.  
Once it does, it will listen for a query until it detects a silence window and then feed the query to the LLM.  
The LLM will stream its response on a queue, from which piper will retrieve chunks of text and vocalize them.  
If during the LLM response you start speaking, the model will stop and listen for additional input, before producing an updated response.  
After every response, the model keeps listening for a few seconds to see if you have a follow up request, otherwise it goes back to idle and requires the wakeword again to activate.

## Todo:

- sometimes after the wakeword the model goes to conversation with segments of audio that are empty (and it detects "thank you"?). 
  I think there should be some sort of filter that helps it understand that it does not have enough input to work on yet and a timeout for no request arrives.
- add debug feature that logs conversation turns and flow locally