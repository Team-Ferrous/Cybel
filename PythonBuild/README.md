# CYBEL (aka automat-llm)

Mobile AI Assistant

## How To Run

To run this demo first ust `pip install -r requirements.txt` to install basic packages from pip. Then you will have to optionally provide a `--use-dia` flag when running py main.py for voice interaction if desired.
the use of dia is entirely optional and requirements should demonstrate the basic demo. The use of HuggingFace CLI has been entirely deprecated for Groq. Users will need to provide a Weaviate and Groq API key to use

this LLM and preferably the keys should be their own environment variables.

After running pip simply using `python main.py` in your preferred Terminal, it should work. If any Library issues come up contact Sasori Zero Labs (mileslitteral@sasorizerolabs.com)

In order to run the AI along with the server use `python3 -m uvicorn api:app --reload --port 8000` and to run the frontend just do `npm run start`.

Some users report issues installing Langchain and Dia, this is currently being addressed, the best we can suggest for now is using venv and also being very careful to not have multiple instances of python on your system.
