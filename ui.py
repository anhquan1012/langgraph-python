from dotenv import load_dotenv
load_dotenv()
import gradio as gr

from langchain_openai import ChatOpenAI

#API Keys
model = ChatOpenAI(model='gpt-4o-mini')

def chat(system_prompt, user_prompt, temperature = 0, verbose = False):
    ''' Normal call of OpenAI API '''
    response = model(
    temperature = temperature,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    res = response.content
    
    if verbose:
        print('System prompt:', system_prompt)
        print('User prompt:', user_prompt)
        print('GPT response:', res)
        
    return res

def format_chat_prompt(message, chat_history, max_convo_length):
    prompt = ""
    for turn in chat_history[-max_convo_length:]:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history, max_convo_length = 10):
        formatted_prompt = format_chat_prompt(message, chat_history, max_convo_length)
        print('Prompt + Context:')
        print(formatted_prompt)
        bot_message = chat(system_prompt = 'You are a friendly chatbot. Generate the output for only the Assistant.',
                           user_prompt = formatted_prompt)
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=300) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
gr.close_all()
demo.launch()